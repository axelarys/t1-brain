# memory/warm_layer.py

import faiss
import numpy as np
import time
import hashlib
import os
import pickle
import logging

# Configure logging
logger = logging.getLogger(__name__)

VECTOR_SIZE = 1536  # for OpenAI embeddings
INDEX_FILE = "/root/projects/t1-brain/warm_index.faiss"
META_FILE = "/root/projects/t1-brain/warm_metadata.pkl"

class WarmMemoryCache:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if hasattr(self, "index"):
            return  # Already initialized

        self.index = faiss.IndexFlatL2(VECTOR_SIZE)
        self.metadata = {}
        self.id_to_idx = {}
        self.load_index()
        logger.info("[FAISS] 🧠 WarmMemoryCache initialized")

    def _hash(self, query: str, response: str) -> str:
        """
        Generate a SHA-256 hash from query and response.
        This method is used for deduplication and lookup.
        """
        return hashlib.sha256((query + response).encode()).hexdigest()

    def exists(self, session_id: str, query: str, response: str) -> bool:
        """
        Check if a memory entry already exists in FAISS.
        This method is used by run_consistency_check in session_memory.py.
        
        Args:
            session_id: The session identifier
            query: The query text
            response: The response text
            
        Returns:
            bool: True if the entry exists, False otherwise
        """
        sha = self._hash(query, response)
        return sha in self.id_to_idx

    def add(self, session_id: str, query: str, response: str, embedding: list, access_score: int = 0, last_accessed: float = None):
        """
        Add a new vector to the FAISS index with metadata.
        
        Args:
            session_id: The session identifier
            query: The query text
            response: The response text
            embedding: The embedding vector (list of floats)
            access_score: The access score for the memory
            last_accessed: The timestamp of last access
        """
        try:
            # Convert embedding to numpy array if it's a list
            if isinstance(embedding, list):
                vector = np.array(embedding).astype("float32")
            elif isinstance(embedding, np.ndarray):
                vector = embedding.astype("float32")
            else:
                logger.error(f"[FAISS] ❌ Invalid embedding type: {type(embedding)}")
                return
            
            # Ensure vector is the right shape
            if vector.shape[0] != VECTOR_SIZE:
                logger.error(f"[FAISS] ❌ Invalid vector size: {vector.shape[0]} (expected {VECTOR_SIZE})")
                return
                
            # Calculate SHA hash for deduplication
            sha = self._hash(query, response)

            # Skip if already exists
            if sha in self.id_to_idx:
                logger.info(f"[FAISS] 🔁 Duplicate skipped: {sha[:10]}")
                return

            # Add to FAISS index
            idx = len(self.metadata)
            self.index.add(np.expand_dims(vector, axis=0))
            
            # Store metadata
            self.metadata[idx] = {
                "session_id": session_id,
                "query": query,
                "response": response,
                "timestamp": time.time(),
                "sha": sha,
                "access_score": access_score,
                "last_accessed": last_accessed or time.time()
            }
            self.id_to_idx[sha] = idx
            
            logger.info(f"[FAISS] ✅ Added vector #{idx} | session={session_id} | score={access_score}")
            
            # Save index periodically (every 10 additions)
            if idx % 10 == 0:
                self.save_index()
                
        except Exception as e:
            logger.error(f"[FAISS] ❌ Error adding vector: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

    def find_similar(self, query_embedding, top_k=5, threshold=0.7):
        """
        Find similar vectors in the FAISS index.
        
        Args:
            query_embedding: The query embedding vector
            top_k: Maximum number of results to return
            threshold: Similarity threshold (0-1)
            
        Returns:
            list: List of similar memory entries with metadata
        """
        logger.info("[FAISS] find_similar called")

        try:
            if self.index.ntotal == 0:
                logger.warning("[FAISS] Index is empty, no vectors stored")
                return []

            # Convert embedding to numpy array if it's a list
            if isinstance(query_embedding, list):
                query_embedding = np.array(query_embedding, dtype=np.float32)

            # Reshape if needed
            if len(query_embedding.shape) == 1:
                query_embedding = query_embedding.reshape(1, -1)

            # Perform search
            distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))

            results = []
            for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
                # Convert distance to similarity score (0-1)
                similarity = 1.0 / (1.0 + dist)

                if similarity >= threshold:
                    result = self.metadata.get(int(idx))
                    if result:
                        result["distance"] = float(dist)
                        result["similarity"] = float(similarity)
                        results.append(result)

            logger.info(f"[FAISS] Found {len(results)} similar results")
            return results

        except Exception as e:
            logger.error(f"[FAISS] Error during search: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def expire_old(self, ttl_seconds=604800):
        """
        Remove old vectors from the index.
        
        Args:
            ttl_seconds: Time-to-live in seconds (default: 7 days)
        """
        try:
            now = time.time()
            to_delete = [idx for idx, m in self.metadata.items() if now - m["timestamp"] > ttl_seconds]

            if not to_delete:
                logger.info("[FAISS] No vectors to expire")
                return

            # Get indices to keep
            keep_indices = [i for i in range(self.index.ntotal) if i not in to_delete]
            
            # Rebuild index with only the vectors to keep
            new_vectors = np.vstack([self.index.reconstruct(i) for i in keep_indices])
            self.index = faiss.IndexFlatL2(VECTOR_SIZE)
            self.index.add(new_vectors)

            # Update metadata and SHA map
            new_meta = {}
            new_sha_map = {}
            for new_idx, old_idx in enumerate(keep_indices):
                meta = self.metadata[old_idx]
                new_meta[new_idx] = meta
                new_sha_map[meta["sha"]] = new_idx

            self.metadata = new_meta
            self.id_to_idx = new_sha_map
            
            logger.info(f"[FAISS] 🧹 Expired {len(to_delete)} vectors, {len(keep_indices)} remaining.")
            
            # Save updated index
            self.save_index()
            
        except Exception as e:
            logger.error(f"[FAISS] Error during expire_old: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

    def save_index(self):
        """
        Save the FAISS index and metadata to disk.
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, INDEX_FILE)
            
            # Save metadata
            with open(META_FILE, "wb") as f:
                pickle.dump((self.metadata, self.id_to_idx), f)
                
            logger.info(f"[FAISS] 💾 Index and metadata saved to disk")
            
        except Exception as e:
            logger.error(f"[FAISS] ❌ Failed to save index: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

    def load_index(self):
        """
        Load the FAISS index and metadata from disk.
        """
        try:
            if os.path.exists(INDEX_FILE):
                self.index = faiss.read_index(INDEX_FILE)
                logger.info(f"[FAISS] 📥 Loaded index from {INDEX_FILE} with {self.index.ntotal} vectors")
                
            if os.path.exists(META_FILE):
                with open(META_FILE, "rb") as f:
                    self.metadata, self.id_to_idx = pickle.load(f)
                logger.info(f"[FAISS] 📥 Loaded metadata from {META_FILE} with {len(self.metadata)} entries")
                
        except Exception as e:
            logger.error(f"[FAISS] ⚠️ Failed to load index or metadata: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Initialize empty index and metadata if loading fails
            self.index = faiss.IndexFlatL2(VECTOR_SIZE)
            self.metadata = {}
            self.id_to_idx = {}
