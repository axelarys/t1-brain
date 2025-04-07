# memory/warm_layer.py

import faiss
import numpy as np
import time
import hashlib
import os
import pickle

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
        print("[FAISS] 🧠 WarmMemoryCache initialized")

    def _hash(self, query: str, response: str) -> str:
        return hashlib.sha256((query + response).encode()).hexdigest()

    def add(self, session_id: str, query: str, response: str, embedding: list):
        vector = np.array(embedding).astype("float32")
        sha = self._hash(query, response)

        if sha in self.id_to_idx:
            print(f"[FAISS] 🔁 Duplicate skipped: {sha}")
            return

        idx = len(self.metadata)
        self.index.add(np.expand_dims(vector, axis=0))
        self.metadata[idx] = {
            "session_id": session_id,
            "query": query,
            "response": response,
            "timestamp": time.time(),
            "sha": sha
        }
        self.id_to_idx[sha] = idx
        print(f"[FAISS] ✅ Added vector #{idx} for session {session_id}. Index now contains {self.index.ntotal} vectors")

    def find_similar(self, query_embedding, top_k=5, threshold=0.7):
        print("[FAISS] find_similar called")

        if self.index.ntotal == 0:
            print("[FAISS] Index is empty, no vectors stored")
            return []

        print(f"[FAISS] Index contains {self.index.ntotal} vectors")

        if isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding, dtype=np.float32)

        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        print(f"[FAISS] Query embedding shape: {query_embedding.shape}")

        try:
            distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
            print(f"[FAISS] Search returned {len(indices[0])} results")
            print(f"[FAISS] Distances: {distances[0]}")
            print(f"[FAISS] Indices: {indices[0]}")

            results = []
            for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
                similarity = 1.0 / (1.0 + dist)
                print(f"[FAISS] Result {i}: idx={idx}, dist={dist:.4f}, similarity={similarity:.4f}")

                if similarity >= threshold:
                    result = self.metadata.get(idx)
                    if result:
                        result["distance"] = float(dist)
                        results.append(result)

            return results

        except Exception as e:
            print(f"[FAISS] Error during search: {e}")
            import traceback
            traceback.print_exc()
            return []

    def expire_old(self, ttl_seconds=604800):
        now = time.time()
        to_delete = [idx for idx, m in self.metadata.items() if now - m["timestamp"] > ttl_seconds]

        if not to_delete:
            return

        keep_indices = [i for i in range(self.index.ntotal) if i not in to_delete]
        new_vectors = self.index.reconstruct_n(0, self.index.ntotal)[keep_indices]
        self.index = faiss.IndexFlatL2(VECTOR_SIZE)
        self.index.add(new_vectors)

        new_meta = {}
        new_sha_map = {}
        for new_idx, old_idx in enumerate(keep_indices):
            meta = self.metadata[old_idx]
            new_meta[new_idx] = meta
            new_sha_map[meta["sha"]] = new_idx

        self.metadata = new_meta
        self.id_to_idx = new_sha_map
        print(f"[FAISS] 🧹 Expired {len(to_delete)} vectors, {len(keep_indices)} remaining.")

    def save_index(self):
        try:
            faiss.write_index(self.index, INDEX_FILE)
            with open(META_FILE, "wb") as f:
                pickle.dump((self.metadata, self.id_to_idx), f)
            print(f"[FAISS] 💾 Index and metadata saved to disk")
        except Exception as e:
            print(f"[FAISS] ❌ Failed to save index: {e}")

    def load_index(self):
        try:
            if os.path.exists(INDEX_FILE):
                self.index = faiss.read_index(INDEX_FILE)
                print(f"[FAISS] 📥 Loaded index from {INDEX_FILE}")
            if os.path.exists(META_FILE):
                with open(META_FILE, "rb") as f:
                    self.metadata, self.id_to_idx = pickle.load(f)
                print(f"[FAISS] 📥 Loaded metadata from {META_FILE}")
        except Exception as e:
            print(f"[FAISS] ⚠️ Failed to load index or metadata: {e}")
