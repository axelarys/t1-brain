# 🧠 T1 Brain — Human-Like Memory System

T1 Brain is an intelligent memory orchestration engine that mimics human-like recall, reinforcement, and reflection. It combines vector search, graph relationships, and GPT-driven enrichment to build persistent and meaningful conversations.

---

## 🚀 Features

- 🔁 **Memory Routing Engine** (Graph, Vector, Update, Delete)
- 🧠 **RAG-style Chunking** for efficient embedding + retrieval
- 🧩 **LangChain Agent Orchestration** with Tool fallback
- 📆 **Persistent Memory Layers**: PostgreSQL, Redis, Neo4j
- 🧠 **Contextual Enrichment** (intent, emotion, topic, priority)
- 🦮 **Reflection + Reinforcement** for self-healing memories
- 🔐 **FastAPI Layer** with secure GPT Action integration
- 📈 **Token Usage Logging** to track API costs (OpenAI)

---

## 🧱 Architecture Overview

```
             +---------------------------+
             |      FastAPI Gateway      |
             +------------+--------------+
                          |
              +-----------v------------+
              |  LangChain AgentExecutor|
              +------------------------+
                |   |    |     |     |
         route  |   |  enrich |  clarify
                |   |    |     |
     +----------v---v----v-----v-----------+
     |      MemoryRouter + Tools           |
     +----------------+--------------------+
                      |
    +-----------------+----------------------------+
    |     Redis     |    PostgreSQL     |   Neo4j  |
    | (Session TTL) | (Chunked Vectors) | (Graph)  |
    +---------------+------------------+-----------+
```

---

## ⚙️ Tech Stack

- **FastAPI** — API Layer
- **LangChain** — Agent Orchestration
- **OpenAI (GPT-4, Ada-002)** — Enrichment + Embedding
- **PostgreSQL** — Vector Store
- **Redis** — Session Cache
- **Neo4j** — Contextual GraphDB
- **Tiktoken** — Token-aware chunking

---

## 🛠️ Setup Instructions

```bash
# Clone project
git clone https://github.com/yourorg/t1-brain.git
cd t1-brain

# Setup virtual environment
python3 -m venv brain
source brain/bin/activate

# Install dependencies
pip install -r requirements.txt

# Add your OpenAI + DB credentials in config/settings.py or .env
```

---

## 🔄 Deploy & Restart

```bash
# Pull latest + deploy
./deploy_prod.sh
```

Or manually:
```bash
sudo systemctl restart t1brain-api
```

---

## 📈 Token Logging (API Usage)

All embedding + enrichment tokens are logged at:

```
/root/projects/t1-brain/logs/token_usage.log
```

Format:
```
[2025-03-28 16:10] embedding | tokens=132 | model=text-embedding-ada-002
[2025-03-28 16:11] enrichment | session=xyz | tokens=82 | model=gpt-4
```

---

## 🤝 Contributors

Built by [@yourgithub](https://github.com/yourgithub)  
Production-ready memory for GPT-native thinking systems.

---

## 📄 License

MIT License — use freely, improve collaboratively.

