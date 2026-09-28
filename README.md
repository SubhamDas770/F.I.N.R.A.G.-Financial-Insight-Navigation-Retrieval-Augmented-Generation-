# FinSight 10-K: Enterprise Financial RAG Engine

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Qdrant Vector DB](https://img.shields.io/badge/VectorDB-Qdrant-red.svg)](https://qdrant.tech/)
[![Fine-Tuning QLoRA](https://img.shields.io/badge/Fine--Tuning-QLoRA%204--bit-orange.svg)](https://github.com/artidoro/qlora)
[![Modal Cloud](https://img.shields.io/badge/Deployment-Modal%20Serverless-purple.svg)](https://modal.com/)
[![License MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise-grade, high-precision Retrieval-Augmented Generation (RAG) platform designed to parse, index, rerank, and analyze complex SEC Form 10-K financial filings. Built with parent-child hierarchical chunking, `BAAI/bge-reranker-large` cross-encoder reranking, QLoRA 4-bit domain fine-tuning, Redis semantic caching, and deployed on **Modal serverless cloud infrastructure**.

---

## 🏛️ System Architecture

```
                                  [ User Query ]
                                         │
                                         ▼
                             ┌──────────────────────┐
                             │ Redis Semantic Cache │
                             └──────────┬───────────┘
                                        │
                         ┌──────────────┴──────────────┐
                   Cache HIT ($<0.05\text{s}$)    Cache MISS (Live Pass)
                         │                             │
                         ▼                             ▼
                 [ Instant Return ]         ┌─────────────────────┐
                                            │ Qdrant Vector Store │
                                            │ (Child Embeddings)  │
                                            └──────────┬──────────┘
                                                       │
                                                       ▼
                                            ┌─────────────────────┐
                                            │ Cross-Encoder       │
                                            │ Reranker (BGE-Large)│
                                            └──────────┬──────────┘
                                                       │
                                                       ▼
                                            ┌─────────────────────┐
                                            │ Parent Context      │
                                            │ Expansion           │
                                            └──────────┬──────────┘
                                                       │
                                                       ▼
                                            ┌─────────────────────┐
                                            │ QLoRA Fine-Tuned    │
                                            │ Llama Model         │
                                            └──────────┬──────────┘
                                                       │
                                                       ▼
                                            ┌─────────────────────┐
                                            │ Dual-Panel UI &     │
                                            │ Real-Time HUD       │
                                            └─────────────────────┘
```

---

## ✨ Key Technical Capabilities

* **Parent-Child Hierarchical Chunking:** Preserves precise SEC filing tables and financial narrative continuity by retrieving targeted child segments while expanding to full parent context blocks for generation.
* **Cross-Encoder Reranking:** Implements `BAAI/bge-reranker-large` to re-score vector search results via deep cross-attention, boosting semantic retrieval precision to $>0.90$.
* **Domain Fine-Tuning (QLoRA):** Utilizes 4-bit NormalFloat quantization with LoRA adapters fine-tuned on financial Q&A datasets to minimize hallucinations in balance sheet and cash flow interpretation.
* **Redis Semantic Caching:** Prevents redundant LLM calls by checking semantically equivalent queries, dropping warm repeated request latency down to $<0.05\text{s}$.
* **Real-Time Telemetry Inspector HUD:** Built-in Gradio interface displaying live token usage, reranker scores, latency breakdowns, and full JSON payloads of fetched parent document chunks.
* **Serverless Cloud Hosting on Modal:** Containerized ASGI app using FastAPI and Gradio, backed by persistent volume storage (`rag-data-vol`) for zero-data-loss model weight and vector index retention.

---

## 🛠️ Tech Stack Breakdown

| Component | Technology / Library | Purpose |
| :--- | :--- | :--- |
| **Embeddings** | `BAAI/bge-small-en-v1.5` | Dense vector generation ($384\text{-dimensional}$) |
| **Vector Database** | Qdrant Client v1.10+ | Disk-backed collection indexing & filtering |
| **Reranker** | `BAAI/bge-reranker-large` | Cross-encoder precision scoring |
| **LLM / Fine-Tuning** | PyTorch, HuggingFace `peft`, `bitsandbytes` | 4-bit QLoRA instruction fine-tuning |
| **Caching & Memory** | Redis Server, Redis-Py | Semantic response caching & chat state persistence |
| **User Interface** | Gradio 4.44.1, FastAPI | Interactive dual-panel RAG portal & telemetry HUD |
| **Cloud Hosting** | Modal Serverless | Stateful serverless deployment with mounted volumes |

---

## 📂 Repository Structure

```
.
├── modal_app.py                 # Modal cloud app definition & ASGI mounting
├── docker/
│   └── docker-compose.yml       # Local Redis & Qdrant container orchestration
├── src/
│   ├── ingestion/               # SEC 10-K PDF parsing & parent-child chunking
│   │   ├── parser.py
│   │   ├── chunker.py
│   │   └── ingest.py
│   ├── retrieval/               # Qdrant vector store, reranker, and Redis cache
│   │   ├── vector_store.py
│   │   ├── cache.py
│   │   └── orchestrator.py
│   ├── memory/                  # Multi-turn conversation history tracker
│   │   └── chat_history.py
│   ├── training/                # QLoRA fine-tuning scripts and dataset builders
│   │   ├── dataset.py
│   │   └── train_qlora.py
│   └── ui/                      # Gradio interface & telemetry components
│       └── app.py
├── data/                        # Persistent volume root (mapped on Modal)
│   ├── vector_db/               # Qdrant collection index files
│   ├── fine_tuned_tinyllama/    # QLoRA adapter checkpoints
│   └── apple_10k_2024.pdf       # Raw SEC Form 10-K source filing
├── requirements.txt             # Python dependencies
└── README.md                    # System documentation
```

---

## 🚀 Quickstart & Local Setup

### 1. Environment Preparation
Clone the repository and create a Python 3.10 virtual environment:
```bash
git clone https://github.com/your-username/FinSight-10K.git
cd FinSight-10K

python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Launch Local Services (Docker)
Start local Qdrant and Redis instances:
```bash
docker-compose -f docker/docker-compose.yml up -d
```

### 3. Run Ingestion Pipeline
Parse Apple's 2024 Form 10-K and push parent-child embeddings into Qdrant:
```bash
python -m src.ingestion.ingest
```

### 4. Execute QLoRA Fine-Tuning (Optional)
Fine-tune domain-specific adapters locally or on a GPU instance:
```bash
python -m src.training.train_qlora
```

### 5. Launch Local Web UI
Start the local interactive Gradio interface:
```bash
python -m src.ui.app
```
Navigate to `http://localhost:7860` in your web browser.

---

## ☁️ Deploying to Modal Serverless Cloud

### 1. Authenticate Modal CLI
```bash
modal setup
```

### 2. Create Data Volume & Sync Local Artifacts
Create the persistent Modal volume and copy your local dataset, vector DB, and fine-tuned weights:
```bash
modal volume create rag-data-vol
modal volume put rag-data-vol ./data/* /data/
```

### 3. Verify Volume Persistence
```bash
modal volume ls rag-data-vol /data
```

### 4. Deploy Serverless Endpoint
Deploy the live FastAPI/Gradio web application:
```bash
modal deploy modal_app.py
```

Upon deployment, Modal provides a public HTTPS web endpoint (e.g., `https://<username>--enterprise-rag-app-web.modal.run`).

---

## 📊 Performance Benchmarks

| Query Phase | Execution Latency | Description |
| :--- | :--- | :--- |
| **Cold Start** | $\approx 27.0\text{s}$ | Initial container startup, mounting volume, loading PyTorch & cross-encoder models into RAM |
| **Warm Live Query** | $1.5\text{s} - 3.0\text{s}$ | Full Qdrant search, `bge-reranker-large` scoring, parent expansion, and LLM generation |
| **Semantic Cache Hit** | $< 0.05\text{s}$ | Redis semantic cache match ($>0.92$ vector similarity threshold), skipping model execution |

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.