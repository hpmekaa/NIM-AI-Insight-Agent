# NIM AI Insight Agent

A high-performance RAG (Retrieval-Augmented Generation) Agent designed to transform massive enterprise documentation into verifiable intelligence. Powered by **NVIDIA NIM** and **Llama 3.3 70B**.

## Performance Metric
* **Cold Start (Document Ingestion):** ~45-60s (for 300+ page PDFs)
* **Hot Query (Persistent Index):** **1.8s - 2.5s**
* **Efficiency:** **95%+ Latency reduction** on recurring document queries via local FAISS persistence.

## Technical Architecture
* **LLM:** Llama-3.3-70B-Instruct (via NVIDIA NIM)
* **Embedding Strategy:** **Asymmetric Retrieval** using `NV-EmbedQA-1B` (Query vs. Passage optimized)
* **Vector Store:** FAISS with disk-level persistence
* **Orchestration:** LangChain & Streamlit
* **Infrastructure:** Dockerized for accelerated computing environments

## Key Features
* **Explainable AI:** Integrated "Chunk Viewer" to inspect the raw context grounding the agent's response.
* **Observability:** Live performance tracking (Latency, Total Chunks, Retrieval Strategy).
* **Scalability:** Optimized to handle high-file-size PDFs (tested on 300+ page AWS Best Practices).

## Deployment

### Live Demo
Access the live application here: [NIM AI Insight Agent Live][(https://nim-ai-insight-agent-5q44qxk7pbphxkesjzsktk.streamlit.app/)]

### Local Development (Docker)
```bash
docker build -t nim-insight-agent .
docker run -p 8501:8501 --env-file .env nim-insight-agent
