# Multimodal RAG Architecture for Computer Vision (YOLO)

An end-to-end, 100% offline, GPU-accelerated Retrieval-Augmented Generation (RAG) pipeline designed to ingest, process, and query complex computer vision research papers. 

This system uses a containerized architecture to securely process multimodal queries, allowing users to ask technical questions about the YOLOv8 architecture and receive locally synthesized answers backed by exact source citations and native architectural diagrams.

## 🏗️ Architecture Stack

* **Frontend:** Streamlit (Containerized)
* **Backend:** FastAPI (Containerized)
* **Vector Database:** Qdrant (Containerized via Podman)
* **Orchestration:** LlamaIndex
* **Embedding Model:** BAAI/bge-small-en-v1.5 (Local CPU)
* **LLM / Synthesis Engine:** Llama 3.2 via Ollama (Local GPU / RTX 3050)
* **Document Parsing:** PyMuPDF (Industrial PDF extraction)

## ⚡ Prerequisites

This project is built for a local Linux/WSL environment. You must have the following installed:
* **Podman & Podman-Compose** (for container orchestration)
* **Ollama** running locally on the host machine (with the `llama3.2` model pulled)
* **Python 3.12+**
* Minimum 10GB System RAM allocated to WSL / 4GB VRAM

## 📂 Data Segregation Protocol

To prevent vector poisoning, data must be strictly segregated:
1. **Source Documents:** Place your target PDF (e.g., YOLO research paper) strictly inside the `/data/` directory. **Do not place raw images here.**
2. **Visual Assets:** Place extracted reference images or diagrams (e.g., `yolov8_diagram.jpg`) inside the `/app/static/` directory so the frontend can render them securely.

## 🚀 Quick Start Guide

### 1. Environment Setup
Clone the repository and set up your Python environment for data ingestion:
```bash
git clone [https://github.com/rajpatel2276/RAG_multimodal_pipeline.git](https://github.com/rajpatel2276/RAG_multimodal_pipeline.git)
cd RAG_multimodal_pipeline
python -m venv rag_env
source rag_env/bin/activate
pip install -r requirements.txt


Boot the Vector Database
Start the container cluster in the background. The backend and frontend will wait for Qdrant to initialize.

Bash
podman-compose up -d
3. Execute Data Ingestion
Run the LlamaIndex pipeline to parse the PDF, generate dense vectors using the BAAI embedding model, and push them over the network into the Qdrant container:

Bash
python run_pipeline.py
(Wait for the terminal to print: [*] SUCCESS: Ingestion Complete!)

4. Query the System
Open your web browser and navigate to the Streamlit UI:
👉 http://localhost:8501

Submit your query (e.g., "What is YOLO in computer vision?"). The backend will retrieve the chunks, constrain the LLM context window to prevent memory leaks, synthesize the answer via your local GPU, and return the text with exact citations.

🛑 Shutting Down
To gracefully shut down the network bridge and release the ports, run:

Bash
podman-compose down
To drop background GPU memory locks, restart the Ollama daemon:

Bash
sudo systemctl restart ollama
⚠️ Troubleshooting
Out-of-Memory (OOM) Crash: If the backend throws a 500 error regarding ggml_aligned_malloc, the LLM attempted to exceed your physical VRAM. Ensure context_window=4096 is clamped in app/main.py.

Address Already in Use (Port 8501): A ghost Streamlit process is running on your host. Kill it with sudo fuser -k 8501/tcp.

Citation Gibberish: If the UI returns UTF-16 hex code instead of English, a raw image or corrupted PDF was ingested. Run sudo rm -rf qdrant_storage/* to nuke the database volume, isolate your PDF, and re-ingest.


### **How to Push This to GitHub**

Once you save the `README.md` file, run these three commands to push the documentation to your repository:

```bash
git add README.md
git commit -m "docs: add comprehensive architecture documentation"
git push