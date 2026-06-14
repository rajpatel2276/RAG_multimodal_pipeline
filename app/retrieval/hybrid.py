import os
from qdrant_client import QdrantClient
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.postprocessor import SentenceTransformerRerank
from app.config import QDRANT_PATH, COLLECTION_NAME, EMBEDDING_MODEL_NAME, OLLAMA_BASE_URL

def get_index():
    print("[*] Initializing Vector Index with Cross-Encoder Guardrails...")
    
    # 1. GPU Configuration for Llama 3.2 (Text generation)
    Settings.llm = Ollama(
        model="llama3.2", 
        base_url=OLLAMA_BASE_URL, 
        request_timeout=120.0,
        additional_kwargs={"num_ctx": 2048}
    )
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)
    
    # 2. Connect to local Qdrant Vector DB
    # 2. Connect to local or containerized Qdrant dynamically
    if QDRANT_PATH.startswith("http"):
        client = QdrantClient(url=QDRANT_PATH)  # Network mode (Podman)
    else:
        client = QdrantClient(path=QDRANT_PATH) # Disk mode (Local)
        
    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    
    print("[*] Database Index Loaded successfully.")
    return index

def get_reranker():
    print("[*] Loading BAAI/bge-reranker-base onto Host CPU...")
    # 3. Initialize the Re-ranker explicitly targeting the CPU to protect VRAM boundaries
    reranker = SentenceTransformerRerank(
        model="BAAI/bge-reranker-base",
        top_n=2,  # Drastically filters down the context size before feeding it to Llama 3.2
        device="cpu"
    )
    return reranker