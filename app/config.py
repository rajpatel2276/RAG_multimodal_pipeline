import os

# If running inside Podman, use the container alias. If running on host, use localhost.
QDRANT_PATH = os.getenv("QDRANT_PATH", "http://localhost:6333") 

# Bare-metal Ollama Daemon
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

# Vector DB Settings
COLLECTION_NAME = "cv_architectures"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5" # Ensures compatibility with the Re-ranker