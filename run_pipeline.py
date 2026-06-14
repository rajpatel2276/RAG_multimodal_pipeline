import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PyMuPDFReader
from qdrant_client import QdrantClient
from app.config import QDRANT_PATH, COLLECTION_NAME, EMBEDDING_MODEL_NAME

def ingest_data():
    print("[*] Starting Data Ingestion Pipeline...")

    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)
    Settings.llm = None 

    print(f"[*] Connecting to Qdrant Database at: {QDRANT_PATH}")
    if QDRANT_PATH.startswith("http"):
        client = QdrantClient(url=QDRANT_PATH)
    else:
        client = QdrantClient(path=QDRANT_PATH)

    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    print(f"[*] Reading raw architecture documents from {data_dir}...")
    
    try:
        # SURGICAL PATCH: Force LlamaIndex to use the industrial PyMuPDF parser
        pdf_extractor = {".pdf": PyMuPDFReader()}
        documents = SimpleDirectoryReader(
            data_dir, 
            file_extractor=pdf_extractor
        ).load_data()
    except Exception as e:
        print(f"[!] ERROR: Failed to read from {data_dir}. Details: {e}")
        return

    if not documents:
        print("[!] ERROR: No documents found. Aborting ingestion.")
        return

    print(f"[*] Processing and pushing {len(documents)} document chunks to Qdrant Container...")
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )

    print("[*] SUCCESS: Ingestion Complete! The Podman database is now populated.")

if __name__ == "__main__":
    ingest_data()