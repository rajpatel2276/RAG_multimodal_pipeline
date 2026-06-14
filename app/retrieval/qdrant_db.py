import os
from qdrant_client import QdrantClient
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from app.config import QDRANT_PATH, COLLECTION_NAME, EMBEDDING_MODEL_NAME

def ingest_chunks_to_qdrant(chunks):
    """
    Takes raw text and image_summary chunks, generates high-dimensional 
    embeddings, and stores them in a local Qdrant vector database.
    """
    print(f"[*] Loading Embedding Model: {EMBEDDING_MODEL_NAME}")
    print("[*] Note: The first run will download the model weights from Hugging Face.")
    
    # We use a local HuggingFace embedding model for high accuracy and zero API costs
    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)

    print(f"[*] Initializing Qdrant Database at {QDRANT_PATH}...")
    client = QdrantClient(path=QDRANT_PATH)
    
    vector_store = QdrantVectorStore(
        client=client, 
        collection_name=COLLECTION_NAME
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("[*] Converting raw chunks to LlamaIndex Documents...")
    documents = []
    for chunk in chunks:
        # We explicitly preserve the chunk type (text vs image_summary) in the vector metadata
        meta = chunk.get("metadata", {})
        meta["chunk_type"] = chunk.get("type")
        
        doc = Document(
            text=chunk["content"],
            metadata=meta
        )
        documents.append(doc)

    print(f"[*] Generating embeddings for {len(documents)} chunks and pushing to Qdrant. This requires compute...")
    
    # This process maps the text to vectors and stores them in Qdrant
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )
    
    print("[*] Data successfully embedded and stored in Qdrant.")
    return index