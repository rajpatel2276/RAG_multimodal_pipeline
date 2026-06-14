import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import requests

from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from qdrant_client import QdrantClient

from app.config import QDRANT_PATH, COLLECTION_NAME, EMBEDDING_MODEL_NAME

app = FastAPI()

# =====================================================================
# SURGICAL CLAMP: Force LlamaIndex to use your Local Models
# =====================================================================
print(f"[*] Clamping Embedding Engine to Local: {EMBEDDING_MODEL_NAME}")
Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)

print("[*] Clamping Synthesis Engine to Local GPU (Ollama: llama3.2)")
Settings.llm = Ollama(
    model="llama3.2", 
    base_url=os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434"),
    request_timeout=120.0,
    context_window=4096
)
# =====================================================================

class ChatMessage(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    question: str
    history: Optional[List[ChatMessage]] = []

@app.post("/query")
async def handle_query(payload: QueryRequest):
    try:
        user_question = payload.question.strip()
        
        # 1. Bypass condensation if there is no chat history to merge
        if not payload.history:
            condensed_query = user_question
            print(f"[*] Fresh session. Using raw query directly: {condensed_query}")
        else:
            formatted_history = "\n".join([f"{msg.role}: {msg.content}" for msg in payload.history])
            
            condense_prompt = (
                f"System: You are an engineering query processor. Rephrase the following follow-up input into a single standalone search query based on the history. "
                f"CRITICAL: Do NOT say 'Sure', do NOT ask for more context, do NOT write explanations. Output ONLY the raw standalone question string.\n\n"
                f"History:\n{formatted_history}\n\n"
                f"Follow-up input: {user_question}\n\n"
                f"Standalone query:"
            )
            
            response = requests.post(
                f"{os.getenv('OLLAMA_BASE_URL', 'http://host.docker.internal:11434')}/api/generate",
                json={"model": "llama3.2", "prompt": condense_prompt, "stream": False}
            )
            
            if response.status_code == 200:
                condensed_query = response.json().get("response", user_question).strip()
                print(f"[*] Condensed question over history: {condensed_query}")
            else:
                print("[!] Ollama condensation failed. Falling back to raw question.")
                condensed_query = user_question

        # 2. Connect to Qdrant and execute Vector Search
        client = QdrantClient(url=QDRANT_PATH)
        vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        
        # 3. Query Engine Execution 
        query_engine = index.as_query_engine(similarity_top_k=5)
        response_obj = query_engine.query(condensed_query)
        
        answer_string = str(response_obj)
        
        # EXPLICITLY EXTRACT CITATIONS & DIAGRAM PATHS
        source_data = []
        if hasattr(response_obj, "source_nodes"):
            for node in response_obj.source_nodes:
                metadata = node.node.metadata
                metadata["score"] = float(node.score) if node.score else 0.0
                metadata["snippet"] = node.node.get_content()[:250] + "..."
                source_data.append(metadata)
        
        # Shotgun Payload with Sources included
        return {
            "response": answer_string,
            "answer": answer_string, 
            "text": answer_string,
            "sources": source_data,
            "status": "success"
        }
    
    
    except Exception as e:
        print(f"[!] CRITICAL BACKEND ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")