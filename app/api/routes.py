from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from llama_index.core.llms import ChatMessage, MessageRole
from app.retrieval.hybrid import get_index, get_reranker

router = APIRouter()

try:
    doc_index = get_index()
    node_reranker = get_reranker()
except Exception as e:
    print(f"[!] Failed to initialize retrieval core: {e}")
    doc_index = None
    node_reranker = None

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    question: str
    history: List[Message] = []

@router.post("/query")
async def chat_with_data(request: ChatRequest):
    if not doc_index or not node_reranker:
        raise HTTPException(status_code=500, detail="Retrieval/Re-ranking pipeline is offline.")
    
    try:
        # 1. Reconstruct conversation history states
        chat_history = []
        for msg in request.history:
            role = MessageRole.USER if msg.role == "user" else MessageRole.ASSISTANT
            chat_history.append(ChatMessage(role=role, content=msg.content))
        
        # 2. Construct Chat Engine compiling Qdrant and the CPU Cross-Encoder
        chat_engine = doc_index.as_chat_engine(
            chat_mode="condense_plus_context",
            similarity_top_k=10,  # Expand initial retrieval pool for the re-ranker to screen
            node_postprocessors=[node_reranker],  # The CPU re-ranker intercepts here
            verbose=True
        )
        
        # 3. Process the query contextually
        response = chat_engine.chat(request.question, chat_history=chat_history)
        
        # 4. Filter and map high-confidence citations 
        sources = []
        for node in response.source_nodes:
            meta = node.node.metadata
            chunk_type = meta.get("chunk_type", "text")
            source_info = {"type": chunk_type, "relevance_score": float(node.score) if node.score else None}
            
            if chunk_type == "image_summary":
                source_info["image_link"] = meta.get("source_image")
            else:
                source_info["page"] = meta.get("source_page")
                
            sources.append(source_info)

        return {
            "answer": str(response),
            "citations": sources
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))