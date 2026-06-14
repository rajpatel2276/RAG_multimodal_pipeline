import os
import base64
import requests
import fitz  # PyMuPDF's internal engine
import shutil
from llama_index.core import VectorStoreIndex, Document, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PyMuPDFReader
from qdrant_client import QdrantClient
from app.config import QDRANT_PATH, COLLECTION_NAME, EMBEDDING_MODEL_NAME

def extract_images_from_pdfs(pdf_dir, img_dir):
    print("[*] Initiating Autonomous PDF Image Extraction...")
    
    # Wipe old images to prevent cross-contamination
    if os.path.exists(img_dir):
        shutil.rmtree(img_dir)
    os.makedirs(img_dir)

    total_extracted = 0
    for filename in os.listdir(pdf_dir):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, filename)
            doc = fitz.open(pdf_path)
            
            for page_index in range(len(doc)):
                page = doc[page_index]
                image_list = page.get_images(full=True)
                
                for image_index, img in enumerate(image_list, start=1):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # Save the image physically to the drive
                    image_name = f"{os.path.splitext(filename)[0]}_p{page_index+1}_img{image_index}.{image_ext}"
                    image_path = os.path.join(img_dir, image_name)
                    
                    with open(image_path, "wb") as f:
                        f.write(image_bytes)
                    total_extracted += 1
                    
    print(f"[*] Extracted {total_extracted} raw images from PDFs.")
    return total_extracted

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def analyze_image_with_llava(image_path):
    print(f"[*] Asking LLaVA to analyze: {os.path.basename(image_path)}...")
    base64_image = encode_image_to_base64(image_path)
    
    payload = {
        "model": "llava",
        "prompt": "You are a senior engineer. Describe this architecture diagram or graph in deep technical detail. Identify the layers, blocks, data flow, and key mathematical concepts. If it is just a logo or useless graphic, reply with 'SKIP'.",
        "stream": False,
        "images": [base64_image]
    }
    
    try:
        response = requests.post("http://127.0.0.1:11434/api/generate", json=payload)
        if response.status_code == 200:
            text = response.json().get("response", "")
            if "SKIP" in text:
                return ""
            return text
    except Exception as e:
        print(f"[!] Failed to process {image_path}: {e}")
    return ""

def ingest_data():
    print("[*] Starting Fully Autonomous Multimodal Ingestion...")

    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)
    Settings.llm = None 

    client = QdrantClient(url=QDRANT_PATH) if QDRANT_PATH.startswith("http") else QdrantClient(path=QDRANT_PATH)
    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    all_documents = []
    base_dir = os.path.dirname(__file__)
    pdf_dir = os.path.join(base_dir, "data", "pdf")
    img_dir = os.path.join(base_dir, "data", "images")

    # --- 1. AUTONOMOUS EXTRACTION ---
    if os.path.exists(pdf_dir):
        extract_images_from_pdfs(pdf_dir, img_dir)

    # --- 2. INGEST TEXT ---
    if os.path.exists(pdf_dir) and os.listdir(pdf_dir):
        print(f"[*] Reading PDF text from {pdf_dir}...")
        pdf_extractor = {".pdf": PyMuPDFReader()}
        pdf_docs = SimpleDirectoryReader(pdf_dir, file_extractor=pdf_extractor).load_data()
        all_documents.extend(pdf_docs)

    # --- 3. INGEST IMAGES (Vision Loop) ---
    if os.path.exists(img_dir):
        for file in os.listdir(img_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(img_dir, file)
                vision_summary = analyze_image_with_llava(img_path)
                
                if vision_summary:
                    doc = Document(
                        text=f"Image Summary:\n{vision_summary}",
                        metadata={"source_image": img_path, "chunk_type": "multimodal_vision"}
                    )
                    all_documents.append(doc)

    if not all_documents:
        print("[!] ERROR: No data found. Aborting.")
        return

    # --- 4. EMBED & PUSH ---
    print(f"[*] Processing and pushing {len(all_documents)} multimodal chunks to Qdrant...")
    VectorStoreIndex.from_documents(all_documents, storage_context=storage_context, show_progress=True)
    print("[*] SUCCESS: Multimodal Pipeline Complete!")

if __name__ == "__main__":
    ingest_data()9