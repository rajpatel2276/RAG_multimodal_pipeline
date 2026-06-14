import os
from app.ingestion.parser import process_research_paper

def main():
    # Target the specific PDF in your data folder
    test_pdf_path = os.path.join(os.getcwd(), "data", "test_paper.pdf")
    
    if not os.path.exists(test_pdf_path):
        print(f"[!] Error: Could not find {test_pdf_path}. Please place a PDF there.")
        return

    print(f"[*] Starting ingestion test on: {test_pdf_path}")
    print("[*] Ensure 'ollama serve' is running in another terminal window.\n")
    
    try:
        # Run the parser
        chunks = process_research_paper(test_pdf_path)
        
        print("\n" + "="*50)
        print("[*] EXTRACTION RESULTS PREVIEW")
        print("="*50)
        
        # We only print the first 5 chunks to avoid nuking your terminal
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i+1} | Type: {chunk['type']} ---")
            
            if chunk['type'] == 'image_summary':
                print(f"Source Image: {chunk['metadata'].get('source_image')}")
            else:
                print(f"Source Page: {chunk['metadata'].get('source_page')}")
                
            # Print a preview of the content
            content_preview = chunk['content'][:300].replace('\n', ' ')
            print(f"Content Preview: {content_preview}...")
            
        print("\n" + "="*50)
        print(f"[*] Total chunks extracted: {len(chunks)}")
        
        # Verify the image directory
        image_dir = os.path.join(os.getcwd(), "data", "extracted_images")
        if os.path.exists(image_dir):
            extracted_files = os.listdir(image_dir)
            print(f"[*] Images physically saved to disk: {len(extracted_files)}")
        else:
            print("[!] Image directory was not created.")

        print("[*] Pipeline test completed.\n")
        
    except Exception as e:
        print(f"\n[!] Pipeline failed critically: {e}")

if __name__ == "__main__":
    main()