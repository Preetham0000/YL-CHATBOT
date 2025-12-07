"""Ingestion script for Google Gemini File Search."""
import os
import glob
import time
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")

client = genai.Client(api_key=API_KEY)

def upload_to_gemini(data_dir: str = "data/raw"):
    print(f"--- Starting Gemini File Search Ingestion from {data_dir} ---")

    # 1. Create a new Store
    # In a production scenario, you might want to update an existing store instead.
    store = client.file_search_stores.create(
        config={"display_name": "Yantra Live Knowledge Base"}
    )
    print(f"Created new Store: {store.name}")
    print(f"Store ID: {store.name.split('/')[-1]}")

    # 2. Find PDFs
    pdf_files = glob.glob(os.path.join(data_dir, "**/*.pdf"), recursive=True)
    if not pdf_files:
        print("No PDFs found to upload.")
        return

    print(f"Found {len(pdf_files)} PDFs. Uploading...")

    # 3. Upload Files
    # We upload files to the Files API first, then link them to the store.
    uploaded_files = []
    for pdf_path in pdf_files:
        print(f"Uploading {os.path.basename(pdf_path)}...")
        try:
            with open(pdf_path, "rb") as f:
                file_ref = client.files.upload(
                    file=f,
                    config={
                        "display_name": os.path.basename(pdf_path),
                        "mime_type": "application/pdf"
                    }
                )
                uploaded_files.append(file_ref)
        except Exception as e:
            print(f"Failed to upload {pdf_path}: {e}")

    if not uploaded_files:
        print("No files uploaded successfully.")
        return

    # 4. Add files to the Store (Indexing)
    print("Indexing files into the store...")
    # The SDK allows adding a batch of files to the store
    # We need to extract the 'name' (resource ID) from the uploaded file objects
    file_resource_names = [f.name for f in uploaded_files]
    
    # Currently, we iterate and import. 
    # (Batch import might be available depending on exact SDK version, loop is safer for now)
    for file_ref in uploaded_files:
        try:
            client.file_search_stores.import_file(
                file_search_store_name=store.name,
                file_name=file_ref.name
            )
            print(f"Linked {file_ref.name}")
        except Exception as e:
            print(f"Error linking file: {e}")

    print("\n--- Ingestion Complete ---")
    print(f"IMPORTANT: Add this line to your .env file:")
    print(f"GEMINI_STORE_ID={store.name.split('/')[-1]}")

if __name__ == "__main__":
    upload_to_gemini()
