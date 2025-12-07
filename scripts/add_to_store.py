"""Script to add a single PDF to the existing Gemini File Search Store."""
import os
import sys
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
STORE_ID = os.getenv("GEMINI_STORE_ID")

if not API_KEY:
    print("Error: GEMINI_API_KEY not found in .env")
    sys.exit(1)

if not STORE_ID:
    print("Error: GEMINI_STORE_ID not found in .env. Please run ingest_gemini.py first to create a store.")
    sys.exit(1)

client = genai.Client(api_key=API_KEY)
# Ensure the store ID is formatted correctly (it might be just the ID or the full resource name)
if not STORE_ID.startswith("fileSearchStores/"):
    store_name = f"fileSearchStores/{STORE_ID}"
else:
    store_name = STORE_ID

def add_file_to_store(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    print(f"--- Adding {os.path.basename(file_path)} to Store {STORE_ID} ---")

    try:
        # 1. Upload the file
        print("Uploading file...")
        with open(file_path, "rb") as f:
            file_ref = client.files.upload(
                file=f,
                config={
                    "display_name": os.path.basename(file_path),
                    "mime_type": "application/pdf"
                }
            )
        print(f"Uploaded: {file_ref.name}")

        # 2. Link to Store
        print("Linking to store...")
        client.file_search_stores.import_file(
            file_search_store_name=store_name,
            file_name=file_ref.name
        )
        print("Success! File added to knowledge base.")

    except Exception as e:
        print(f"Error adding file: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/add_to_store.py <path_to_pdf>")
        print("Example: python scripts/add_to_store.py data/raw/new_brochure.pdf")
    else:
        file_path = sys.argv[1]
        add_file_to_store(file_path)
