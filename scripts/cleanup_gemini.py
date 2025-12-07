"""Cleanup script for Google Gemini File Search Stores."""
import os
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")

client = genai.Client(api_key=API_KEY)

def list_and_delete_stores():
    print("--- Cleanup Process Started ---")

    # 1. Delete Files
    print("\n--- Step 1: Checking for uploaded files ---")
    try:
        files = list(client.files.list())
        if files:
            print(f"Found {len(files)} files taking up storage.")
            confirm_files = input("Do you want to delete ALL these files to free up space? (yes/no): ").strip().lower()
            if confirm_files == 'yes':
                print("Deleting files...")
                for f in files:
                    try:
                        print(f"Deleting file {f.name} ({f.display_name})...")
                        client.files.delete(name=f.name)
                    except Exception as e:
                        print(f"Failed to delete file {f.name}: {e}")
        else:
            print("No files found.")
    except Exception as e:
        print(f"Error listing/deleting files: {e}")

    # 2. Delete Stores
    print("\n--- Step 2: Listing Gemini File Search Stores ---")
    
    # List all stores
    # Note: The SDK iterator handles pagination automatically
    stores = list(client.file_search_stores.list())
    
    if not stores:
        print("No stores found.")
        return

    print(f"Found {len(stores)} stores:")
    for i, store in enumerate(stores):
        print(f"{i+1}. {store.name}")
        print(f"   Display Name: {store.display_name}")
        print(f"   Size: {store.size_bytes} bytes")
        print(f"   Docs: Active={store.active_documents_count}, Failed={store.failed_documents_count}, Pending={store.pending_documents_count}")



    confirm = input("\nDo you want to delete ALL these stores to free up space? (yes/no): ").strip().lower()
    
    if confirm == "yes":
        print("\nDeleting stores...")
        for store in stores:
            print(f"Processing {store.name}...")
            
            # Try to delete the store directly with force=True
            try:
                print(f"  Attempting to delete store {store.name} with force=True...")
                client.file_search_stores.delete(name=store.name, config={'force': True})
                print("  Store deleted successfully.")
                continue # Skip document deletion if store is deleted
            except Exception as e:
                print(f"  Direct force delete failed: {e}")

            # If direct delete fails, try deleting documents first
            try:
                print(f"  Listing documents in {store.name}...")
                docs = list(client.file_search_stores.documents.list(parent=store.name))
                if docs:
                    print(f"  Found {len(docs)} documents. Deleting them...")
                    for doc in docs:
                        try:
                            client.file_search_stores.documents.delete(name=doc.name, config={'force': True})
                            # print(f"    Deleted {doc.name}")
                        except Exception as e:
                            print(f"    Failed to delete document {doc.name}: {e}")
                    print("  All documents deleted.")
                else:
                    print("  No documents found in store.")
            except Exception as e:
                print(f"  Error listing/deleting documents: {e}")

            # 2. Delete the store itself (again)
            try:
                print(f"  Deleting store {store.name}...")
                client.file_search_stores.delete(name=store.name, config={'force': True})
                print("  Store deleted.")
            except Exception as e:
                print(f"  Failed to delete store {store.name}: {e}")

        print("\n--- Cleanup Complete ---")
        print("You can now re-run 'python scripts/ingest_gemini.py' to create a fresh store.")
    else:
        print("Operation cancelled.")

if __name__ == "__main__":
    list_and_delete_stores()
