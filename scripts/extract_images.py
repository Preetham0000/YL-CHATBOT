import logging
import sys
from pathlib import Path

# Add the project root to the python path so we can import yantra_rag
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from yantra_rag.pdf_utils import extract_pdf_content

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)

def main():
    pdf_path = project_root / "data" / "raw" / "JCB_Service manual 3CX 4CX.pdf"
    output_dir = project_root / "data" / "processed" / "images" / pdf_path.stem
    
    if not pdf_path.exists():
        LOGGER.error(f"PDF file not found at {pdf_path}")
        return

    LOGGER.info(f"Extracting images from {pdf_path} to {output_dir}...")
    
    try:
        # We don't need OCR for image extraction, so we can set engine to None or keep defaults
        # Enable render_vector_pages=True to capture diagrams drawn as vectors
        pages = extract_pdf_content(pdf_path, output_dir, render_vector_pages=True)
        
        total_images = sum(len(page[2]) for page in pages)
        LOGGER.info(f"Extraction complete. Found {total_images} images across {len(pages)} pages.")
        
    except Exception as e:
        LOGGER.error(f"An error occurred during extraction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
