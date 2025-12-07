import fitz
from pathlib import Path

pdf_path = Path("data/raw/JCB_Service manual 3CX 4CX.pdf")
doc = fitz.open(pdf_path)

# Page 320 (index 319)
page_num = 320
page_index = page_num - 1
page = doc.load_page(page_index)

images = page.get_images(full=True)
drawings = page.get_drawings()

print(f"Page {page_num}:")
print(f"  Images found: {len(images)}")
print(f"  Drawings found: {len(drawings)}")

if len(drawings) > 0:
    print("  Page has vector graphics.")
