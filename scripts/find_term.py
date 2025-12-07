import fitz
from pathlib import Path

pdf_path = Path("data/raw/JCB_Service manual 3CX 4CX.pdf")
doc = fitz.open(pdf_path)

term = "Pressure Test Adapters"
print(f"Searching for '{term}'...")

for page in doc:
    text = page.get_text()
    if term.lower() in text.lower():
        print(f"Found on page {page.number + 1}")
