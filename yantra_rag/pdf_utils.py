"""PDF parsing helpers for text and image extraction."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple, Optional

import fitz  # PyMuPDF
from PIL import Image

from .data_models import ImageRecord
try:
    import pytesseract
except ImportError:  # pragma: no cover - optional dependency
    pytesseract = None


LOGGER = logging.getLogger(__name__)


def ensure_dir(path: Path) -> None:
    """Create the provided directory if it does not exist."""

    path.mkdir(parents=True, exist_ok=True)


def _normalize_pixmap(pix: fitz.Pixmap) -> Optional[fitz.Pixmap]:
    """Ensure pixmap is RGB or grayscale without alpha so PNG save succeeds."""

    try:
        if pix.colorspace is None:
            # Handle masks/stencils by converting to grayscale
            try:
                return fitz.Pixmap(fitz.csGRAY, pix)
            except Exception:
                # If conversion fails, we can't save it easily
                return None
            
        if pix.n >= 5:
            pix = fitz.Pixmap(fitz.csRGB, pix)
        if pix.alpha:
            pix = fitz.Pixmap(pix, 0)
        if pix.n not in (1, 3):
            pix = fitz.Pixmap(fitz.csRGB, pix)
    except RuntimeError as exc:
        LOGGER.warning("Skipping image conversion due to error: %s", exc)
        return None
    return pix


def _pixmap_to_pil(pix: fitz.Pixmap) -> Optional[Image.Image]:
    pix = _normalize_pixmap(pix)
    if pix is None:
        return None
    mode = "RGB" if pix.n > 1 else "L"
    try:
        return Image.frombytes(mode, [pix.width, pix.height], pix.samples)
    except Exception as exc:  # pragma: no cover - PIL specific edge cases
        LOGGER.warning("Failed to construct PIL image: %s", exc)
        return None


def _apply_tesseract(page: fitz.Page, dpi: int) -> str:
    if pytesseract is None:
        raise ImportError(
            "pytesseract is not installed. Install it (and the Tesseract binary) to enable OCR."
        )
    pix = page.get_pixmap(dpi=dpi)
    pil_image = _pixmap_to_pil(pix)
    if pil_image is None:
        return ""
    try:
        return pytesseract.image_to_string(pil_image)
    except Exception as exc:
        LOGGER.warning("pytesseract failed on page %s: %s", page.number + 1, exc)
        return ""


def _run_ocr(page: fitz.Page, engine: str, dpi: int) -> str:
    if not engine or engine.lower() == "none":
        return ""
    if engine.lower() == "tesseract":
        return _apply_tesseract(page, dpi)
    LOGGER.warning("Unsupported OCR engine '%s' requested", engine)
    return ""


def _fix_broken_drop_caps(text: str) -> str:
    """Merge single-letter lines with the following line if it starts lowercase."""

    if not text:
        return text

    lines = text.splitlines()
    fixed_lines: List[str] = []
    idx = 0
    while idx < len(lines):
        current = lines[idx]
        stripped = current.strip()
        if len(stripped) == 1 and stripped.isalpha() and stripped.isupper():
            if idx + 1 < len(lines):
                nxt = lines[idx + 1]
                nxt_lstrip = nxt.lstrip()
                if nxt_lstrip and nxt_lstrip[0].islower():
                    fixed_lines.append(f"{stripped}{nxt_lstrip}")
                    idx += 2
                    continue
        fixed_lines.append(current)
        idx += 1
    return "\n".join(fixed_lines)


def _merge_text(existing: str, addition: str) -> str:
    addition = addition.strip()
    if not addition:
        return existing
    if addition in existing:
        return existing
    if existing:
        return f"{existing}\n{addition}".strip()
    return addition


def extract_pdf_content(
    pdf_path: Path,
    image_output_dir: Path,
    ocr_engine: Optional[str] = None,
    ocr_dpi: int = 300,
    ocr_min_chars: int = 40,
    ocr_always_on: bool = False,
    render_vector_pages: bool = False,
) -> List[Tuple[str, int, List[ImageRecord]]]:
    """Return per-page text and image metadata for the PDF."""

    ensure_dir(image_output_dir)
    doc = fitz.open(pdf_path)
    pages: List[Tuple[str, int, List[ImageRecord]]] = []

    for page_index in range(doc.page_count):
        page = doc.load_page(page_index)
        text = page.get_text("text").strip()
        text = _fix_broken_drop_caps(text)
        run_ocr = ocr_engine and ocr_engine.lower() != "none"
        needs_ocr = run_ocr and (ocr_always_on or len(text) < ocr_min_chars)
        if run_ocr and needs_ocr:
            ocr_text = _run_ocr(page, ocr_engine, ocr_dpi).strip()
            if ocr_text:
                text = _merge_text(text, ocr_text)
        page_number = page_index + 1
        images: List[ImageRecord] = []

        # 1. Extract embedded raster images
        for img_idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            
            # Try raw extraction first to avoid color profile issues and get original quality
            try:
                image_info = doc.extract_image(xref)
            except Exception:
                image_info = None

            if image_info:
                ext = image_info["ext"]
                image_data = image_info["image"]
                image_path = image_output_dir / f"{pdf_path.stem}_page{page_number}_img{img_idx + 1}.{ext}"
                try:
                    image_path.write_bytes(image_data)
                except Exception as exc:
                    LOGGER.warning("Failed to write raw image: %s", exc)
                    continue
            else:
                # Fallback to Pixmap if raw extraction isn't possible
                try:
                    pix = fitz.Pixmap(doc, xref)
                    pix = _normalize_pixmap(pix)
                    if pix is None:
                        continue
                    image_path = image_output_dir / f"{pdf_path.stem}_page{page_number}_img{img_idx + 1}.png"
                    pix.save(str(image_path))
                except Exception as exc:  # PyMuPDF raises custom exceptions
                    LOGGER.warning(
                        "Skipping image xref %s on page %s due to save error: %s",
                        xref,
                        page_number,
                        exc,
                    )
                    continue

            images.append(
                ImageRecord(
                    source_file=pdf_path.name,
                    page_number=page_number,
                    path=image_path,
                    caption=f"Image extracted from {pdf_path.name} page {page_number}",
                )
            )

        # 2. If no raster images found but vector graphics exist, render the page
        if not images and render_vector_pages:
            # Check for significant vector content (heuristic: > 20 drawing paths)
            drawings = page.get_drawings()
            if len(drawings) > 20:
                try:
                    # Render page at 150 DPI (good balance for screen)
                    pix = page.get_pixmap(dpi=150)
                    image_path = image_output_dir / f"{pdf_path.stem}_page{page_number}_render.png"
                    pix.save(str(image_path))
                    
                    images.append(
                        ImageRecord(
                            source_file=pdf_path.name,
                            page_number=page_number,
                            path=image_path,
                            caption=f"Diagram rendered from {pdf_path.name} page {page_number}",
                        )
                    )
                except Exception as exc:
                    LOGGER.warning("Failed to render page %s: %s", page_number, exc)

        pages.append((text, page_number, images))

    return pages
