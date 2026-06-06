"""OCR and text extraction for uploaded learner essays."""

from __future__ import annotations

import io
import logging
import os
import re
import shutil
from typing import Tuple

logger = logging.getLogger(__name__)

IMAGE_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif", "image/bmp", "image/tiff"}
PDF_TYPES = {"application/pdf"}
TEXT_TYPES = {"text/plain"}

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".pdf", ".txt"}


def _normalise_text(raw: str) -> str:
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _tesseract_available() -> bool:
    cmd = os.environ.get("TESSERACT_CMD", "tesseract")
    return shutil.which(cmd) is not None


def _ocr_image_bytes(data: bytes) -> str:
    import pytesseract
    from PIL import Image

    cmd = os.environ.get("TESSERACT_CMD", "").strip()
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
    if not _tesseract_available() and not cmd:
        raise RuntimeError(
            "Tesseract OCR not found. Install it (e.g. dnf install tesseract) "
            "or set TESSERACT_CMD to the binary path."
        )
    lang = os.environ.get("CAMBRIDGE_OCR_LANG", "eng").strip() or "eng"
    img = Image.open(io.BytesIO(data))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    return pytesseract.image_to_string(img, lang=lang)


def _extract_pdf(data: bytes) -> Tuple[str, str]:
    try:
        import fitz
    except ImportError as e:
        raise RuntimeError(
            "PDF support requires pymupdf. Run: pip install pymupdf"
        ) from e

    doc = fitz.open(stream=data, filetype="pdf")
    parts: list[str] = []
    for page in doc:
        parts.append(page.get_text("text"))
    joined = _normalise_text("\n\n".join(parts))
    if len(joined) >= 40:
        return joined, "pdf_text"

    ocr_parts: list[str] = []
    for page in doc:
        pix = page.get_pixmap(dpi=200, alpha=False)
        from PIL import Image

        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        ocr_parts.append(_ocr_image_bytes(buf.getvalue()))
    doc.close()
    return _normalise_text("\n\n".join(ocr_parts)), "ocr"


def transcribe_upload(filename: str, content_type: str | None, data: bytes) -> dict:
    ext = os.path.splitext(filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type ({ext or 'unknown'}). "
            f"Use: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
    if len(data) > 20 * 1024 * 1024:
        raise ValueError("File too large (max 20 MB).")

    if ext == ".txt" or content_type in TEXT_TYPES:
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return {"text": _normalise_text(data.decode(enc)), "method": "text"}
            except UnicodeDecodeError:
                continue
        raise ValueError("Could not decode text file.")

    if ext == ".pdf" or content_type in PDF_TYPES:
        text, method = _extract_pdf(data)
        if not text:
            raise ValueError("No text found in PDF.")
        return {"text": text, "method": method}

    if ext in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"} or content_type in IMAGE_TYPES:
        text = _normalise_text(_ocr_image_bytes(data))
        if not text:
            raise ValueError("OCR returned empty text — try a clearer scan.")
        return {"text": text, "method": "ocr"}

    raise ValueError(f"Unsupported upload: {filename}")
