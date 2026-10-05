import io
import os
import logging
from typing import Dict, Any
from pypdf import PdfReader

logger = logging.getLogger("DocumentProcessor")

# Maximum characters to pass to the LLM to stay safely within context & rate limits
MAX_EXTRACTED_CHARS = 25000


def extract_document_content(file_bytes: bytes, filename: str = "document.pdf") -> Dict[str, Any]:
    """
    Extracts text content from user-uploaded documents (PDF, TXT, CSV, MD).
    
    Returns a dictionary:
    {
        "status": "success" | "scanned_or_empty" | "encrypted" | "unsupported" | "error",
        "filename": filename,
        "page_count": int,
        "text": str,
        "has_images": bool,
        "error_message": Optional[str]
    }
    """
    if not file_bytes:
        return {
            "status": "error",
            "filename": filename,
            "page_count": 0,
            "text": "",
            "has_images": False,
            "error_message": "Empty file received."
        }

    ext = os.path.splitext(filename)[1].lower()

    # -------------------------------------------------------------
    # 1. Plain Text Documents (.txt, .csv, .md, .json, .log, .py)
    # -------------------------------------------------------------
    if ext in [".txt", ".csv", ".md", ".json", ".log", ".py", ".html", ".xml"]:
        try:
            try:
                decoded_text = file_bytes.decode("utf-8").strip()
            except UnicodeDecodeError:
                decoded_text = file_bytes.decode("latin-1", errors="replace").strip()

            if len(decoded_text) > MAX_EXTRACTED_CHARS:
                decoded_text = decoded_text[:MAX_EXTRACTED_CHARS] + "\n\n...[Document truncated for length]"

            return {
                "status": "success",
                "filename": filename,
                "page_count": 1,
                "text": decoded_text,
                "has_images": False,
                "error_message": None
            }
        except Exception as e:
            return {
                "status": "error",
                "filename": filename,
                "page_count": 0,
                "text": "",
                "has_images": False,
                "error_message": f"Could not read text file: {e}"
            }

    # -------------------------------------------------------------
    # 2. PDF Documents (.pdf)
    # -------------------------------------------------------------
    if ext == ".pdf" or ext == "":
        try:
            stream = io.BytesIO(file_bytes)
            reader = PdfReader(stream)

            # Check for password protection
            if reader.is_encrypted:
                return {
                    "status": "encrypted",
                    "filename": filename,
                    "page_count": len(reader.pages),
                    "text": "",
                    "has_images": False,
                    "error_message": "Document is password protected."
                }

            page_count = len(reader.pages)
            extracted_pages = []
            has_images = False

            for idx, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    extracted_pages.append(f"--- Page {idx} ---\n{page_text.strip()}")
                
                # Check if page contains images
                if not has_images and len(page.images) > 0:
                    has_images = True

            combined_text = "\n\n".join(extracted_pages).strip()

            # Detection for Scanned / Image-Only PDFs
            if len(combined_text) < 50:
                return {
                    "status": "scanned_or_empty",
                    "filename": filename,
                    "page_count": page_count,
                    "text": "",
                    "has_images": has_images,
                    "error_message": "Scanned or image-only PDF with no digital text."
                }

            # Enforce max length guard to protect LLM context window
            if len(combined_text) > MAX_EXTRACTED_CHARS:
                combined_text = combined_text[:MAX_EXTRACTED_CHARS] + f"\n\n...[Truncated: showing first {MAX_EXTRACTED_CHARS} characters of {page_count} pages]"

            return {
                "status": "success",
                "filename": filename,
                "page_count": page_count,
                "text": combined_text,
                "has_images": has_images,
                "error_message": None
            }

        except Exception as e:
            logger.error(f"Error parsing PDF {filename}: {e}", exc_info=True)
            return {
                "status": "error",
                "filename": filename,
                "page_count": 0,
                "text": "",
                "has_images": False,
                "error_message": f"Corrupted or unreadable PDF: {str(e)}"
            }

    # -------------------------------------------------------------
    # 3. Unsupported Formats (.zip, .exe, .bin, etc.)
    # -------------------------------------------------------------
    return {
        "status": "unsupported",
        "filename": filename,
        "page_count": 0,
        "text": "",
        "has_images": False,
        "error_message": f"Unsupported file format '{ext}'. Supported formats: PDF, TXT, CSV, MD, JSON."
    }
