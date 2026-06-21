"""PDF text extraction utilities."""

import io

from pypdf import PdfReader


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract plain text from a PDF file given as raw bytes.

    Args:
        file_bytes: Raw bytes of the PDF file.

    Returns:
        Concatenated text content from all pages.

    Raises:
        ValueError: If the PDF cannot be read or contains no extractable text.
    """
    reader = PdfReader(io.BytesIO(file_bytes))
    if not reader.pages:
        raise ValueError("PDF contains no pages.")

    pages_text = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(pages_text).strip()

    if not text:
        raise ValueError(
            "No extractable text found in PDF. "
            "Scanned (image-only) PDFs are not supported."
        )
    return text
