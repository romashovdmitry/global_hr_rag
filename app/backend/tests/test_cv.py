"""Tests for CV upload, PDF text extraction, and singleton-conversation attachment.

Single-user mode: the CV upload endpoint operates on the singleton CONVERSATION_ID;
there is no per-request session id in the URL.
"""

from unittest.mock import patch

import pytest

from backend.core.config import CONVERSATION_ID


def _make_minimal_pdf(text: str) -> bytes:
    """Build a minimal single-page PDF containing *text* (no external deps).

    Produces a structurally valid PDF with correct xref byte offsets so that
    ``pypdf`` can parse it and extract the embedded text.
    """
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
    ]
    stream = b"BT /F1 24 Tf 72 700 Td (" + text.encode("latin-1") + b") Tj ET"
    objects.append(
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n"
        + stream + b"\nendstream"
    )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    pdf = b"%PDF-1.4\n"
    offsets: list[int] = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf += str(i).encode() + b" 0 obj\n" + obj + b"\nendobj\n"

    xref_pos = len(pdf)
    pdf += b"xref\n0 " + str(len(objects) + 1).encode() + b"\n"
    pdf += b"0000000000 65535 f \n"
    for off in offsets:
        pdf += ("%010d 00000 n \n" % off).encode()
    pdf += b"trailer\n<< /Size " + str(len(objects) + 1).encode() + b" /Root 1 0 R >>\n"
    pdf += b"startxref\n" + str(xref_pos).encode() + b"\n%%EOF"
    return pdf


async def _ensure_singleton_conversation(db_session) -> None:
    """Create the singleton conversation row required by CV endpoints."""
    from backend.chat.models.session import Session

    db_session.add(Session(id=CONVERSATION_ID))
    await db_session.commit()


class TestPDFExtraction:
    """Unit tests for the PDF text extraction utility."""

    def test_extract_text_from_valid_pdf(self):
        """A well-formed PDF with text returns non-empty extracted content."""
        from backend.utils.pdf import extract_text_from_pdf

        pdf_bytes = _make_minimal_pdf("Python Developer with FastAPI experience")
        text = extract_text_from_pdf(pdf_bytes)
        assert "Python" in text

    def test_empty_pdf_raises(self):
        """A truncated PDF with no extractable text raises an error."""
        from backend.utils.pdf import extract_text_from_pdf

        with pytest.raises(Exception):
            extract_text_from_pdf(b"%PDF-1.4\n")

    def test_non_pdf_bytes_raises(self):
        """Passing non-PDF bytes raises an error."""
        from backend.utils.pdf import extract_text_from_pdf

        with pytest.raises(Exception):
            extract_text_from_pdf(b"this is not a pdf")


class TestCVUploadEndpoint:
    """Integration tests for the session-free CV upload REST endpoint."""

    @pytest.mark.asyncio
    async def test_upload_valid_pdf(self, client, db_session):
        """A valid PDF upload returns 200 with a non-zero text_length."""
        await _ensure_singleton_conversation(db_session)
        cv_text_mock = "Senior Python Developer with 7 years of experience."

        with patch(
            "backend.cv.service.extract_text_from_pdf",
            return_value=cv_text_mock,
        ):
            response = await client.post(
                "/api/v1/cv/upload",
                files={"file": ("cv.pdf", b"%PDF-fake-content", "application/pdf")},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == CONVERSATION_ID
        assert data["text_length"] == len(cv_text_mock)
        assert cv_text_mock[:200] in data["preview"]

    @pytest.mark.asyncio
    async def test_upload_non_pdf_returns_415(self, client, db_session):
        """Uploading a non-PDF file returns 415 Unsupported Media Type."""
        await _ensure_singleton_conversation(db_session)

        response = await client.post(
            "/api/v1/cv/upload",
            files={"file": ("cv.txt", b"plain text content", "text/plain")},
        )
        assert response.status_code == 415

    @pytest.mark.asyncio
    async def test_upload_empty_pdf_returns_422(self, client, db_session):
        """Uploading a PDF with no extractable text returns 422."""
        await _ensure_singleton_conversation(db_session)

        with patch(
            "backend.cv.service.extract_text_from_pdf",
            side_effect=ValueError("No extractable text found in PDF."),
        ):
            response = await client.post(
                "/api/v1/cv/upload",
                files={"file": ("cv.pdf", b"%PDF-fake", "application/pdf")},
            )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_cv_status_and_clear(self, client, db_session):
        """CV status reflects attachment, and clear detaches it."""
        await _ensure_singleton_conversation(db_session)
        cv_text_mock = "Backend engineer, Python, Go."

        with patch(
            "backend.cv.service.extract_text_from_pdf",
            return_value=cv_text_mock,
        ):
            await client.post(
                "/api/v1/cv/upload",
                files={"file": ("cv.pdf", b"%PDF-fake", "application/pdf")},
            )

        status_resp = await client.get("/api/v1/cv/status")
        assert status_resp.status_code == 200
        assert status_resp.json()["has_cv"] is True

        clear_resp = await client.delete("/api/v1/cv/clear")
        assert clear_resp.status_code == 204

        status_after = await client.get("/api/v1/cv/status")
        assert status_after.json()["has_cv"] is False
