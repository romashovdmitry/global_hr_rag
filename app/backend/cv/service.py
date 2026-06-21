"""CV processing service — PDF text extraction and session attachment."""

from sqlalchemy.ext.asyncio import AsyncSession

from backend.chat import repository
from backend.utils.pdf import extract_text_from_pdf


async def process_cv_upload(
    db: AsyncSession,
    session_id: str,
    file_bytes: bytes,
) -> str:
    """Extract text from a PDF CV and attach it to the given session.

    Args:
        db: Active async database session.
        session_id: Target session UUID to attach the CV to.
        file_bytes: Raw bytes of the uploaded PDF.

    Returns:
        Extracted plain text from the PDF.

    Raises:
        ValueError: If the PDF is unreadable or contains no extractable text.
        LookupError: If the session does not exist.
    """
    cv_text = extract_text_from_pdf(file_bytes)

    session = await repository.update_session_cv(db, session_id, cv_text)
    if session is None:
        raise LookupError(f"Session '{session_id}' not found.")

    return cv_text
