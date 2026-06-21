"""CV upload and management endpoints."""

import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.chat import repository as repo
from backend.core.config import CONVERSATION_ID
from backend.cv.schemas import CVStatusResponse, CVUploadResponse
from backend.cv.service import process_cv_upload
from backend.db.session import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cv", tags=["CV"])

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@router.post(
    "/upload",
    response_model=CVUploadResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload a CV PDF",
    description=(
        "Accepts a PDF file, extracts plain text, and attaches it as context "
        "for all subsequent RAG queries. "
        "Maximum file size: 10 MB. Scanned (image-only) PDFs are not supported."
    ),
)
async def upload_cv(
    file: UploadFile = File(..., description="PDF CV file"),
    db: AsyncSession = Depends(get_db),
) -> CVUploadResponse:
    """Process and attach a CV PDF to the conversation."""
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PDF files are accepted.",
        )

    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File exceeds the 10 MB size limit.",
        )

    try:
        cv_text = await process_cv_upload(db, CONVERSATION_ID, file_bytes)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except LookupError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    return CVUploadResponse(
        session_id=CONVERSATION_ID,
        text_length=len(cv_text),
        preview=cv_text[:200],
    )


@router.get(
    "/status",
    response_model=CVStatusResponse,
    summary="Check whether a CV is currently attached",
    description="Returns whether the conversation has CV text attached.",
)
async def cv_status(db: AsyncSession = Depends(get_db)) -> CVStatusResponse:
    """Return CV attachment status for the singleton conversation."""
    session = await repo.get_session(db, CONVERSATION_ID)
    return CVStatusResponse(has_cv=session is not None and session.cv_text is not None)


@router.delete(
    "/clear",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove the attached CV",
    description="Detaches the CV from the conversation. Subsequent queries will have no CV context.",
)
async def clear_cv(db: AsyncSession = Depends(get_db)) -> None:
    """Remove CV text from the singleton conversation."""
    session = await repo.get_session(db, CONVERSATION_ID)
    if session is not None:
        session.cv_text = None
        await db.commit()
