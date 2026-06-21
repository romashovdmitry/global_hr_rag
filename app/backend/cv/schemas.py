"""Pydantic schemas for CV upload and processing responses."""

from pydantic import BaseModel


class CVUploadResponse(BaseModel):
    """Response returned after a successful CV upload."""

    session_id: str
    text_length: int
    preview: str


class CVStatusResponse(BaseModel):
    """Response returned when checking whether a CV is attached."""

    has_cv: bool
