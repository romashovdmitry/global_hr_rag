"""Pydantic schemas for vacancy ingestion and response serialization."""

from enum import Enum

from pydantic import BaseModel, field_validator


class RoleStatus(str, Enum):
    """Allowed values for the status_of_role payload field."""

    full_time = "Full-time"
    part_time = "Part-time"
    project = "Project"
    remote = "Remote"
    hybrid = "Hybrid"


class VacancyCreate(BaseModel):
    """Schema for a single vacancy record as it appears in vacancies.json."""

    role: str
    description_of_vacancy: str
    salary: int | None = None
    company: str
    contacts: str
    status_of_role: list[RoleStatus]

    @field_validator("description_of_vacancy")
    @classmethod
    def description_must_not_be_empty(cls, v: str) -> str:
        """Reject vacancies with no embeddable text."""
        if not v.strip():
            raise ValueError("description_of_vacancy must not be empty.")
        return v

    @field_validator("status_of_role", mode="before")
    @classmethod
    def coerce_status_to_list(cls, v: object) -> list:
        """Accept both a list and a bare string for status_of_role."""
        if isinstance(v, str):
            return [v]
        return v  # type: ignore[return-value]


class IngestResponse(BaseModel):
    """Response returned after a bulk ingestion request."""

    total: int
    ingested: int
    skipped: int
    errors: list[str]
