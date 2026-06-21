"""Tests for vacancy schema validation and ingestion logic."""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from backend.vacancies.schemas import VacancyCreate, IngestResponse
from backend.vacancies.service import ingest_from_file
from backend.utils.qdrant import vacancy_id


class TestVacancySchema:
    """Unit tests for VacancyCreate Pydantic schema."""

    def test_valid_vacancy_passes(self, sample_vacancies):
        """A well-formed vacancy record validates without errors."""
        vacancy = VacancyCreate.model_validate(sample_vacancies[0])
        assert vacancy.role
        assert vacancy.description_of_vacancy

    def test_missing_role_raises(self):
        """A vacancy without a role field must raise ValidationError."""
        with pytest.raises(ValidationError):
            VacancyCreate.model_validate(
                {
                    "description_of_vacancy": "Some description",
                    "salary": 3000,
                    "company": "TestCo",
                    "contacts": "test@test.com",
                    "status_of_role": ["Remote"],
                }
            )

    def test_empty_description_raises(self):
        """A vacancy with an empty description must raise ValidationError."""
        with pytest.raises(ValidationError):
            VacancyCreate.model_validate(
                {
                    "role": "Python Dev",
                    "description_of_vacancy": "",
                    "salary": 3000,
                    "company": "TestCo",
                    "contacts": "test@test.com",
                    "status_of_role": ["Remote"],
                }
            )

    def test_status_string_coerced_to_list(self):
        """A plain string in status_of_role is coerced to a single-element list."""
        vacancy = VacancyCreate.model_validate(
            {
                "role": "Python Dev",
                "description_of_vacancy": "Backend development with Python.",
                "salary": 3000,
                "company": "TestCo",
                "contacts": "test@test.com",
                "status_of_role": "Remote",
            }
        )
        assert vacancy.status_of_role[0].value == "Remote"

    def test_null_salary_is_allowed(self, sample_vacancies):
        """Null salary is a valid value per the schema."""
        record = dict(sample_vacancies[0])
        record["salary"] = None
        vacancy = VacancyCreate.model_validate(record)
        assert vacancy.salary is None

    def test_invalid_status_raises(self):
        """An unrecognised status value must raise ValidationError."""
        with pytest.raises(ValidationError):
            VacancyCreate.model_validate(
                {
                    "role": "Python Dev",
                    "description_of_vacancy": "Valid description.",
                    "salary": 3000,
                    "company": "TestCo",
                    "contacts": "test@test.com",
                    "status_of_role": ["Freelance"],  # not in enum
                }
            )


class TestVacancyId:
    """Tests for the deterministic vacancy ID generator."""

    def test_same_input_produces_same_id(self):
        """Two calls with identical role and company return the same UUID."""
        v = {"role": "Python Dev", "company": "Acme"}
        assert vacancy_id(v) == vacancy_id(v)

    def test_different_input_produces_different_id(self):
        """Different role/company combinations produce different UUIDs."""
        v1 = {"role": "Python Dev", "company": "Acme"}
        v2 = {"role": "Go Dev", "company": "Acme"}
        assert vacancy_id(v1) != vacancy_id(v2)


class TestIngestionService:
    """Integration-level tests for ingest_from_file."""

    @pytest.mark.asyncio
    async def test_ingest_counts_valid_and_invalid(self, mock_qdrant_client):
        """Ingestion correctly separates valid records from validation errors."""
        test_data = [
            {
                "role": "Python Dev",
                "description_of_vacancy": "Backend with Python and FastAPI.",
                "salary": 5000,
                "company": "Acme",
                "contacts": "hr@acme.com",
                "status_of_role": ["Full-time", "Remote"],
            },
            # Fault: missing role
            {
                "description_of_vacancy": "Some work.",
                "salary": 2000,
                "company": "BadCo",
                "contacts": "hr@bad.co",
                "status_of_role": ["Remote"],
            },
            # Fault: empty description
            {
                "role": "DevOps",
                "description_of_vacancy": "",
                "salary": 4000,
                "company": "OtherCo",
                "contacts": "hr@other.co",
                "status_of_role": ["Hybrid"],
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            tmp_path = f.name

        with (
            patch("backend.vacancies.service.embed_dense", new_callable=AsyncMock,
                  return_value=[[0.0] * 1536]),
            patch("backend.vacancies.service.embed_sparse_async", new_callable=AsyncMock,
                  return_value=[{"indices": [1], "values": [1.0]}]),
        ):
            result = await ingest_from_file(mock_qdrant_client, path=tmp_path)

        assert result.total == 3
        assert result.ingested == 1
        assert len(result.errors) == 2
