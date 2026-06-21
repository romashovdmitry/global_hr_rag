"""LangGraph node: extract hard-filter metadata from the user query."""

import logging

from pydantic import BaseModel

from backend.ml.llm_service import get_llm_service
from backend.ml.state import RAGState
from backend.ml.timing import timed_node

logger = logging.getLogger(__name__)

# Qdrant payload index only supports these exact values for status_of_role.
# Any value outside this set will match nothing and silently empty the results.
_VALID_STATUSES = frozenset({"Full-time", "Part-time", "Project", "Remote", "Hybrid"})

# Keywords that indicate the user actually mentioned salary.
# If none are present, any LLM-produced salary filter is treated as hallucination.
_SALARY_KEYWORDS = frozenset({
    "salary", "salaries", "pay", "paid", "wage", "compensation", "earn",
    "зарплата", "зарплат", "оклад", "платят", "платит",
    "$", "€", "usd", "eur", "k/month", "k/year", "тысяч",
})

# Keywords that indicate the user mentioned work format / employment type.
_STATUS_KEYWORDS = frozenset({
    "remote", "remotely", "full-time", "full time", "part-time", "part time",
    "hybrid", "office", "project", "freelance", "contract",
    "удалённо", "удалённая", "офис", "гибрид", "фриланс",
})

# Geographic tokens that are NOT stored in our vacancy data.
# Checked as substrings in the lowercased query.
_LOCATION_TOKENS: list[str] = [
    # Continents / macro-regions
    "europe", "asia", "africa", "australia", "oceania",
    "north america", "south america", "latin america",
    "eastern europe", "western europe", "southeast asia", "middle east",
    # EU
    "european union", " eu ", " eu,",
    # Countries (most common in IT job searches)
    "germany", "france", " uk ", "england", "britain", "spain", "italy",
    "netherlands", "holland", "poland", "ukraine", "russia", "turkey",
    "usa", "united states", "canada", "india", "china", "brazil",
    "singapore", "japan", "australia", "switzerland", "austria", "sweden",
    "norway", "denmark", "finland", "belgium", "portugal", "czechia",
    "czech republic", "slovakia", "hungary", "romania", "bulgaria",
    # Cities
    "berlin", "london", "paris", "amsterdam", "warsaw", "kyiv", "moscow",
    "new york", "san francisco", "toronto", "bangalore", "mumbai", "dubai",
    "zurich", "vienna", "stockholm", "oslo", "copenhagen", "helsinki",
    "prague", "budapest", "bucharest", "lisbon", "madrid", "barcelona",
    "milan", "rome",
]

# Benefit / perk tokens that are NOT stored in our vacancy data.
_BENEFIT_TOKENS: list[str] = [
    "health insurance", "medical insurance", "dental", "vision insurance",
    "relocation", "relocation package",
    "stock option", "stock options", "equity", " esop", "shares",
    "annual bonus", "signing bonus",
    "gym", "fitness", "wellness",
    "free lunch", "meal allowance", "food allowance",
    "childcare", "day care",
    "visa sponsorship", "work permit", "work visa",
    "team retreat", "company retreat",
]

# Company-characteristic tokens not stored in our data.
_COMPANY_TOKENS: list[str] = [
    "startup", "startups",
    "enterprise company", "large company", "big company",
    "small company", "small team",
    "people in company", "employees",
    "series a", "series b", "seed stage",
]


class MetadataFilters(BaseModel):
    """Structured output from the metadata extractor LLM call.

    Note: unsupported_criteria is detected programmatically (not by LLM) to
    avoid non-deterministic output from small models.
    """

    status_of_role: list[str] | None = None
    salary_min: int | None = None
    salary_max: int | None = None
    exclude_skills: list[str] | None = None


_SYSTEM = """You are a structured data extractor.
Given a user message, optional conversation history, and optional CV, extract job search filters:

- status_of_role: subset of ["Full-time", "Part-time", "Project", "Remote", "Hybrid"] — only include values explicitly mentioned.
- salary_min / salary_max: integer monthly USD/EUR amounts if mentioned, else null.
- exclude_skills: list of technologies, frameworks, or tools the user explicitly wants to EXCLUDE.
  These are negative constraints expressed with phrases like:
  "without Django", "not Flask", "no Java", "except Kubernetes", "avoiding AWS".
  Examples: "Python but not Django" → exclude_skills: ["Django"]
            "backend without Java and PHP" → exclude_skills: ["Java", "PHP"]
  If no exclusions are mentioned, return null or empty list.

If the user refers to a previous message (e.g. "same as before", "add remote"), take filters from context.
Respond with JSON matching the schema. If no filter is mentioned, return empty lists / null values."""


def _detect_unsupported_criteria(query_lower: str) -> list[str]:
    """Programmatically detect criteria the user mentioned that are not in our data.

    Uses deterministic keyword matching instead of LLM to avoid non-determinism.
    Returns human-readable labels suitable for displaying to the user.
    """
    found: list[str] = []

    detected_locations = [tok for tok in _LOCATION_TOKENS if tok in query_lower]
    if detected_locations:
        # Use the first / most specific match as the label
        label = detected_locations[0].strip().title()
        found.append(f"location ({label})")

    detected_benefits = [tok for tok in _BENEFIT_TOKENS if tok in query_lower]
    if detected_benefits:
        label = detected_benefits[0].strip()
        found.append(f"benefit ({label})")

    detected_company = [tok for tok in _COMPANY_TOKENS if tok in query_lower]
    if detected_company:
        label = detected_company[0].strip()
        found.append(f"company characteristic ({label})")

    return found


def _format_history(state: RAGState) -> str:
    """Format the full chat_history (summary entry + recent messages) as plain text."""
    history = state.get("chat_history") or []
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history)


@timed_node("metadata_extractor")
async def metadata_extractor_node(state: RAGState) -> dict:
    """Extract hard metadata filters from the user query and CV context.

    Args:
        state: Current pipeline state.

    Returns:
        Partial state update with ``metadata_filters`` and a status entry.
    """
    # Use the heavy model: the 1B model hallucinates filters that were never
    # mentioned (e.g. salary_min=60000 for a plain role query), which silently
    # empties Qdrant results. Correctness matters more than speed here.
    llm = get_llm_service().get_llm().with_structured_output(MetadataFilters)

    context = f"User query: {state['user_query']}"
    if history_text := _format_history(state):
        context = f"Recent conversation:\n{history_text}\n\n{context}"
    if state.get("cv_text"):
        context += f"\n\nCV snippet (first 500 chars): {state['cv_text'][:500]}"

    result: MetadataFilters = await llm.ainvoke(
        [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": context}]
    )

    logger.info(
        "metadata_extractor raw output: status_of_role=%s salary_min=%s "
        "salary_max=%s exclude_skills=%s",
        result.status_of_role,
        result.salary_min,
        result.salary_max,
        result.exclude_skills,
    )

    query_lower = state["user_query"].lower()

    # Programmatic detection — deterministic, not affected by LLM non-determinism.
    unsupported_criteria = _detect_unsupported_criteria(query_lower)
    if unsupported_criteria:
        logger.info(
            "metadata_extractor: unsupported criteria detected programmatically: %s",
            unsupported_criteria,
        )

    filters: dict = {}

    # Only apply status_of_role if the user explicitly mentioned a work format
    # AND the extracted values are from the valid Qdrant keyword set.
    user_mentioned_status = any(kw in query_lower for kw in _STATUS_KEYWORDS)
    if result.status_of_role and user_mentioned_status:
        valid = [s for s in result.status_of_role if s in _VALID_STATUSES]
        if valid:
            filters["status_of_role"] = valid
        else:
            logger.warning(
                "metadata_extractor: status_of_role values all invalid (%s) — dropped.",
                result.status_of_role,
            )
    elif result.status_of_role and not user_mentioned_status:
        logger.warning(
            "metadata_extractor: status_of_role=%s extracted but no status "
            "keyword in query — dropping to avoid spurious filtering.",
            result.status_of_role,
        )

    # Only apply salary filters if the user actually mentioned salary.
    user_mentioned_salary = any(kw in query_lower for kw in _SALARY_KEYWORDS)
    if user_mentioned_salary:
        if result.salary_min is not None:
            filters["salary_min"] = result.salary_min
        if result.salary_max is not None:
            filters["salary_max"] = result.salary_max
    elif result.salary_min is not None or result.salary_max is not None:
        logger.warning(
            "metadata_extractor: salary filter (min=%s max=%s) extracted but "
            "no salary keyword in query — dropping to prevent empty retrieval.",
            result.salary_min,
            result.salary_max,
        )

    exclude_skills = [s.strip() for s in (result.exclude_skills or []) if s.strip()]

    status_msg = "Extracting search filters…"
    if exclude_skills:
        status_msg += f" (excluding: {', '.join(exclude_skills)})"

    return {
        "metadata_filters": filters,
        "exclude_skills": exclude_skills,
        "unsupported_criteria": unsupported_criteria,
        "status_stream": state.get("status_stream", []) + [status_msg],
    }
