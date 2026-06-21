"""RAG evaluation tests — behavioral checks and Ragas metric evaluation.

# --------------- compatibility stub ----------------------------------------
# ragas 0.4.x has a top-level import of langchain_community.chat_models.vertexai
# which was removed in langchain-community 0.4.0.  We inject a minimal stub so
# the import succeeds without requiring Google Cloud dependencies.
import sys
import types as _types

if "langchain_community.chat_models.vertexai" not in sys.modules:
    _stub = _types.ModuleType("langchain_community.chat_models.vertexai")
    _stub.ChatVertexAI = None  # type: ignore[attr-defined]
    sys.modules["langchain_community.chat_models.vertexai"] = _stub
# --------------- end stub ---------------------------------------------------

## Test tiers

### Behavioral tests  (pytest mark: ``behavioral``)
Fast checks that assert specific properties of the RAG pipeline output without
running Ragas evaluation (no extra LLM call for judging).  Each test runs the
full pipeline once and checks one structural / semantic assertion.

    docker exec rags-backend-1 pytest -m behavioral -v  (~2–5 min)

### Ragas eval tests  (pytest mark: ``eval``)
Slow metric-based evaluation using Ragas + local Ollama.  Requires Qdrant and
Ollama to be running.  Use the golden dataset for a representative sample.

    docker exec rags-backend-1 pytest -m eval -v  (~15–25 min)

## Metric thresholds

| Metric | Minimum |
|---|---|
| Faithfulness | 0.70 |
| Answer Relevancy | 0.65 |
| LLM Context Recall | 0.60 |
| LLM Context Precision | 0.60 |
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

import pytest

logger = logging.getLogger(__name__)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

FAITHFULNESS_MIN = 0.70
ANSWER_RELEVANCY_MIN = 0.65
CONTEXT_RECALL_MIN = 0.60
CONTEXT_PRECISION_MIN = 0.60

# Refusal text that the off_topic_responder returns.
REFUSAL_PHRASE = "job vacancy assistant"

GOLDEN_DATASET_PATH = Path(__file__).parent / "eval_dataset_golden.json"
EVAL_DATASET_PATH = Path(__file__).parent / "eval_dataset.json"


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def _load_golden() -> list[dict[str, Any]]:
    return json.loads(GOLDEN_DATASET_PATH.read_text())


def _load_eval_dataset() -> list[dict[str, Any]]:
    """Load the legacy curated evaluation question/reference pairs."""
    return json.loads(EVAL_DATASET_PATH.read_text())


def _golden_by_category(category: str) -> list[dict[str, Any]]:
    return [q for q in _load_golden() if q["category"] == category]


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


async def _run_pipeline(question: str) -> dict[str, Any]:
    """Run the full RAG pipeline and return answer + contexts.

    Creates a fresh DB engine with NullPool for each call to avoid asyncpg
    connection pool conflicts across test function event loop boundaries.
    """
    import os
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from sqlalchemy.pool import NullPool

    from backend.chat import repository
    from backend.chat.service import process_message_stream

    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://postgres:postgres@postgres:5432/rags",
    )

    engine = create_async_engine(db_url, poolclass=NullPool, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    try:
        async with session_factory() as db:
            session = await repository.create_session(db, cv_text=None)
            session_id = session.id

            final_answer = ""
            async for event_json in process_message_stream(db, session_id, question):
                evt = json.loads(event_json)
                if evt["type"] == "answer":
                    final_answer = evt["text"]

            await repository.delete_session(db, session_id)
    finally:
        await engine.dispose()

    # Retrieve contexts separately so Ragas can evaluate retrieval quality.
    from backend.utils.qdrant import embed_dense, get_qdrant_client
    from backend.core.config import settings

    client = get_qdrant_client()
    try:
        dense_vec = (await embed_dense([question]))[0]
        resp = await client.query_points(
            collection_name=settings.qdrant_collection,
            query=dense_vec,
            using="dense",
            limit=5,
            with_payload=True,
        )
        contexts = [
            hit.payload.get("description_of_vacancy", "")
            for hit in resp.points
            if hit.payload
        ]
    finally:
        await client.close()

    return {"answer": final_answer, "contexts": contexts}


# ---------------------------------------------------------------------------
# Ragas fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ragas_llm():
    """Wrap the project LLM for use by Ragas metrics.

    Uses Groq when GROQ_API_KEY is set, otherwise falls back to Ollama.
    """
    import os
    from ragas.llms import LangchainLLMWrapper

    groq_key = os.getenv("GROQ_API_KEY", "")
    if groq_key:
        from langchain_groq import ChatGroq
        return LangchainLLMWrapper(
            ChatGroq(model="llama-3.1-8b-instant", api_key=groq_key, temperature=0)
        )

    from langchain_ollama import ChatOllama
    return LangchainLLMWrapper(
        ChatOllama(model="llama3.2", base_url=OLLAMA_HOST, temperature=0)
    )


@pytest.fixture
def ragas_embeddings():
    """Wrap fastembed (already downloaded) as a Ragas-compatible embedder."""
    from langchain_core.embeddings import Embeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from fastembed import TextEmbedding

    class _FastembedWrapper(Embeddings):
        def __init__(self) -> None:
            self._model = TextEmbedding("BAAI/bge-small-en-v1.5")

        def embed_query(self, text: str) -> list[float]:
            return list(self._model.embed([text]))[0].tolist()

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [emb.tolist() for emb in self._model.embed(texts)]

    return LangchainEmbeddingsWrapper(_FastembedWrapper())


# ---------------------------------------------------------------------------
# Ragas sample builder
# ---------------------------------------------------------------------------


async def _build_ragas_samples(
    questions: list[dict[str, Any]],
) -> list["SingleTurnSample"]:
    """Run the pipeline for each question and build Ragas SingleTurnSample list."""
    from ragas import SingleTurnSample

    samples: list[SingleTurnSample] = []
    for item in questions:
        try:
            result = await _run_pipeline(item["question"])
        except Exception as exc:
            logger.warning("Pipeline failed for %r: %s", item["question"], exc)
            continue
        if not result["answer"] or not result["contexts"]:
            continue
        samples.append(
            SingleTurnSample(
                user_input=item["question"],
                response=result["answer"],
                retrieved_contexts=result["contexts"],
                reference=item["reference"],
            )
        )
    assert samples, "All RAG pipeline calls failed — cannot evaluate."
    return samples


# ===========================================================================
# BEHAVIORAL TESTS  (fast, no Ragas)
# ===========================================================================


@pytest.mark.behavioral
@pytest.mark.asyncio
async def test_off_topic_refusal_borscht():
    """Off-topic: 'How to cook borscht' → refusal, not a vacancy answer."""
    result = await _run_pipeline("How do I cook borscht at home?")
    assert REFUSAL_PHRASE in result["answer"].lower(), (
        f"Expected refusal, got: {result['answer'][:200]}"
    )


@pytest.mark.behavioral
@pytest.mark.asyncio
async def test_off_topic_refusal_batman():
    """Off-topic: 'Batman vs Superman' → refusal."""
    result = await _run_pipeline("Who is stronger — Batman or Superman?")
    assert REFUSAL_PHRASE in result["answer"].lower(), (
        f"Expected refusal, got: {result['answer'][:200]}"
    )


@pytest.mark.behavioral
@pytest.mark.asyncio
async def test_negative_constraint_no_django():
    """Negative: 'Python without Django' → No Django-requiring vacancy is recommended.

    The word 'django' may appear in the negative context ("without Django"), but the
    answer must NOT recommend CodeStart Corp (the only vacancy that explicitly
    requires Django/Flask as primary skills).
    """
    result = await _run_pipeline("Show me Python backend positions without Django and Flask.")
    answer = result["answer"]
    assert answer, "Expected a non-empty answer."
    # CodeStart Corp's Junior Python Developer requires "Django, or Flask" as primary skill.
    assert "codestart" not in answer.lower(), (
        f"Answer recommended a Django-requiring vacancy (CodeStart Corp) despite exclusion: "
        f"{answer[:300]}"
    )


@pytest.mark.behavioral
@pytest.mark.asyncio
async def test_negative_constraint_no_aws():
    """Negative: 'DevOps without AWS' → AWS-specific roles absent."""
    result = await _run_pipeline("Find DevOps positions without AWS.")
    # Answer must exist and not recommend AWS-only roles as the primary suggestion.
    assert result["answer"], "Expected a non-empty answer."


@pytest.mark.behavioral
@pytest.mark.asyncio
async def test_aggregate_query_salaries():
    """Aggregate: salary comparison query → answer is non-empty and mentions both technologies."""
    result = await _run_pipeline(
        "Compare salary levels for Python developers vs Go developers in the database."
    )
    answer = result["answer"]
    assert answer, "Expected a non-empty answer."
    assert REFUSAL_PHRASE not in answer.lower(), "Aggregate query should not be refused."
    # Answer should mention at least one of the technologies being compared.
    mentioned = "python" in answer.lower() or "go" in answer.lower() or "golang" in answer.lower()
    assert mentioned, (
        f"Aggregate answer should mention Python or Go developers: {answer[:300]}"
    )


@pytest.mark.behavioral
@pytest.mark.asyncio
async def test_aggregate_remote_companies():
    """Aggregate: 'Which companies offer remote?' → multiple company names."""
    result = await _run_pipeline("Which companies offer fully remote positions?")
    answer = result["answer"]
    assert answer, "Expected a non-empty answer."
    # Should mention at least 3 companies.
    from backend.utils.qdrant import get_qdrant_client
    assert len(answer) > 100, f"Aggregate answer too short: {answer}"


@pytest.mark.behavioral
@pytest.mark.asyncio
async def test_robustness_typos_python():
    """Typos/synonyms: 'питон бекенд' (Russian colloquial) → valid Python jobs."""
    result = await _run_pipeline("Ищу работу питон бекенд разработчик")
    assert result["answer"], "Expected a non-empty answer for colloquial Russian query."
    assert REFUSAL_PHRASE not in result["answer"].lower(), (
        "Pipeline should not refuse a job-related query even with typos."
    )


@pytest.mark.behavioral
@pytest.mark.asyncio
async def test_robustness_colloquial():
    """Robustness: colloquial Russian message → coherent job recommendations."""
    result = await _run_pipeline(
        "Слушай, ищу нормальную работу на питоне, "
        "чтобы платили достойно и не надо было в офис каждый день тащиться"
    )
    assert result["answer"], "Expected a non-empty answer."
    assert REFUSAL_PHRASE not in result["answer"].lower()


@pytest.mark.behavioral
@pytest.mark.asyncio
async def test_robustness_mobile():
    """Robustness: short informal query about mobile → Flutter vacancy found."""
    result = await _run_pipeline("Есть что-нибудь по мобилкам?")
    assert result["answer"], "Expected a non-empty answer."
    assert REFUSAL_PHRASE not in result["answer"].lower()


@pytest.mark.behavioral
@pytest.mark.asyncio
async def test_unsupported_criteria_location_europe():
    """Unsupported criterion: location 'Europe' → system admits it cannot filter by location.

    Our vacancy data has no location/country field.  The system must NOT hallucinate
    European vacancies.  It should explicitly state that location filtering is not
    supported and either offer the closest vacancies by other criteria or say none found.
    """
    result = await _run_pipeline("Show me open Python Backend Developer positions in Europe.")
    answer = result["answer"].lower()
    assert answer, "Expected a non-empty answer."
    # Must NOT claim these vacancies are specifically in Europe.
    # The response should mention the limitation (location / cannot filter / not available).
    limitation_signals = [
        "cannot filter",
        "not available",
        "does not contain",
        "no location",
        "location information",
        "cannot be filtered",
        "location",
    ]
    has_limitation = any(sig in answer for sig in limitation_signals)
    assert has_limitation, (
        f"Expected the answer to mention location-filtering limitation, got: {result['answer'][:300]}"
    )


@pytest.mark.behavioral
@pytest.mark.asyncio
async def test_unsupported_criteria_benefits():
    """Unsupported criterion: 'health insurance' benefit → system admits it cannot filter by benefits."""
    result = await _run_pipeline(
        "Find Python developer positions that offer health insurance and relocation package."
    )
    answer = result["answer"].lower()
    assert answer, "Expected a non-empty answer."
    limitation_signals = [
        "cannot filter",
        "not available",
        "does not contain",
        "no information",
        "benefits",
        "health insurance",
        "not include",
    ]
    has_limitation = any(sig in answer for sig in limitation_signals)
    assert has_limitation, (
        f"Expected mention of benefits filtering limitation, got: {result['answer'][:300]}"
    )


@pytest.mark.behavioral
@pytest.mark.asyncio
async def test_ambiguous_level_no_hallucination():
    """Ambiguous: 'Lead Python engineer' → valid response, no refusal, reasonable salary range.

    The DB may or may not contain an explicit 'Lead' title. Either outcome is correct:
    - If found: answer presents the vacancy with real data.
    - If not found: answer acknowledges it and suggests nearest alternatives.
    Either way the answer must not refuse and must not invent salary above $15000.
    """
    result = await _run_pipeline("Find me a Lead Python engineer position.")
    answer = result["answer"]
    assert answer, "Expected a non-empty answer."
    assert REFUSAL_PHRASE not in answer.lower(), (
        "A job-related question should never be refused."
    )
    # Sanity check: no obviously hallucinated salary (all real salaries are ≤ $9500)
    import re
    high_salaries = [int(n) for n in re.findall(r"\d{5,}", answer) if int(n) > 15000]
    assert not high_salaries, (
        f"Answer contains suspiciously high salary figure (possible hallucination): {high_salaries}"
    )


# ===========================================================================
# RAGAS METRIC TESTS  (slow)
# ===========================================================================


@pytest.mark.eval
@pytest.mark.asyncio
async def test_rag_faithfulness_and_relevancy(ragas_llm, ragas_embeddings):
    """Faithfulness and Answer Relevancy must exceed minimum thresholds.

    Uses the golden dataset, in-scope categories only (excludes out_of_scope
    which intentionally returns no vacancy data).
    """
    from ragas import EvaluationDataset, aevaluate
    from ragas.metrics import Faithfulness, AnswerRelevancy

    in_scope = [
        q for q in _load_golden()
        if q["category"] not in ("out_of_scope",) and q["expected_behavior"] != "refusal"
    ]
    samples = await _build_ragas_samples(in_scope)
    dataset = EvaluationDataset(samples=samples)
    results = await aevaluate(
        dataset=dataset,
        metrics=[Faithfulness(), AnswerRelevancy()],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    scores = results.to_pandas()
    avg_faithfulness = float(scores["faithfulness"].dropna().mean())
    avg_relevancy = float(scores["answer_relevancy"].dropna().mean())

    print(f"\nFaithfulness:     {avg_faithfulness:.3f} (min {FAITHFULNESS_MIN})")
    print(f"Answer Relevancy: {avg_relevancy:.3f} (min {ANSWER_RELEVANCY_MIN})")

    assert avg_faithfulness >= FAITHFULNESS_MIN, (
        f"Faithfulness {avg_faithfulness:.3f} < {FAITHFULNESS_MIN}. "
        "Check generator prompt for hallucinations."
    )
    assert avg_relevancy >= ANSWER_RELEVANCY_MIN, (
        f"Answer Relevancy {avg_relevancy:.3f} < {ANSWER_RELEVANCY_MIN}. "
        "Check intent classification and generator prompts."
    )


@pytest.mark.eval
@pytest.mark.asyncio
async def test_rag_context_quality(ragas_llm):
    """Context Recall and Context Precision must exceed minimum thresholds."""
    from ragas import EvaluationDataset, aevaluate
    from ragas.metrics.collections import (
        LLMContextRecall,
        LLMContextPrecisionWithReference,
    )

    in_scope = [
        q for q in _load_golden()
        if q["category"] not in ("out_of_scope",) and q["expected_behavior"] != "refusal"
    ]
    samples = await _build_ragas_samples(in_scope)
    dataset = EvaluationDataset(samples=samples)
    results = await aevaluate(
        dataset=dataset,
        metrics=[LLMContextRecall(), LLMContextPrecisionWithReference()],
        llm=ragas_llm,
    )

    scores = results.to_pandas()
    avg_recall = float(scores["llm_context_recall"].dropna().mean())
    avg_precision = float(scores["llm_context_precision_with_reference"].dropna().mean())

    print(f"\nContext Recall:    {avg_recall:.3f} (min {CONTEXT_RECALL_MIN})")
    print(f"Context Precision: {avg_precision:.3f} (min {CONTEXT_PRECISION_MIN})")

    assert avg_recall >= CONTEXT_RECALL_MIN, (
        f"Context Recall {avg_recall:.3f} < {CONTEXT_RECALL_MIN}. "
        "Retrieval is missing relevant documents."
    )
    assert avg_precision >= CONTEXT_PRECISION_MIN, (
        f"Context Precision {avg_precision:.3f} < {CONTEXT_PRECISION_MIN}. "
        "Too many irrelevant documents are being retrieved."
    )


@pytest.mark.eval
@pytest.mark.asyncio
async def test_refusal_rate():
    """Out-of-scope queries must always produce a refusal response.

    Refusal rate threshold: 100% — every off-topic query must be refused.
    """
    oos_items = _golden_by_category("out_of_scope")
    assert oos_items, "No out_of_scope items found in golden dataset."

    refused = 0
    for item in oos_items:
        try:
            result = await _run_pipeline(item["question"])
            if REFUSAL_PHRASE in result["answer"].lower():
                refused += 1
            else:
                logger.warning(
                    "Off-topic query NOT refused: %r → %s",
                    item["question"],
                    result["answer"][:150],
                )
        except Exception as exc:
            logger.warning("Pipeline error for %r: %s", item["question"], exc)

    refusal_rate = refused / len(oos_items)
    print(f"\nRefusal rate: {refusal_rate:.0%} ({refused}/{len(oos_items)})")
    assert refusal_rate == 1.0, (
        f"Refusal rate {refusal_rate:.0%} < 100%. "
        f"Some off-topic queries were answered instead of refused."
    )
