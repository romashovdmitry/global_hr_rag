"""LangGraph state definition for the RAG pipeline."""

from typing import Any, TypedDict


class RAGState(TypedDict):
    """Shared mutable state threaded through all LangGraph nodes.

    Attributes:
        session_id: Identifier of the originating chat session.
        user_query: Raw message text submitted by the user.
        cv_text: Extracted plain text of the uploaded CV, or None.
        chat_history: List of previous messages as dicts with role/content keys.
        intent: Classified intent of the user query. One of:
            ``"find_matching"``   — find vacancies matching the user's profile.
            ``"role_transition"`` — user wants to switch to a different role.
            ``"compare"``         — compare specific vacancies or choose the best.
            ``"clarify_vacancy"`` — user asks a follow-up about a specific vacancy.
            ``"followup"``        — general refinement of a previous answer.
            ``"aggregate"``       — multi-hop synthesis / aggregation across many vacancies.
            ``"off_topic"``       — query outside the job-search domain.
        current_role: The role the user is currently in (populated for
            ``"role_transition"`` intent so downstream nodes can exclude it).
        exclude_skills: Technologies or tools to exclude from results, extracted by
            the metadata extractor from negative constraints like "without Django".
        metadata_filters: Structured hard-filter conditions parsed by the
            metadata extractor node (e.g. ``{"status_of_role": ["Remote"]}``)
        search_queries: Three diversified search strings produced by the
            query-translation node.
        retrieved_vacancies: Raw vacancy payloads returned from Qdrant.
        graded_vacancies: Subset of retrieved vacancies that passed the
            context-relevance grader.
        generated_answer: Draft answer produced by the generator node.
        is_grounded: Whether the groundedness grader approved the draft.
        grounding_feedback: Reason the previous draft failed groundedness, if any.
            The generator uses this on retry to avoid repeating unsupported claims.
        rejected_answer: The last draft discarded by the groundedness grader.
        answer_relevant: Whether the final answer addresses the user query.
        retry_count: Number of re-route iterations performed so far.
        status_stream: List of status update strings emitted during processing,
            used by the WebSocket handler to push intermediate progress.
        node_timings: Wall-clock time spent inside each node in milliseconds.
            Populated by the ``@timed_node`` decorator on each node function.
        input_tokens: Input (prompt) token count from the generator LLM call.
        output_tokens: Output (completion) token count from the generator LLM call.
        unsupported_criteria: Criteria mentioned by the user that have no corresponding
            field in our vacancy data (e.g. location, country, benefits).
            When non-empty the generator informs the user about the limitation.
    """

    session_id: str
    user_query: str
    cv_text: str | None
    chat_history: list[dict[str, Any]]
    intent: str
    current_role: str | None
    exclude_skills: list[str]
    metadata_filters: dict[str, Any]
    unsupported_criteria: list[str]
    search_queries: list[str]
    retrieved_vacancies: list[dict[str, Any]]
    graded_vacancies: list[dict[str, Any]]
    generated_answer: str
    is_grounded: bool
    grounding_feedback: str | None
    rejected_answer: str | None
    answer_relevant: bool
    retry_count: int
    status_stream: list[str]
    node_timings: dict[str, int]
    input_tokens: int | None
    output_tokens: int | None
