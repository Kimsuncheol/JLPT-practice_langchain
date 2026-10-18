"""Stateless AI services using langchain-ollama (ChatOllama) and local multiple-choice grading.

Privacy & self-study guarantees:
- No prompts, answers, generated questions, passages, explanations, scores, conversation
  content, or credentials are ever logged or stored.
- No database, checkpoints, files, or persistent memory are used.
- Ordinary multiple-choice grading is performed deterministically against the answer key
  without calling Ollama.
- Difficulty is labeled as approximate and all material as unofficial AI practice; practice
  accuracy never claims to predict an official JLPT score.
- Tutor instructions are strictly server-controlled; submitted question context and recent
  messages are treated purely as untrusted data in request-local scope.
- No Ollama client instantiation or network call occurs at import, startup, or health check time.
"""

import asyncio
import json
import re
from typing import Any
import httpx
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_ollama import ChatOllama
from pydantic import ValidationError

from app.config import (
    Settings,
    enforce_privacy_environment,
    get_settings,
    is_localhost_url,
)
from app.prompts import (
    PRACTICE_BATCH_GENERATE_PROMPT,
    PRACTICE_FOLLOW_UP_PROMPT,
    PRACTICE_QUESTION_PROMPT,
    TUTOR_ITEM_PROMPT,
)
from app.schemas import (
    JAPANESE_SCRIPT_PATTERN,
    MAX_READING_PASSAGE_CHARS,
    MAX_READING_QUESTIONS,
    MAX_TUTOR_HISTORY_MESSAGES,
    AlternativeExplanation,
    AnswerEvaluationRequest,
    AnswerEvaluationResponse,
    ChatMessage,
    FollowUpDifficulty,
    GeneratedPracticeQuestion,
    GenerationCategory,
    LLMGeneratedQuestionItem,
    LLMPracticeBatchOutput,
    PracticeFollowUpRequest,
    PracticeFollowUpResponse,
    PracticeGenerateRequest,
    PracticeGenerateResponse,
    PracticeQuestionRequest,
    PracticeQuestionResponse,
    SelectedPracticeQuestionContext,
    TutorChatRequest,
    TutorChatResponse,
    TutorGeneratedContent,
    TutorRequest,
    TutorResponse,
    compute_stable_question_id,
)

JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


class OllamaUnavailableError(RuntimeError):
    """Raised when the Ollama endpoint is unreachable, misconfigured, or the model is not pulled."""


# Alias for backward compatibility in exception handlers
MissingCredentialsError = OllamaUnavailableError


class UnsupportedStructuredOutputError(ValueError):
    """Raised when the selected Ollama model does not support structured output or valid JSON."""


class QuotaExceededError(RuntimeError):
    """Raised when a remote Ollama gateway returns a 429 rate-limit / quota error."""


class StructuralValidationError(ValueError):
    """Raised when AI output fails deterministic structural validation.

    Distinct from linguistic correctness: structural validation verifies schema
    conformance, batch count, Japanese script presence, reading passage bounds,
    supporting sentence presence, 4 distinct options, valid answer index (0-3),
    and 4 non-empty option explanations.
    """


def is_quota_or_rate_limit_error(exc: BaseException) -> bool:
    """Detect quota / 429 rate-limit errors without logging exception payloads."""
    if isinstance(exc, QuotaExceededError):
        return True
    cls_name = type(exc).__name__.lower()
    if any(
        token in cls_name
        for token in ("resourceexhausted", "quota", "ratelimit", "toomanyrequests")
    ):
        return True
    status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status_code == 429:
        return True
    msg = str(exc).lower()
    return any(
        token in msg
        for token in ("429", "resource_exhausted", "quota", "rate limit", "rate_limit")
    )


def is_timeout_error(exc: BaseException) -> bool:
    """Return True if the exception represents an async or HTTP timeout."""
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, httpx.TimeoutException)):
        return True
    cls_name = type(exc).__name__.lower()
    return "timeout" in cls_name or "timed out" in str(exc).lower()


def is_model_unavailable_error(exc: BaseException) -> bool:
    """Return True if the exception indicates Ollama is unreachable or the model is missing."""
    if isinstance(
        exc,
        (
            OllamaUnavailableError,
            httpx.ConnectError,
            httpx.NetworkError,
            ConnectionError,
            OSError,
        ),
    ):
        return True
    status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status_code in {404, 502, 503}:
        return True
    msg = str(exc).lower()
    return any(
        token in msg
        for token in (
            "connection refused",
            "failed to connect",
            "connecterror",
            "not found",
            "try pulling it first",
            "model",
            "unreachable",
            "503",
        )
    )


def is_unsupported_structured_output_error(exc: BaseException) -> bool:
    """Return True if the exception indicates the model does not support structured output."""
    if isinstance(exc, (UnsupportedStructuredOutputError, NotImplementedError)):
        return True
    msg = str(exc).lower()
    return any(
        token in msg
        for token in (
            "structured output",
            "with_structured_output",
            "does not support tools",
            "does not support json",
            "schema",
            "notimplemented",
        )
    )


def create_chat_model(settings: Settings | None = None) -> ChatOllama:
    """Instantiate ChatOllama lazily for a single request with bounded parameters.

    Never called during application startup, /health checks, or multiple-choice grading.
    """
    enforce_privacy_environment()
    cfg = settings or get_settings()
    base_url = cfg.ollama_base_url.strip()
    model_name = cfg.ollama_model.strip()

    if not base_url or not model_name:
        raise OllamaUnavailableError(
            "OLLAMA_BASE_URL and OLLAMA_MODEL must be configured on the backend."
        )

    if cfg.render_environment and is_localhost_url(base_url):
        raise OllamaUnavailableError(
            "Render Free cannot host an Ollama model or access localhost. "
            "Configure OLLAMA_BASE_URL to a separately hosted, reachable HTTPS Ollama endpoint."
        )

    client_kwargs: dict[str, Any] = {"timeout": cfg.ollama_timeout_seconds}
    api_key = cfg.ollama_api_key.get_secret_value().strip()
    if api_key:
        client_kwargs["headers"] = {"Authorization": f"Bearer {api_key}"}

    return ChatOllama(
        model=model_name,
        base_url=base_url,
        temperature=cfg.ollama_temperature,
        num_predict=cfg.ollama_num_predict,
        client_kwargs=client_kwargs,
    )


def get_chat_model() -> Any:
    """FastAPI dependency hook for injecting a chat model in tests.

    Returns None by default so service functions lazily instantiate ChatOllama on demand.
    """
    return None


def _build_request_local_history(history: list[ChatMessage]) -> list[BaseMessage]:
    """Convert browser-supplied messages into a bounded request-local LangChain message list.

    Strictly maps 'user' -> HumanMessage and 'assistant' -> AIMessage so user input
    can never inject SystemMessage instructions.
    """
    bounded_window = history[-MAX_TUTOR_HISTORY_MESSAGES:]
    local_messages: list[BaseMessage] = []
    for item in bounded_window:
        if item.role == "user":
            local_messages.append(HumanMessage(content=item.content))
        elif item.role == "assistant":
            local_messages.append(AIMessage(content=item.content))
        else:
            raise ValueError(f"Unsupported message role: {item.role}")
    return local_messages


def _format_selected_question_data(
    question: SelectedPracticeQuestionContext | None,
) -> str:
    """Serialize the single selected practice question into a bounded data block."""
    if question is None:
        return "No specific practice question selected."

    lines: list[str] = []
    if question.passage:
        lines.append(f"Reading Passage: {question.passage}")
    lines.append(f"Question Text: {question.question_text}")
    if question.options:
        formatted_opts = ", ".join(
            f"{idx + 1}. {opt}" for idx, opt in enumerate(question.options)
        )
        lines.append(f"Options: {formatted_opts}")
    if question.correct_answer:
        lines.append(f"Correct Answer: {question.correct_answer}")
    elif (
        question.correct_index is not None
        and 0 <= question.correct_index < len(question.options)
    ):
        lines.append(f"Correct Answer: {question.options[question.correct_index]}")
    if question.selected_answer:
        lines.append(f"Learner Selected Answer: {question.selected_answer}")
    if question.supporting_sentence:
        lines.append(f"Supporting Sentence: {question.supporting_sentence}")
    if question.option_explanations:
        exp_str = " | ".join(
            f"Option {idx + 1}: {exp}"
            for idx, exp in enumerate(question.option_explanations)
        )
        lines.append(f"Option Explanations: {exp_str}")
    return "\n".join(lines)


def _resolve_reading_supporting_sentence(
    passage: str, item: LLMGeneratedQuestionItem
) -> tuple[str, list[str]]:
    """Validate and ensure the supporting sentence from the passage is in each explanation."""
    candidate = (item.supporting_sentence or "").strip()

    if not candidate:
        passage_sentences = [
            s.strip()
            for s in re.split(r"[。！？\n]+", passage)
            if s.strip() and JAPANESE_SCRIPT_PATTERN.search(s)
        ]
        for sent in passage_sentences:
            if any(sent in exp for exp in item.option_explanations):
                candidate = sent
                break

    if not candidate or not JAPANESE_SCRIPT_PATTERN.search(candidate):
        raise StructuralValidationError(
            "Structural validation failed: reading question is missing a Japanese supporting_sentence."
        )

    normalized_candidate = candidate.rstrip("。！？ ")
    if normalized_candidate not in passage and candidate not in passage:
        raise StructuralValidationError(
            "Structural validation failed: supporting_sentence must come from the reading passage."
        )

    enriched_explanations: list[str] = []
    for exp in item.option_explanations:
        if candidate in exp or normalized_candidate in exp:
            if candidate not in exp:
                enriched_explanations.append(f"{exp} (Supporting sentence: 「{candidate}」)")
            else:
                enriched_explanations.append(exp)
        else:
            enriched_explanations.append(f"{exp} (Supporting sentence: 「{candidate}」)")

    return candidate, enriched_explanations


def _parse_llm_batch_output(raw_batch: Any) -> LLMPracticeBatchOutput:
    """Validate structured or JSON-string model output into LLMPracticeBatchOutput."""
    if raw_batch is None:
        raise UnsupportedStructuredOutputError(
            "Selected Ollama model returned empty structured output."
        )
    if isinstance(raw_batch, LLMPracticeBatchOutput):
        return raw_batch

    text_content: str | None = None
    if isinstance(raw_batch, AIMessage) and isinstance(raw_batch.content, str):
        text_content = raw_batch.content.strip()
    elif isinstance(raw_batch, str):
        text_content = raw_batch.strip()

    if text_content is not None:
        fence_match = JSON_FENCE_PATTERN.search(text_content)
        if fence_match:
            text_content = fence_match.group(1).strip()
        try:
            parsed_json = json.loads(text_content)
        except json.JSONDecodeError as exc:
            raise StructuralValidationError(
                "Model output is not valid JSON and failed structural validation."
            ) from exc
        return LLMPracticeBatchOutput.model_validate(parsed_json)

    return LLMPracticeBatchOutput.model_validate(raw_batch)


async def _invoke_batch_generation_chain(
    llm: Any,
    prompt_vars: dict[str, Any],
    timeout_seconds: float,
) -> Any:
    """Invoke structured output where supported by the Ollama model, with fallback handling."""
    if hasattr(llm, "with_structured_output"):
        try:
            structured_llm = llm.with_structured_output(LLMPracticeBatchOutput)
            chain = PRACTICE_BATCH_GENERATE_PROMPT | structured_llm
            result = await asyncio.wait_for(
                chain.ainvoke(prompt_vars),
                timeout=timeout_seconds,
            )
            if result is not None:
                return result
        except Exception as exc:
            if is_timeout_error(exc):
                raise asyncio.TimeoutError("Ollama generation timed out.") from None
            if is_quota_or_rate_limit_error(exc):
                raise QuotaExceededError(
                    "Ollama endpoint rate limit exceeded. Please wait and try again."
                ) from None
            if is_unsupported_structured_output_error(exc):
                # If the model also supports raw invocation, try raw JSON fallback; otherwise raise
                if not hasattr(llm, "ainvoke") and not callable(llm):
                    raise UnsupportedStructuredOutputError(
                        "The selected Ollama model does not support structured output."
                    ) from None
            elif is_model_unavailable_error(exc):
                raise OllamaUnavailableError(
                    "Ollama service or selected model is unavailable."
                ) from None
            else:
                raise

    # Fallback for models that do not implement or support native structured output
    try:
        fallback_chain = PRACTICE_BATCH_GENERATE_PROMPT | llm
        return await asyncio.wait_for(
            fallback_chain.ainvoke(prompt_vars),
            timeout=timeout_seconds,
        )
    except Exception as exc:
        if is_timeout_error(exc):
            raise asyncio.TimeoutError("Ollama generation timed out.") from None
        if is_quota_or_rate_limit_error(exc):
            raise QuotaExceededError(
                "Ollama endpoint rate limit exceeded. Please wait and try again."
            ) from None
        if is_unsupported_structured_output_error(exc):
            raise UnsupportedStructuredOutputError(
                "The selected Ollama model does not support structured output."
            ) from None
        if is_model_unavailable_error(exc):
            raise OllamaUnavailableError(
                "Ollama service or selected model is unavailable."
            ) from None
        raise


async def generate_practice_batch(
    payload: PracticeGenerateRequest,
    chat_model: Any = None,
    settings: Settings | None = None,
) -> PracticeGenerateResponse:
    """Generate JLPT vocabulary, grammar, or reading practice items in a single async LangChain call.

    Uses ChatOllama structured output where supported, enforces timeout and Pydantic
    structural validation, assigns deterministic stable IDs, and labels all material
    as unofficial AI practice with approximate difficulty.
    """
    enforce_privacy_environment()
    cfg = settings or get_settings()
    llm = chat_model if chat_model is not None else create_chat_model(cfg)

    prompt_vars = {
        "level": payload.level.value,
        "category": payload.category.value,
        "question_count": payload.question_count,
        "explanation_language": payload.explanation_language.value,
    }

    raw_batch = await _invoke_batch_generation_chain(
        llm=llm,
        prompt_vars=prompt_vars,
        timeout_seconds=cfg.ollama_timeout_seconds,
    )

    try:
        parsed_batch = _parse_llm_batch_output(raw_batch)
    except UnsupportedStructuredOutputError:
        raise
    except ValidationError as exc:
        raise StructuralValidationError(
            "Model output failed structural schema validation."
        ) from exc

    if len(parsed_batch.questions) != payload.question_count:
        raise StructuralValidationError(
            f"Structural validation failed: requested {payload.question_count} question(s), "
            f"but model returned {len(parsed_batch.questions)}."
        )

    is_reading = payload.category == GenerationCategory.READING
    passage_text: str | None = None

    if is_reading:
        if payload.question_count > MAX_READING_QUESTIONS:
            raise StructuralValidationError(
                f"Structural validation failed: reading category supports at most {MAX_READING_QUESTIONS} questions."
            )
        if not parsed_batch.passage or not JAPANESE_SCRIPT_PATTERN.search(
            parsed_batch.passage
        ):
            raise StructuralValidationError(
                "Structural validation failed: reading category requires a non-empty Japanese passage."
            )
        if len(parsed_batch.passage) > MAX_READING_PASSAGE_CHARS:
            raise StructuralValidationError(
                f"Structural validation failed: reading passage exceeds {MAX_READING_PASSAGE_CHARS} characters."
            )
        passage_text = parsed_batch.passage

    try:
        questions: list[GeneratedPracticeQuestion] = []
        for item in parsed_batch.questions:
            if not (0 <= item.correct_index < len(item.options)):
                raise StructuralValidationError(
                    "Structural validation failed: correct_index out of range."
                )

            supporting_sentence: str | None = None
            option_explanations = item.option_explanations

            if is_reading and passage_text is not None:
                supporting_sentence, option_explanations = (
                    _resolve_reading_supporting_sentence(passage_text, item)
                )

            stable_id = compute_stable_question_id(
                level=payload.level.value,
                category=payload.category.value,
                question_text=item.question_text,
                options=item.options,
            )
            questions.append(
                GeneratedPracticeQuestion(
                    id=stable_id,
                    question_text=item.question_text,
                    options=item.options,
                    correct_index=item.correct_index,
                    correct_answer=item.options[item.correct_index],
                    option_explanations=option_explanations,
                    supporting_sentence=supporting_sentence,
                )
            )

        return PracticeGenerateResponse(
            level=payload.level,
            category=payload.category,
            question_count=payload.question_count,
            explanation_language=payload.explanation_language,
            passage=passage_text,
            questions=questions,
        )
    except ValidationError as exc:
        raise StructuralValidationError(
            "Generated questions failed structural validation."
        ) from exc


def _normalize_question_text_for_comparison(text: str) -> str:
    """Normalize Japanese question text to detect duplicate/repeated follow-up questions."""
    return re.sub(r"[\s\u3000]+", "", text.strip())


def _parse_llm_follow_up_output(raw_output: Any) -> LLMGeneratedQuestionItem:
    """Validate structured or JSON-string model output into LLMGeneratedQuestionItem."""
    if raw_output is None:
        raise UnsupportedStructuredOutputError(
            "Selected Ollama model returned empty structured output."
        )
    if isinstance(raw_output, LLMGeneratedQuestionItem):
        return raw_output
    if isinstance(raw_output, LLMPracticeBatchOutput):
        if not raw_output.questions:
            raise StructuralValidationError(
                "Model returned an empty question list for follow-up generation."
            )
        return raw_output.questions[0]

    text_content: str | None = None
    if isinstance(raw_output, AIMessage) and isinstance(raw_output.content, str):
        text_content = raw_output.content.strip()
    elif isinstance(raw_output, str):
        text_content = raw_output.strip()

    if text_content is not None:
        fence_match = JSON_FENCE_PATTERN.search(text_content)
        if fence_match:
            text_content = fence_match.group(1).strip()
        try:
            parsed_json = json.loads(text_content)
        except json.JSONDecodeError as exc:
            raise StructuralValidationError(
                "Model output is not valid JSON and failed structural validation."
            ) from exc
        if (
            isinstance(parsed_json, dict)
            and "questions" in parsed_json
            and isinstance(parsed_json["questions"], list)
            and parsed_json["questions"]
        ):
            return LLMGeneratedQuestionItem.model_validate(parsed_json["questions"][0])
        return LLMGeneratedQuestionItem.model_validate(parsed_json)

    if (
        isinstance(raw_output, dict)
        and "questions" in raw_output
        and isinstance(raw_output["questions"], list)
        and raw_output["questions"]
    ):
        return LLMGeneratedQuestionItem.model_validate(raw_output["questions"][0])

    return LLMGeneratedQuestionItem.model_validate(raw_output)


async def _invoke_follow_up_generation_chain(
    llm: Any,
    prompt_vars: dict[str, Any],
    timeout_seconds: float,
) -> Any:
    """Invoke structured output for a single follow-up question where supported, with JSON fallback."""
    if hasattr(llm, "with_structured_output"):
        try:
            structured_llm = llm.with_structured_output(LLMGeneratedQuestionItem)
            chain = PRACTICE_FOLLOW_UP_PROMPT | structured_llm
            result = await asyncio.wait_for(
                chain.ainvoke(prompt_vars),
                timeout=timeout_seconds,
            )
            if result is not None:
                return result
        except Exception as exc:
            if is_timeout_error(exc):
                raise asyncio.TimeoutError("Ollama follow-up generation timed out.") from None
            if is_quota_or_rate_limit_error(exc):
                raise QuotaExceededError(
                    "Ollama endpoint rate limit exceeded. Please wait and try again."
                ) from None
            if is_unsupported_structured_output_error(exc):
                if not hasattr(llm, "ainvoke") and not callable(llm):
                    raise UnsupportedStructuredOutputError(
                        "The selected Ollama model does not support structured output."
                    ) from None
            elif is_model_unavailable_error(exc):
                raise OllamaUnavailableError(
                    "Ollama service or selected model is unavailable."
                ) from None
            else:
                raise

    try:
        fallback_chain = PRACTICE_FOLLOW_UP_PROMPT | llm
        return await asyncio.wait_for(
            fallback_chain.ainvoke(prompt_vars),
            timeout=timeout_seconds,
        )
    except Exception as exc:
        if is_timeout_error(exc):
            raise asyncio.TimeoutError("Ollama follow-up generation timed out.") from None
        if is_quota_or_rate_limit_error(exc):
            raise QuotaExceededError(
                "Ollama endpoint rate limit exceeded. Please wait and try again."
            ) from None
        if is_unsupported_structured_output_error(exc):
            raise UnsupportedStructuredOutputError(
                "The selected Ollama model does not support structured output."
            ) from None
        if is_model_unavailable_error(exc):
            raise OllamaUnavailableError(
                "Ollama service or selected model is unavailable."
            ) from None
        raise


async def generate_follow_up_question(
    payload: PracticeFollowUpRequest,
    chat_model: Any = None,
    settings: Settings | None = None,
) -> PracticeFollowUpResponse:
    """Generate one original follow-up question targeting the same learning point.

    Validates output with Pydantic, ensures 4 distinct options, 1 correct answer, and
    4 explanations, and rejects outputs that repeat the previous question.
    Never logs or persists questions, answers, or conversation content.
    """
    enforce_privacy_environment()
    cfg = settings or get_settings()
    llm = chat_model if chat_model is not None else create_chat_model(cfg)

    current_item = (
        payload.current_item
        if isinstance(payload.current_item, SelectedPracticeQuestionContext)
        else SelectedPracticeQuestionContext(question_text=str(payload.current_item))
    )
    selected_answer = (payload.selected_answer or current_item.selected_answer or "").strip()

    if payload.difficulty == FollowUpDifficulty.HARDER:
        difficulty_instruction = (
            f"Create a MORE CHALLENGING question than the previous item (upper-band JLPT {payload.level.value} "
            "or subtler distractor contrast) while testing the exact same core grammar/vocabulary learning point."
        )
    else:
        difficulty_instruction = (
            f"Keep the SAME approximate JLPT {payload.level.value} difficulty as the previous item "
            "to reinforce the exact same core learning point in a new context."
        )

    prompt_vars = {
        "level": payload.level.value,
        "category": payload.category.value,
        "difficulty": payload.difficulty.value,
        "difficulty_instruction": difficulty_instruction,
        "explanation_language": payload.explanation_language.value,
        "practice_item_context": _format_selected_question_data(current_item),
        "selected_answer": selected_answer,
    }

    raw_output = await _invoke_follow_up_generation_chain(
        llm=llm,
        prompt_vars=prompt_vars,
        timeout_seconds=cfg.ollama_timeout_seconds,
    )

    try:
        parsed_item = _parse_llm_follow_up_output(raw_output)
    except UnsupportedStructuredOutputError:
        raise
    except ValidationError as exc:
        raise StructuralValidationError(
            "Follow-up model output failed structural schema validation."
        ) from exc

    if _normalize_question_text_for_comparison(
        parsed_item.question_text
    ) == _normalize_question_text_for_comparison(current_item.question_text):
        raise StructuralValidationError(
            "Structural validation failed: follow-up question repeated the previous question."
        )

    if not (0 <= parsed_item.correct_index < len(parsed_item.options)):
        raise StructuralValidationError(
            "Structural validation failed: correct_index out of range."
        )

    try:
        stable_id = compute_stable_question_id(
            level=payload.level.value,
            category=payload.category.value,
            question_text=parsed_item.question_text,
            options=parsed_item.options,
        )
        question_obj = GeneratedPracticeQuestion(
            id=stable_id,
            question_text=parsed_item.question_text,
            options=parsed_item.options,
            correct_index=parsed_item.correct_index,
            correct_answer=parsed_item.options[parsed_item.correct_index],
            option_explanations=parsed_item.option_explanations,
            supporting_sentence=parsed_item.supporting_sentence,
        )
        return PracticeFollowUpResponse(
            id=question_obj.id,
            level=payload.level,
            category=payload.category,
            difficulty=payload.difficulty,
            explanation_language=payload.explanation_language,
            previous_question_id=current_item.id,
            passage=current_item.passage,
            question_text=question_obj.question_text,
            options=question_obj.options,
            correct_index=question_obj.correct_index,
            correct_answer=question_obj.correct_answer,
            option_explanations=question_obj.option_explanations,
            supporting_sentence=question_obj.supporting_sentence,
            question=question_obj,
        )
    except ValidationError as exc:
        raise StructuralValidationError(
            "Generated follow-up question failed structural validation."
        ) from exc


async def generate_practice_question(
    payload: PracticeQuestionRequest,
    settings: Settings | None = None,
) -> PracticeQuestionResponse:
    """Generate a single structured JLPT practice question without logging or persistence."""
    llm = create_chat_model(settings)
    structured_llm = llm.with_structured_output(PracticeQuestionResponse)
    chain = PRACTICE_QUESTION_PROMPT | structured_llm

    result = await chain.ainvoke(
        {
            "level": payload.level.value,
            "category": payload.category.value,
            "topic": payload.topic or "general",
        }
    )
    if isinstance(result, PracticeQuestionResponse):
        return result
    return PracticeQuestionResponse.model_validate(result)


async def evaluate_practice_answer(
    payload: AnswerEvaluationRequest,
    settings: Settings | None = None,
) -> AnswerEvaluationResponse:
    """Compare a learner's selected multiple-choice answer locally against the answer key.

    Does NOT call Ollama or persist any answers, explanations, or scores.
    """
    _ = settings
    enforce_privacy_environment()

    selected = payload.selected_answer.strip()
    options = payload.options or []
    explanations = payload.option_explanations or []

    resolved_correct_index = payload.correct_index
    if (
        resolved_correct_index is not None
        and 0 <= resolved_correct_index < len(options)
    ):
        resolved_correct_answer = options[resolved_correct_index].strip()
    else:
        resolved_correct_answer = (payload.correct_answer or "").strip()
        if resolved_correct_answer and resolved_correct_answer in options:
            resolved_correct_index = options.index(resolved_correct_answer)

    is_correct = bool(
        resolved_correct_answer and selected == resolved_correct_answer
    )

    why_correct_fits = ""
    alternatives: list[AlternativeExplanation] = []

    if (
        len(options) == 4
        and len(explanations) == 4
        and resolved_correct_index is not None
        and 0 <= resolved_correct_index < 4
    ):
        why_correct_fits = explanations[resolved_correct_index]
        for idx, opt_text in enumerate(options):
            if idx == resolved_correct_index:
                continue
            alternatives.append(
                AlternativeExplanation(
                    option_index=idx,
                    option_text=opt_text,
                    was_selected=(opt_text.strip() == selected),
                    why_it_does_not_fit=explanations[idx],
                )
            )
    elif resolved_correct_answer:
        why_correct_fits = f"The correct option is '{resolved_correct_answer}'."

    if is_correct:
        feedback = f"Correct! '{resolved_correct_answer}' is the right choice. {why_correct_fits}".strip()
    elif resolved_correct_answer:
        feedback = (
            f"Not quite. You selected '{selected}', but the correct answer is "
            f"'{resolved_correct_answer}'. {why_correct_fits}".strip()
        )
    else:
        feedback = "No answer key was provided for local comparison."

    return AnswerEvaluationResponse(
        is_correct=is_correct,
        selected_answer=selected,
        correct_answer=resolved_correct_answer,
        supporting_sentence=payload.supporting_sentence,
        why_correct_fits=why_correct_fits,
        why_alternatives_do_not_fit=alternatives,
        feedback=feedback,
        grammar_notes=[],
    )


async def answer_practice_item_tutor_question(
    payload: TutorRequest,
    chat_model: Any = None,
    settings: Settings | None = None,
) -> TutorResponse:
    """Answer a learner's question about the selected practice item in one stateless call.

    Uses server-controlled system instructions, treats all submitted question/history/message
    fields strictly as data in request-local scope, and never persists or logs content.
    """
    enforce_privacy_environment()
    cfg = settings or get_settings()
    llm = chat_model if chat_model is not None else create_chat_model(cfg)
    chain = TUTOR_ITEM_PROMPT | llm

    question_ctx = (
        payload.question
        if isinstance(payload.question, SelectedPracticeQuestionContext)
        else SelectedPracticeQuestionContext(question_text=str(payload.question))
    )
    request_local_history = _build_request_local_history(payload.history)
    practice_item_context = _format_selected_question_data(question_ctx)

    try:
        ai_output = await asyncio.wait_for(
            chain.ainvoke(
                {
                    "level": payload.level.value,
                    "explanation_language": payload.explanation_language.value,
                    "practice_item_context": practice_item_context,
                    "history": request_local_history,
                    "message": payload.message,
                }
            ),
            timeout=cfg.ollama_timeout_seconds,
        )
    except (asyncio.TimeoutError, OllamaUnavailableError, StructuralValidationError):
        raise
    except Exception as exc:
        if is_timeout_error(exc):
            raise asyncio.TimeoutError("Ollama tutor request timed out.") from None
        if is_quota_or_rate_limit_error(exc):
            raise QuotaExceededError(
                "Ollama endpoint rate limit exceeded. Please wait and try again."
            ) from None
        if is_model_unavailable_error(exc):
            raise OllamaUnavailableError(
                "Ollama service or selected model is unavailable."
            ) from None
        raise

    # No follow-up model call or JSON-repair call: both parts must arrive
    # atomically from the single invocation above.
    try:
        if isinstance(ai_output, TutorGeneratedContent):
            generated = ai_output
        elif isinstance(ai_output, dict):
            generated = TutorGeneratedContent.model_validate(ai_output)
        else:
            raw = ai_output.content if isinstance(ai_output, AIMessage) else ai_output
            if not isinstance(raw, str):
                raise ValueError("Unexpected tutor output type")
            text = raw.strip()
            fence = JSON_FENCE_PATTERN.fullmatch(text)
            if fence:
                text = fence.group(1)
            generated = TutorGeneratedContent.model_validate(json.loads(text))
    except (ValidationError, ValueError, TypeError):
        raise StructuralValidationError(
            "Tutor model must return an answer and three distinct follow-up questions together."
        ) from None

    return TutorResponse(
        level=payload.level,
        explanation_language=payload.explanation_language,
        question_id=question_ctx.id,
        reply=generated.reply,
        follow_up_questions=generated.follow_up_questions,
    )


async def run_tutor_chat(
    payload: TutorChatRequest,
    chat_model: Any = None,
    settings: Settings | None = None,
) -> TutorChatResponse:
    """Run a stateless tutor turn using only request-local conversation context."""
    question_ctx = payload.question or SelectedPracticeQuestionContext(
        question_text="General JLPT study question"
    )
    tutor_req = TutorRequest(
        level=payload.level,
        explanation_language=payload.explanation_language,
        question=question_ctx,
        message=payload.message,
        history=payload.history,
    )
    result = await answer_practice_item_tutor_question(
        payload=tutor_req,
        chat_model=chat_model,
        settings=settings,
    )
    return TutorChatResponse(
        level=result.level,
        explanation_language=result.explanation_language,
        reply=result.reply,
        follow_up_questions=result.follow_up_questions,
    )
