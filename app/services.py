"""Stateless AI services using langchain-google-genai and local multiple-choice grading.

Privacy & self-study guarantees:
- No prompts, answers, generated questions, passages, explanations, scores, conversation
  content, or API keys are ever logged or stored.
- No database, checkpoints, files, or persistent memory are used.
- Ordinary multiple-choice grading is performed deterministically against the answer key
  without calling Gemini.
- Difficulty is labeled as approximate and all material as unofficial AI practice; practice
  accuracy never claims to predict an official JLPT score.
- Tutor instructions are strictly server-controlled; submitted question context and recent
  messages are treated purely as untrusted data in request-local scope.
- No Gemini client instantiation or network call occurs at import, startup, or health check time.
"""

import asyncio
import re
from typing import Any
from fastapi import Depends
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import ValidationError

from app.config import Settings, enforce_privacy_environment, get_settings
from app.prompts import (
    PRACTICE_BATCH_GENERATE_PROMPT,
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
    GeneratedPracticeQuestion,
    GenerationCategory,
    LLMGeneratedQuestionItem,
    LLMPracticeBatchOutput,
    PracticeGenerateRequest,
    PracticeGenerateResponse,
    PracticeQuestionRequest,
    PracticeQuestionResponse,
    SelectedPracticeQuestionContext,
    TutorChatRequest,
    TutorChatResponse,
    TutorRequest,
    TutorResponse,
    compute_stable_question_id,
)


class MissingCredentialsError(RuntimeError):
    """Raised when GEMINI_API_KEY is not configured on the backend."""


class StructuralValidationError(ValueError):
    """Raised when AI output fails deterministic structural validation.

    Distinct from linguistic correctness: structural validation verifies schema
    conformance, batch count, Japanese script presence, reading passage bounds,
    supporting sentence presence, 4 distinct options, valid answer index (0-3),
    and 4 non-empty option explanations.
    """


def create_chat_model(settings: Settings | None = None) -> ChatGoogleGenerativeAI:
    """Instantiate ChatGoogleGenerativeAI lazily for a single request with bounded parameters.

    Never called during application startup, /health checks, or multiple-choice grading.
    """
    enforce_privacy_environment()
    cfg = settings or get_settings()
    api_key = cfg.gemini_api_key.get_secret_value().strip()
    if not api_key or api_key == "your_gemini_api_key_here":
        raise MissingCredentialsError(
            "GEMINI_API_KEY is not configured on the backend."
        )

    return ChatGoogleGenerativeAI(
        model=cfg.gemini_model,
        google_api_key=api_key,
        temperature=cfg.gemini_temperature,
        max_output_tokens=cfg.gemini_max_output_tokens,
        timeout=cfg.gemini_timeout_seconds,
        max_retries=cfg.gemini_max_retries,
    )


def get_chat_model() -> Any:
    """FastAPI dependency hook for injecting a chat model in tests.

    Returns None by default so service functions lazily instantiate
    ChatGoogleGenerativeAI on demand.
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
        # Try to find a passage sentence already quoted inside option_explanations
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


async def generate_practice_batch(
    payload: PracticeGenerateRequest,
    chat_model: Any = None,
    settings: Settings | None = None,
) -> PracticeGenerateResponse:
    """Generate JLPT vocabulary, grammar, or reading practice items in a single async LangChain call.

    Uses Pydantic structured output, enforces timeout and structural validation,
    assigns deterministic stable IDs, and labels all material as unofficial AI practice
    with approximate difficulty. Never logs or persists requests or generated questions.
    """
    enforce_privacy_environment()
    cfg = settings or get_settings()
    llm = chat_model if chat_model is not None else create_chat_model(cfg)

    if hasattr(llm, "with_structured_output"):
        structured_llm = llm.with_structured_output(LLMPracticeBatchOutput)
    else:
        structured_llm = llm

    chain = PRACTICE_BATCH_GENERATE_PROMPT | structured_llm

    # Single async generation call per batch, bounded by configured timeout
    raw_batch = await asyncio.wait_for(
        chain.ainvoke(
            {
                "level": payload.level.value,
                "category": payload.category.value,
                "question_count": payload.question_count,
                "explanation_language": payload.explanation_language.value,
            }
        ),
        timeout=cfg.gemini_timeout_seconds,
    )

    try:
        if isinstance(raw_batch, LLMPracticeBatchOutput):
            parsed_batch = raw_batch
        elif isinstance(raw_batch, AIMessage) and isinstance(raw_batch.content, str):
            parsed_batch = LLMPracticeBatchOutput.model_validate_json(raw_batch.content)
        elif isinstance(raw_batch, str):
            parsed_batch = LLMPracticeBatchOutput.model_validate_json(raw_batch)
        else:
            parsed_batch = LLMPracticeBatchOutput.model_validate(raw_batch)
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

    Does NOT call Gemini or persist any answers, explanations, or scores.
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
        timeout=cfg.gemini_timeout_seconds,
    )

    if isinstance(ai_output, AIMessage):
        reply_text = (
            ai_output.content
            if isinstance(ai_output.content, str)
            else str(ai_output.content)
        )
    elif isinstance(ai_output, str):
        reply_text = ai_output
    elif isinstance(ai_output, dict) and "reply" in ai_output:
        reply_text = str(ai_output["reply"])
    else:
        reply_text = str(getattr(ai_output, "content", ai_output))

    reply_text = reply_text.strip()
    if not reply_text:
        raise StructuralValidationError("Tutor model returned an empty response.")

    return TutorResponse(
        level=payload.level,
        explanation_language=payload.explanation_language,
        question_id=question_ctx.id,
        reply=reply_text,
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
    )
