"""Stateless AI services using langchain-google-genai.

Privacy guarantees:
- No prompts, answers, generated questions, conversation content, or API keys are ever logged or stored.
- No database, files, or persistent memory are used.
- Conversation context and generated batches exist strictly in request-local scope during execution.
- No Gemini client instantiation or network call occurs at import, startup, or health check time.
"""

import asyncio
from typing import Any
from fastapi import Depends
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import ValidationError

from app.config import Settings, enforce_privacy_environment, get_settings
from app.prompts import (
    ANSWER_EVALUATION_PROMPT,
    PRACTICE_BATCH_GENERATE_PROMPT,
    PRACTICE_QUESTION_PROMPT,
    TUTOR_CHAT_PROMPT,
)
from app.schemas import (
    AnswerEvaluationRequest,
    AnswerEvaluationResponse,
    ChatMessage,
    GeneratedPracticeQuestion,
    LLMPracticeBatchOutput,
    PracticeGenerateRequest,
    PracticeGenerateResponse,
    PracticeQuestionRequest,
    PracticeQuestionResponse,
    TutorChatRequest,
    TutorChatResponse,
    compute_stable_question_id,
)


class MissingCredentialsError(RuntimeError):
    """Raised when GEMINI_API_KEY is not configured on the backend."""


class StructuralValidationError(ValueError):
    """Raised when AI output fails deterministic structural validation.

    Distinct from linguistic correctness: structural validation verifies schema
    conformance, batch count, Japanese script presence, 4 distinct options,
    valid answer index (0-3), and 4 non-empty option explanations.
    """


def create_chat_model(settings: Settings | None = None) -> ChatGoogleGenerativeAI:
    """Instantiate ChatGoogleGenerativeAI lazily for a single request with bounded parameters.

    Never called during application startup or /health checks.
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


def get_chat_model(settings: Settings = Depends(get_settings)) -> Any:
    """FastAPI dependency providing the chat model (overridable in tests with a fake model)."""
    return create_chat_model(settings)


def _build_request_local_history(history: list[ChatMessage]) -> list[BaseMessage]:
    """Convert browser-supplied messages into a request-local LangChain message list."""
    local_messages: list[BaseMessage] = []
    for item in history:
        if item.role == "user":
            local_messages.append(HumanMessage(content=item.content))
        else:
            local_messages.append(AIMessage(content=item.content))
    return local_messages


async def generate_practice_batch(
    payload: PracticeGenerateRequest,
    chat_model: Any = None,
    settings: Settings | None = None,
) -> PracticeGenerateResponse:
    """Generate 1 to 5 JLPT vocabulary/grammar practice questions in a single async LangChain call.

    Uses Pydantic structured output, enforces timeout and structural validation,
    assigns deterministic stable IDs, and labels all material as AI-generated.
    Never logs or persists requests or generated questions.
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

    try:
        questions: list[GeneratedPracticeQuestion] = []
        for item in parsed_batch.questions:
            if not (0 <= item.correct_index < len(item.options)):
                raise StructuralValidationError(
                    "Structural validation failed: correct_index out of range."
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
                    option_explanations=item.option_explanations,
                )
            )

        return PracticeGenerateResponse(
            level=payload.level,
            category=payload.category,
            question_count=payload.question_count,
            explanation_language=payload.explanation_language,
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
    """Evaluate a learner's answer without logging or persistence."""
    llm = create_chat_model(settings)
    structured_llm = llm.with_structured_output(AnswerEvaluationResponse)
    chain = ANSWER_EVALUATION_PROMPT | structured_llm

    result = await chain.ainvoke(
        {
            "level": payload.level.value,
            "question": payload.question,
            "selected_answer": payload.selected_answer,
            "correct_answer": payload.correct_answer or "Not specified",
        }
    )
    if isinstance(result, AnswerEvaluationResponse):
        return result
    return AnswerEvaluationResponse.model_validate(result)


async def run_tutor_chat(
    payload: TutorChatRequest,
    settings: Settings | None = None,
) -> TutorChatResponse:
    """Run a stateless tutor turn using only request-local conversation context."""
    llm = create_chat_model(settings)
    chain = TUTOR_CHAT_PROMPT | llm

    # Strictly request-local conversation list; discarded immediately after response
    request_local_history = _build_request_local_history(payload.history)

    ai_message = await chain.ainvoke(
        {
            "level": payload.level.value,
            "history": request_local_history,
            "message": payload.message,
        }
    )

    content = (
        ai_message.content
        if isinstance(ai_message.content, str)
        else str(ai_message.content)
    )
    return TutorChatResponse(level=payload.level, reply=content)
