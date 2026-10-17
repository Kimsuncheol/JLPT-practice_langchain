"""Stateless AI services using langchain-google-genai.

Privacy guarantees:
- No prompts, answers, conversation content, or API keys are ever logged.
- No database, files, or persistent memory are used.
- Conversation context exists strictly in request-local scope during execution.
- No Gemini client instantiation or network call occurs at import, startup, or health check time.
"""

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import Settings, enforce_privacy_environment, get_settings
from app.prompts import (
    ANSWER_EVALUATION_PROMPT,
    PRACTICE_QUESTION_PROMPT,
    TUTOR_CHAT_PROMPT,
)
from app.schemas import (
    AnswerEvaluationRequest,
    AnswerEvaluationResponse,
    ChatMessage,
    PracticeQuestionRequest,
    PracticeQuestionResponse,
    TutorChatRequest,
    TutorChatResponse,
)


class MissingCredentialsError(RuntimeError):
    """Raised when GEMINI_API_KEY is not configured on the backend."""


def create_chat_model(settings: Settings | None = None) -> ChatGoogleGenerativeAI:
    """Instantiate ChatGoogleGenerativeAI lazily for a single request.

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
    )


def _build_request_local_history(history: list[ChatMessage]) -> list[BaseMessage]:
    """Convert browser-supplied messages into a request-local LangChain message list."""
    local_messages: list[BaseMessage] = []
    for item in history:
        if item.role == "user":
            local_messages.append(HumanMessage(content=item.content))
        else:
            local_messages.append(AIMessage(content=item.content))
    return local_messages


async def generate_practice_question(
    payload: PracticeQuestionRequest,
    settings: Settings | None = None,
) -> PracticeQuestionResponse:
    """Generate a structured JLPT practice question without logging or persistence."""
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
