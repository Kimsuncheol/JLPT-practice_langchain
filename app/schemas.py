"""Pydantic schemas for JLPT practice, answer evaluation, tutor chat, and health checks."""

from enum import Enum
from typing import Literal, Optional
from pydantic import BaseModel, Field


class JLPTLevel(str, Enum):
    """Supported JLPT proficiency levels."""

    N5 = "N5"
    N4 = "N4"
    N3 = "N3"
    N2 = "N2"
    N1 = "N1"


class PracticeCategory(str, Enum):
    """Supported JLPT practice categories."""

    VOCABULARY = "vocabulary"
    GRAMMAR = "grammar"
    KANJI = "kanji"
    READING = "reading"


class ChatMessage(BaseModel):
    """Single message in an ephemeral, browser-maintained conversation history."""

    role: Literal["user", "assistant"] = Field(
        ..., description="Speaker role in the conversation."
    )
    content: str = Field(
        ..., min_length=1, max_length=2000, description="Message text content."
    )


class PracticeQuestionRequest(BaseModel):
    """Request payload for generating a JLPT practice question."""

    level: JLPTLevel = Field(default=JLPTLevel.N5, description="Target JLPT level.")
    category: PracticeCategory = Field(
        default=PracticeCategory.GRAMMAR, description="Question skill category."
    )
    topic: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Optional focus topic (e.g., particles, te-form, daily life).",
    )


class PracticeQuestionResponse(BaseModel):
    """Structured JLPT practice question returned by the AI service."""

    level: JLPTLevel
    category: PracticeCategory
    question: str = Field(..., description="The JLPT multiple-choice question prompt.")
    options: list[str] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Four multiple-choice answer options.",
    )
    correct_index: int = Field(
        ..., ge=0, le=3, description="Zero-based index of the correct option (0-3)."
    )
    explanation: str = Field(
        ..., description="Explanation of why the correct option is right."
    )


class AnswerEvaluationRequest(BaseModel):
    """Request payload for evaluating a learner's answer to a JLPT question."""

    level: JLPTLevel = Field(default=JLPTLevel.N5)
    question: str = Field(..., min_length=1, max_length=2000)
    selected_answer: str = Field(..., min_length=1, max_length=500)
    correct_answer: Optional[str] = Field(default=None, max_length=500)


class AnswerEvaluationResponse(BaseModel):
    """Structured feedback for a learner's submitted answer."""

    is_correct: bool
    feedback: str
    grammar_notes: list[str] = Field(default_factory=list)


class TutorChatRequest(BaseModel):
    """Stateless tutor request carrying ephemeral history from browser memory."""

    level: JLPTLevel = Field(default=JLPTLevel.N5, description="Learner's JLPT level.")
    message: str = Field(
        ..., min_length=1, max_length=2000, description="Current learner question."
    )
    history: list[ChatMessage] = Field(
        default_factory=list,
        max_length=20,
        description="Ephemeral conversation context held only in browser and request-local memory.",
    )


class TutorChatResponse(BaseModel):
    """Stateless tutor reply."""

    level: JLPTLevel
    reply: str


class HealthResponse(BaseModel):
    """Health check response verifying service readiness without calling Gemini."""

    status: Literal["ok"] = "ok"
    model: str
    tracing_disabled: bool
    persistence: Literal["none"] = "none"
