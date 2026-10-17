"""Pydantic schemas for JLPT practice generation, local quiz feedback, tutor chat, and health checks.

Distinguishes:
1. Structural validation (enforced deterministically by Pydantic validators) vs.
   linguistic correctness (not guaranteed for AI-generated content).
2. Self-study local grading: answer keys are held in browser memory (inspectable by the
   learner) and compared locally without calling Gemini or persisting scores/answers.
   Scoring is explicitly for self-study and is never described as tamper-proof.
"""

import hashlib
import re
from enum import Enum
from typing import Literal, Optional
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# Regex matching at least one Hiragana, Katakana, or CJK Unified Ideograph (Kanji) character
JAPANESE_SCRIPT_PATTERN = re.compile(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]")

STRUCTURAL_VS_LINGUISTIC_DISCLAIMER = (
    "Structural validation passed (schema, Japanese script presence, 4 distinct options, "
    "single valid answer index, and per-option explanations verified). "
    "Note: Structural validation does not guarantee linguistic nuance or pedagogical accuracy "
    "of AI-generated content."
)

SELF_STUDY_GRADING_NOTICE = (
    "Self-study mode: The answer key is held in browser memory and compared locally without "
    "calling Gemini or storing scores. Because a browser-held answer key is inspectable, "
    "this scoring is for self-study feedback only and is not tamper-proof."
)


class JLPTLevel(str, Enum):
    """Supported JLPT proficiency levels (N5 to N1)."""

    N5 = "N5"
    N4 = "N4"
    N3 = "N3"
    N2 = "N2"
    N1 = "N1"


class PracticeCategory(str, Enum):
    """Supported JLPT practice categories across the app."""

    VOCABULARY = "vocabulary"
    GRAMMAR = "grammar"
    KANJI = "kanji"
    READING = "reading"


class GenerationCategory(str, Enum):
    """Supported categories for POST /api/practice/generate (vocabulary and grammar)."""

    VOCABULARY = "vocabulary"
    GRAMMAR = "grammar"


class ExplanationLanguage(str, Enum):
    """Supported languages for option explanations."""

    ENGLISH = "English"
    KOREAN = "Korean"
    JAPANESE = "Japanese"


def compute_stable_question_id(
    level: str,
    category: str,
    question_text: str,
    options: list[str],
) -> str:
    """Compute a deterministic, reproducible ID from the normalized question content."""
    canonical = f"{level.upper()}|{category.lower()}|{question_text.strip()}|{'|'.join(o.strip() for o in options)}"
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    return f"jlpt-{level.lower()}-{category.lower()}-{digest}"


class PracticeGenerateRequest(BaseModel):
    """Bounded request payload for POST /api/practice/generate."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    level: JLPTLevel = Field(
        ..., description="Target JLPT proficiency level (N5 through N1)."
    )
    category: GenerationCategory = Field(
        ..., description="Practice category: 'vocabulary' or 'grammar'."
    )
    question_count: int = Field(
        default=1,
        ge=1,
        le=5,
        validation_alias=AliasChoices("question_count", "count"),
        description="Number of questions to generate in a single batch (1 to 5).",
    )
    explanation_language: ExplanationLanguage = Field(
        default=ExplanationLanguage.ENGLISH,
        description="Language for explanations: 'English', 'Korean', or 'Japanese'.",
    )

    @field_validator("category", mode="before")
    @classmethod
    def normalize_category(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("explanation_language", mode="before")
    @classmethod
    def normalize_explanation_language(cls, value: object) -> object:
        if isinstance(value, str):
            mapping = {
                "english": "English",
                "en": "English",
                "korean": "Korean",
                "ko": "Korean",
                "japanese": "Japanese",
                "ja": "Japanese",
            }
            return mapping.get(value.strip().lower(), value)
        return value


class LLMGeneratedQuestionItem(BaseModel):
    """Structured question item produced by the LLM before batch stabilization."""

    question_text: str = Field(
        ...,
        min_length=2,
        max_length=500,
        description="Original Japanese JLPT question sentence (often with a blank like （　）).",
    )
    options: list[str] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Four distinct multiple-choice answer options.",
    )
    correct_index: int = Field(
        ...,
        ge=0,
        le=3,
        description="Zero-based index (0 to 3) of the single correct option.",
    )
    option_explanations: list[str] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Four explanations corresponding to each option (index 0 to 3) in the requested language.",
    )

    @field_validator("question_text")
    @classmethod
    def validate_japanese_question_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not JAPANESE_SCRIPT_PATTERN.search(cleaned):
            raise ValueError(
                "Structural validation failed: question_text must contain Japanese script "
                "(Hiragana, Katakana, or Kanji)."
            )
        return cleaned

    @field_validator("options")
    @classmethod
    def validate_four_distinct_options(cls, value: list[str]) -> list[str]:
        if len(value) != 4:
            raise ValueError("Structural validation failed: exactly 4 options are required.")
        cleaned = [opt.strip() for opt in value]
        if any(not opt for opt in cleaned):
            raise ValueError("Structural validation failed: options must not be empty.")
        if len(set(cleaned)) != 4:
            raise ValueError(
                "Structural validation failed: all 4 options must be distinct."
            )
        return cleaned

    @field_validator("option_explanations")
    @classmethod
    def validate_four_option_explanations(cls, value: list[str]) -> list[str]:
        if len(value) != 4:
            raise ValueError(
                "Structural validation failed: exactly 4 option explanations are required."
            )
        cleaned = [exp.strip() for exp in value]
        if any(not exp for exp in cleaned):
            raise ValueError(
                "Structural validation failed: each option explanation must be non-empty."
            )
        return cleaned


class LLMPracticeBatchOutput(BaseModel):
    """Top-level structured output schema requested from the LLM in a single batch call."""

    questions: list[LLMGeneratedQuestionItem] = Field(
        ...,
        min_length=1,
        max_length=5,
        description="List of generated JLPT practice questions.",
    )


class GeneratedPracticeQuestion(BaseModel):
    """Validated, AI-labeled JLPT practice question returned to the client."""

    id: str = Field(
        ...,
        min_length=5,
        max_length=80,
        description="Deterministic stable question identifier derived from content.",
    )
    question_text: str = Field(
        ...,
        description="Japanese question prompt text.",
    )
    options: list[str] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Four distinct answer choices.",
    )
    correct_index: int = Field(
        ...,
        ge=0,
        le=3,
        description="Zero-based index of the single correct option (0-3).",
    )
    correct_answer: str = Field(
        ...,
        description="The correct option string matching options[correct_index].",
    )
    option_explanations: list[str] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Explanation for each of the four options (indices 0-3).",
    )
    ai_generated: Literal[True] = Field(
        default=True,
        description="Indicates this question is AI-generated practice material.",
    )
    content_label: str = Field(
        default="AI-generated original JLPT practice material",
        description="Human-readable label identifying the material as AI-generated.",
    )

    @field_validator("question_text")
    @classmethod
    def check_japanese_text(cls, value: str) -> str:
        return LLMGeneratedQuestionItem.validate_japanese_question_text(value)

    @field_validator("options")
    @classmethod
    def check_distinct_options(cls, value: list[str]) -> list[str]:
        return LLMGeneratedQuestionItem.validate_four_distinct_options(value)

    @field_validator("option_explanations")
    @classmethod
    def check_option_explanations(cls, value: list[str]) -> list[str]:
        return LLMGeneratedQuestionItem.validate_four_option_explanations(value)

    @model_validator(mode="after")
    def verify_correct_answer_consistency(self) -> "GeneratedPracticeQuestion":
        """Ensure correct_index is in bounds and matches correct_answer."""
        if not (0 <= self.correct_index < len(self.options)):
            raise ValueError(
                "Structural validation failed: correct_index is out of bounds for options."
            )
        expected_answer = self.options[self.correct_index]
        if self.correct_answer != expected_answer:
            raise ValueError(
                "Structural validation failed: correct_answer does not match options[correct_index]."
            )
        return self


class ValidationReport(BaseModel):
    """Explicit separation between structural validation and linguistic correctness."""

    structural_validation_passed: Literal[True] = True
    linguistic_correctness_guaranteed: Literal[False] = False
    notice: str = Field(default=STRUCTURAL_VS_LINGUISTIC_DISCLAIMER)


class PracticeGenerateResponse(BaseModel):
    """Response schema for POST /api/practice/generate."""

    level: JLPTLevel
    category: GenerationCategory
    question_count: int = Field(..., ge=1, le=5)
    explanation_language: ExplanationLanguage
    ai_generated: Literal[True] = True
    content_label: str = Field(
        default="AI-generated original JLPT practice material (not official past exam items)"
    )
    self_study_notice: str = Field(default=SELF_STUDY_GRADING_NOTICE)
    validation: ValidationReport = Field(default_factory=ValidationReport)
    questions: list[GeneratedPracticeQuestion] = Field(..., min_length=1, max_length=5)

    @model_validator(mode="after")
    def verify_batch_integrity(self) -> "PracticeGenerateResponse":
        """Verify batch count matches requested question_count and question IDs are unique."""
        if len(self.questions) != self.question_count:
            raise ValueError(
                f"Structural validation failed: expected {self.question_count} questions, "
                f"received {len(self.questions)}."
            )
        ids = [q.id for q in self.questions]
        if len(set(ids)) != len(ids):
            raise ValueError(
                "Structural validation failed: duplicate question IDs in generated batch."
            )
        return self


class AlternativeExplanation(BaseModel):
    """Explanation for why a distractor option does not fit."""

    option_index: int = Field(..., ge=0, le=3)
    option_text: str
    was_selected: bool = False
    why_it_does_not_fit: str


class AnswerEvaluationRequest(BaseModel):
    """Request payload for deterministic, non-Gemini multiple-choice answer evaluation."""

    level: JLPTLevel = Field(default=JLPTLevel.N5)
    question: str = Field(..., min_length=1, max_length=2000)
    selected_answer: str = Field(..., min_length=1, max_length=500)
    correct_answer: Optional[str] = Field(default=None, max_length=500)
    options: Optional[list[str]] = Field(default=None, min_length=4, max_length=4)
    correct_index: Optional[int] = Field(default=None, ge=0, le=3)
    option_explanations: Optional[list[str]] = Field(
        default=None, min_length=4, max_length=4
    )


class AnswerEvaluationResponse(BaseModel):
    """Local/deterministic feedback for a submitted multiple-choice answer (no Gemini call)."""

    is_correct: bool
    selected_answer: str = ""
    correct_answer: str = ""
    why_correct_fits: str = ""
    why_alternatives_do_not_fit: list[AlternativeExplanation] = Field(
        default_factory=list
    )
    feedback: str
    grammar_notes: list[str] = Field(default_factory=list)
    self_study_mode: Literal[True] = True
    tamper_proof: Literal[False] = False
    scoring_notice: str = Field(default=SELF_STUDY_GRADING_NOTICE)


class ChatMessage(BaseModel):
    """Single message in an ephemeral, browser-maintained conversation history."""

    role: Literal["user", "assistant"] = Field(
        ..., description="Speaker role in the conversation."
    )
    content: str = Field(
        ..., min_length=1, max_length=2000, description="Message text content."
    )


class PracticeQuestionRequest(BaseModel):
    """Request payload for generating a single JLPT practice question."""

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
    """Structured single JLPT practice question returned by the AI service."""

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
