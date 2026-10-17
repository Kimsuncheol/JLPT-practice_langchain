"""Pydantic schemas for JLPT practice generation, local quiz feedback, tutor chat, and health checks.

Distinguishes:
1. Structural validation (enforced deterministically by Pydantic validators) vs.
   linguistic correctness (not guaranteed for AI-generated content).
2. Self-study local grading: answer keys are held in browser memory (inspectable by the
   learner) and compared locally without calling Ollama or persisting scores/answers.
   Scoring is explicitly for self-study, is never described as tamper-proof, and does not
   predict an official JLPT score.
3. Unofficial AI practice & approximate difficulty: all generated material is labeled as
   unofficial AI-generated practice with approximate JLPT difficulty.
4. Server-controlled tutor validation: strictly bounds message roles ('user', 'assistant' only),
   message lengths, recent history window size, and selected practice question context.
"""

import hashlib
import re
from enum import Enum
from typing import Literal, Optional, Union
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

MAX_TUTOR_HISTORY_MESSAGES = 6
MAX_TUTOR_MESSAGE_LENGTH = 500
MAX_READING_QUESTIONS = 3
MAX_READING_PASSAGE_CHARS = 800

STRUCTURAL_VS_LINGUISTIC_DISCLAIMER = (
    "Structural validation passed (schema, Japanese script presence, 4 distinct options, "
    "single valid answer index, and per-option explanations verified). "
    "Note: Structural validation does not guarantee linguistic nuance or pedagogical accuracy "
    "of AI-generated content."
)

SELF_STUDY_GRADING_NOTICE = (
    "Self-study mode: The answer key is held in browser memory and compared locally without "
    "calling the Ollama model or storing scores. Because a browser-held answer key is inspectable, "
    "this scoring is for self-study feedback only, is not tamper-proof, and does not "
    "predict an official JLPT score."
)

APPROXIMATE_DIFFICULTY_LABEL = (
    "Approximate JLPT difficulty (unofficial AI-generated estimate)"
)

UNOFFICIAL_AI_CONTENT_LABEL = (
    "Unofficial AI-generated original JLPT practice material (not official exam items)"
)

SCORE_PREDICTION_DISCLAIMER = (
    "Unofficial AI practice with approximate difficulty: practice accuracy is for self-study "
    "only and does not predict or guarantee an official JLPT score."
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
    """Supported categories for POST /api/practice/generate (vocabulary, grammar, reading)."""

    VOCABULARY = "vocabulary"
    GRAMMAR = "grammar"
    READING = "reading"


class ExplanationLanguage(str, Enum):
    """Supported languages for option and tutor explanations."""

    ENGLISH = "English"
    KOREAN = "Korean"
    JAPANESE = "Japanese"


def _normalize_explanation_language_value(value: object) -> object:
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
        ...,
        description="Practice category: 'vocabulary', 'grammar', or 'reading'.",
    )
    question_count: int = Field(
        default=1,
        ge=1,
        le=5,
        validation_alias=AliasChoices("question_count", "count"),
        description="Number of questions to generate (1-5 for vocabulary/grammar; 1-3 for reading).",
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
        return _normalize_explanation_language_value(value)

    @model_validator(mode="after")
    def validate_reading_question_count(self) -> "PracticeGenerateRequest":
        """Bound reading category to at most 3 multiple-choice questions per passage."""
        if (
            self.category == GenerationCategory.READING
            and self.question_count > MAX_READING_QUESTIONS
        ):
            raise ValueError(
                f"Reading category supports at most {MAX_READING_QUESTIONS} questions per passage (1 to 3)."
            )
        return self


class LLMGeneratedQuestionItem(BaseModel):
    """Structured question item produced by the LLM before batch stabilization."""

    question_text: str = Field(
        ...,
        min_length=2,
        max_length=500,
        description="Original Japanese JLPT question sentence.",
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
    supporting_sentence: Optional[str] = Field(
        default=None,
        max_length=350,
        description="For reading questions, the exact supporting sentence from the passage.",
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

    passage: Optional[str] = Field(
        default=None,
        max_length=MAX_READING_PASSAGE_CHARS,
        description="Short original Japanese reading passage (required when category is 'reading').",
    )
    questions: list[LLMGeneratedQuestionItem] = Field(
        ...,
        min_length=1,
        max_length=5,
        description="List of generated JLPT practice questions.",
    )

    @field_validator("passage")
    @classmethod
    def validate_passage_if_present(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            return None
        if len(cleaned) > MAX_READING_PASSAGE_CHARS:
            raise ValueError(
                f"Structural validation failed: reading passage exceeds {MAX_READING_PASSAGE_CHARS} characters."
            )
        if not JAPANESE_SCRIPT_PATTERN.search(cleaned):
            raise ValueError(
                "Structural validation failed: reading passage must contain Japanese script."
            )
        return cleaned


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
    supporting_sentence: Optional[str] = Field(
        default=None,
        description="Exact supporting sentence from the reading passage (for reading questions).",
    )
    ai_generated: Literal[True] = Field(
        default=True,
        description="Indicates this question is unofficial AI-generated practice material.",
    )
    difficulty_label: str = Field(
        default=APPROXIMATE_DIFFICULTY_LABEL,
        description="Labels difficulty as an approximate estimate.",
    )
    content_label: str = Field(
        default=UNOFFICIAL_AI_CONTENT_LABEL,
        description="Human-readable label identifying the material as unofficial AI practice.",
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
    official_score_predictive: Literal[False] = False
    notice: str = Field(default=STRUCTURAL_VS_LINGUISTIC_DISCLAIMER)


class PracticeGenerateResponse(BaseModel):
    """Response schema for POST /api/practice/generate (vocabulary, grammar, and reading)."""

    level: JLPTLevel
    category: GenerationCategory
    question_count: int = Field(..., ge=1, le=5)
    explanation_language: ExplanationLanguage
    passage: Optional[str] = Field(
        default=None,
        max_length=MAX_READING_PASSAGE_CHARS,
        description="Short original Japanese passage when category is 'reading'.",
    )
    ai_generated: Literal[True] = True
    unofficial_practice: Literal[True] = True
    difficulty_label: str = Field(default=APPROXIMATE_DIFFICULTY_LABEL)
    content_label: str = Field(default=UNOFFICIAL_AI_CONTENT_LABEL)
    score_disclaimer: str = Field(default=SCORE_PREDICTION_DISCLAIMER)
    self_study_notice: str = Field(default=SELF_STUDY_GRADING_NOTICE)
    validation: ValidationReport = Field(default_factory=ValidationReport)
    questions: list[GeneratedPracticeQuestion] = Field(..., min_length=1, max_length=5)

    @model_validator(mode="after")
    def verify_batch_integrity(self) -> "PracticeGenerateResponse":
        """Verify batch count, unique question IDs, and reading passage/supporting sentence rules."""
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
        if self.category == GenerationCategory.READING:
            if self.question_count > MAX_READING_QUESTIONS:
                raise ValueError(
                    f"Structural validation failed: reading category allows at most {MAX_READING_QUESTIONS} questions."
                )
            if not self.passage or not JAPANESE_SCRIPT_PATTERN.search(self.passage):
                raise ValueError(
                    "Structural validation failed: reading category requires a Japanese passage."
                )
            for q in self.questions:
                if not q.supporting_sentence or not JAPANESE_SCRIPT_PATTERN.search(
                    q.supporting_sentence
                ):
                    raise ValueError(
                        "Structural validation failed: each reading question requires a Japanese supporting_sentence."
                    )
                if not all(q.supporting_sentence in exp for exp in q.option_explanations):
                    raise ValueError(
                        "Structural validation failed: each reading option explanation must include the supporting sentence."
                    )
        return self


class AlternativeExplanation(BaseModel):
    """Explanation for why a distractor option does not fit."""

    option_index: int = Field(..., ge=0, le=3)
    option_text: str
    was_selected: bool = False
    why_it_does_not_fit: str


class AnswerEvaluationRequest(BaseModel):
    """Request payload for deterministic, non-LLM multiple-choice answer evaluation."""

    level: JLPTLevel = Field(default=JLPTLevel.N5)
    question: str = Field(..., min_length=1, max_length=2000)
    selected_answer: str = Field(..., min_length=1, max_length=500)
    correct_answer: Optional[str] = Field(default=None, max_length=500)
    options: Optional[list[str]] = Field(default=None, min_length=4, max_length=4)
    correct_index: Optional[int] = Field(default=None, ge=0, le=3)
    option_explanations: Optional[list[str]] = Field(
        default=None, min_length=4, max_length=4
    )
    supporting_sentence: Optional[str] = Field(default=None, max_length=350)


class AnswerEvaluationResponse(BaseModel):
    """Local/deterministic feedback for a submitted multiple-choice answer (no LLM call)."""

    is_correct: bool
    selected_answer: str = ""
    correct_answer: str = ""
    supporting_sentence: Optional[str] = None
    why_correct_fits: str = ""
    why_alternatives_do_not_fit: list[AlternativeExplanation] = Field(
        default_factory=list
    )
    feedback: str
    grammar_notes: list[str] = Field(default_factory=list)
    self_study_mode: Literal[True] = True
    tamper_proof: Literal[False] = False
    official_score_predictive: Literal[False] = False
    difficulty_label: str = Field(default=APPROXIMATE_DIFFICULTY_LABEL)
    scoring_notice: str = Field(default=SELF_STUDY_GRADING_NOTICE)
    score_disclaimer: str = Field(default=SCORE_PREDICTION_DISCLAIMER)


class ChatMessage(BaseModel):
    """Single message in an ephemeral, browser-maintained conversation window.

    Server-side role validation strictly permits only 'user' and 'assistant'
    (rejecting 'system', 'developer', or 'tool' roles to prevent prompt injection).
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    role: Literal["user", "assistant"] = Field(
        ...,
        description="Speaker role ('user' or 'assistant' only; 'system' is forbidden).",
    )
    content: str = Field(
        ...,
        min_length=1,
        max_length=MAX_TUTOR_MESSAGE_LENGTH,
        description="Bounded message text content (1 to 500 characters).",
    )


class SelectedPracticeQuestionContext(BaseModel):
    """Bounded context for the single selected practice item sent to POST /api/tutor."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    id: Optional[str] = Field(default=None, max_length=80)
    passage: Optional[str] = Field(default=None, max_length=MAX_READING_PASSAGE_CHARS)
    question_text: str = Field(
        ...,
        min_length=1,
        max_length=500,
        validation_alias=AliasChoices("question_text", "question"),
        description="Japanese text of the selected practice question.",
    )
    options: list[str] = Field(
        default_factory=list,
        max_length=4,
        description="Up to 4 options for the selected question.",
    )
    correct_index: Optional[int] = Field(default=None, ge=0, le=3)
    correct_answer: Optional[str] = Field(default=None, max_length=200)
    selected_answer: Optional[str] = Field(default=None, max_length=200)
    supporting_sentence: Optional[str] = Field(default=None, max_length=350)
    option_explanations: list[str] = Field(
        default_factory=list,
        max_length=4,
        description="Per-option explanations for the selected question.",
    )

    @field_validator("options")
    @classmethod
    def bound_option_lengths(cls, value: list[str]) -> list[str]:
        for opt in value:
            if len(opt) > 200:
                raise ValueError("Each option must be at most 200 characters.")
        return value

    @field_validator("option_explanations")
    @classmethod
    def bound_explanation_lengths(cls, value: list[str]) -> list[str]:
        for exp in value:
            if len(exp) > 600:
                raise ValueError("Each option explanation must be at most 600 characters.")
        return value


class TutorRequest(BaseModel):
    """Bounded, stateless request for POST /api/tutor about the current practice item."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    level: JLPTLevel = Field(
        default=JLPTLevel.N5, description="Target JLPT level (N5-N1)."
    )
    explanation_language: ExplanationLanguage = Field(
        default=ExplanationLanguage.ENGLISH,
        description="Explanation language: English, Korean, or Japanese.",
    )
    question: Union[SelectedPracticeQuestionContext, str] = Field(
        ...,
        validation_alias=AliasChoices("question", "current_question", "practice_item"),
        description="The single selected practice item (structured object or question text).",
    )
    message: str = Field(
        ...,
        min_length=1,
        max_length=MAX_TUTOR_MESSAGE_LENGTH,
        validation_alias=AliasChoices("message", "user_message", "query"),
        description="Learner's question about the selected practice item (1-500 chars).",
    )
    history: list[ChatMessage] = Field(
        default_factory=list,
        max_length=MAX_TUTOR_HISTORY_MESSAGES,
        validation_alias=AliasChoices("history", "recent_messages", "messages"),
        description="Bounded recent conversation window from browser memory (max 6 messages).",
    )

    @field_validator("explanation_language", mode="before")
    @classmethod
    def normalize_language(cls, value: object) -> object:
        return _normalize_explanation_language_value(value)

    @field_validator("question", mode="after")
    @classmethod
    def normalize_question_context(
        cls, value: Union[SelectedPracticeQuestionContext, str]
    ) -> SelectedPracticeQuestionContext:
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned or len(cleaned) > 500:
                raise ValueError(
                    "Question text must be between 1 and 500 characters."
                )
            return SelectedPracticeQuestionContext(question_text=cleaned)
        return value


class TutorResponse(BaseModel):
    """Stateless tutor response for POST /api/tutor."""

    level: JLPTLevel
    explanation_language: ExplanationLanguage
    question_id: Optional[str] = None
    reply: str
    ai_generated: Literal[True] = True


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
    explanation_language: ExplanationLanguage = Field(
        default=ExplanationLanguage.ENGLISH,
        description="Explanation language: English, Korean, or Japanese.",
    )
    question: Optional[Union[SelectedPracticeQuestionContext, str]] = Field(
        default=None,
        validation_alias=AliasChoices("question", "current_question", "practice_item"),
    )
    message: str = Field(
        ...,
        min_length=1,
        max_length=MAX_TUTOR_MESSAGE_LENGTH,
        description="Current learner question.",
    )
    history: list[ChatMessage] = Field(
        default_factory=list,
        max_length=MAX_TUTOR_HISTORY_MESSAGES,
        description="Bounded recent conversation window held only in browser and request-local memory.",
    )

    @field_validator("explanation_language", mode="before")
    @classmethod
    def normalize_language(cls, value: object) -> object:
        return _normalize_explanation_language_value(value)


class TutorChatResponse(BaseModel):
    """Stateless tutor reply."""

    level: JLPTLevel
    explanation_language: ExplanationLanguage = ExplanationLanguage.ENGLISH
    reply: str


class HealthResponse(BaseModel):
    """Health check response verifying service readiness without calling Ollama."""

    status: Literal["ok"] = "ok"
    model: str
    base_url_configured: bool = True
    tracing_disabled: bool
    persistence: Literal["none"] = "none"
