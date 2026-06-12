from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from app.domain.enums import Level, QuestionType


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def make_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value):
        return {k: _plain(v) for k, v in asdict(value).items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return value


@dataclass
class Passage:
    title: str
    paragraphs: list[str]

    def to_dict(self) -> dict[str, Any]:
        return _plain(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Passage":
        return cls(title=data["title"], paragraphs=list(data["paragraphs"]))


@dataclass
class Question:
    id: str
    prompt: str
    options: list[str] = field(default_factory=list)
    skill_tag: str = "general"
    hint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _plain(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Question":
        return cls(
            id=data["id"],
            prompt=data["prompt"],
            options=list(data.get("options", [])),
            skill_tag=data.get("skill_tag", "general"),
            hint=data.get("hint"),
        )


@dataclass
class ItemExplanation:
    question_id: str
    correct_answer: str
    explanation: str
    skill_tag: str = "general"

    def to_dict(self) -> dict[str, Any]:
        return _plain(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ItemExplanation":
        return cls(
            question_id=data["question_id"],
            correct_answer=data["correct_answer"],
            explanation=data["explanation"],
            skill_tag=data.get("skill_tag", "general"),
        )


@dataclass
class AnalysisReport:
    overall_strategy: str
    overall_summary: str
    item_explanations: list[ItemExplanation] = field(default_factory=list)
    test_tips: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _plain(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalysisReport":
        return cls(
            overall_strategy=data.get("overall_strategy", ""),
            overall_summary=data.get("overall_summary", ""),
            item_explanations=[
                ItemExplanation.from_dict(item) for item in data.get("item_explanations", [])
            ],
            test_tips=list(data.get("test_tips", [])),
        )


@dataclass
class VocabularyItem:
    lemma: str
    surface_form: str
    level_hint: str
    meaning_zh: str
    example_en: str = ""
    frequency_score: int = 1
    error_related_score: int = 0
    last_seen_at: datetime = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return _plain(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VocabularyItem":
        return cls(
            lemma=data["lemma"],
            surface_form=data.get("surface_form", data["lemma"]),
            level_hint=data.get("level_hint", ""),
            meaning_zh=data.get("meaning_zh", ""),
            example_en=data.get("example_en", ""),
            frequency_score=int(data.get("frequency_score", 1)),
            error_related_score=int(data.get("error_related_score", 0)),
            last_seen_at=datetime.fromisoformat(data["last_seen_at"])
            if data.get("last_seen_at")
            else utc_now(),
        )


@dataclass
class ScoreDimension:
    name: str
    score: float
    max_score: float
    feedback_zh: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _plain(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScoreDimension":
        return cls(
            name=data["name"],
            score=float(data.get("score", 0.0)),
            max_score=float(data.get("max_score", 0.0)),
            feedback_zh=data.get("feedback_zh", ""),
        )


@dataclass
class WordCorrection:
    original: str
    corrected: str
    reason_zh: str
    meaning_zh: str = ""
    skill_tag: str = "lexical_accuracy"

    def to_dict(self) -> dict[str, Any]:
        return _plain(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WordCorrection":
        return cls(
            original=data["original"],
            corrected=data["corrected"],
            meaning_zh=data.get("meaning_zh", ""),
            reason_zh=data.get("reason_zh", ""),
            skill_tag=data.get("skill_tag", "lexical_accuracy"),
        )


@dataclass
class SentenceRewrite:
    original_sentence: str
    revised_sentence: str
    reason_zh: str
    skill_tag: str = "grammar"

    def to_dict(self) -> dict[str, Any]:
        return _plain(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SentenceRewrite":
        return cls(
            original_sentence=data["original_sentence"],
            revised_sentence=data["revised_sentence"],
            reason_zh=data.get("reason_zh", ""),
            skill_tag=data.get("skill_tag", "grammar"),
        )


@dataclass
class SentenceAnnotation:
    original_sentence: str
    strengths_zh: str = ""
    issues_zh: str = ""
    revised_sentence: str = ""
    skill_tag: str = "general"

    def to_dict(self) -> dict[str, Any]:
        return _plain(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SentenceAnnotation":
        return cls(
            original_sentence=data.get("original_sentence", ""),
            strengths_zh=data.get("strengths_zh", ""),
            issues_zh=data.get("issues_zh", ""),
            revised_sentence=data.get("revised_sentence", ""),
            skill_tag=data.get("skill_tag", "general"),
        )


@dataclass
class SubjectiveEvaluation:
    score_15: float
    estimated_reported_score: float
    grade_band: str
    overall_feedback_zh: str
    score_dimensions: list[ScoreDimension] = field(default_factory=list)
    wrong_words: list[WordCorrection] = field(default_factory=list)
    sentence_rewrites: list[SentenceRewrite] = field(default_factory=list)
    sentence_annotations: list[SentenceAnnotation] = field(default_factory=list)
    high_score_version: str = ""
    weakness_tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _plain(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SubjectiveEvaluation":
        return cls(
            score_15=float(data.get("score_15", 0.0)),
            estimated_reported_score=float(data.get("estimated_reported_score", 0.0)),
            grade_band=data.get("grade_band", ""),
            overall_feedback_zh=data.get("overall_feedback_zh", ""),
            score_dimensions=[
                ScoreDimension.from_dict(item) for item in data.get("score_dimensions", [])
            ],
            wrong_words=[
                WordCorrection.from_dict(item) for item in data.get("wrong_words", [])
            ],
            sentence_rewrites=[
                SentenceRewrite.from_dict(item) for item in data.get("sentence_rewrites", [])
            ],
            sentence_annotations=[
                SentenceAnnotation.from_dict(item)
                for item in data.get("sentence_annotations", [])
            ],
            high_score_version=data.get("high_score_version", ""),
            weakness_tags=list(data.get("weakness_tags", [])),
        )


@dataclass
class QuestionSet:
    id: str
    level: Level
    question_type: QuestionType
    title: str
    topic: str
    passage: Passage
    questions: list[Question]
    answer_key: list[str]
    analysis: AnalysisReport
    vocabulary: list[VocabularyItem]
    shared_options: list[str] = field(default_factory=list)
    task_prompt: str = ""
    reference_answer: str = ""
    rubric_focus: list[str] = field(default_factory=list)
    min_response_words: int = 0
    max_response_words: int = 0
    slot: int | None = None
    word_count: int = 0
    created_at: datetime = field(default_factory=utc_now)
    generator_model: str = "mock"
    source_type: str = "ai"

    @property
    def layout_mode(self) -> str:
        return (
            "two"
            if self.question_type
            in {
                QuestionType.LONG_READING,
                QuestionType.WRITING,
                QuestionType.TRANSLATION,
            }
            else "three"
        )

    @property
    def is_subjective(self) -> bool:
        return self.question_type in {QuestionType.WRITING, QuestionType.TRANSLATION}

    def to_dict(self) -> dict[str, Any]:
        return _plain(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QuestionSet":
        return cls(
            id=data["id"],
            level=Level(data["level"]),
            question_type=QuestionType(data["question_type"]),
            title=data["title"],
            topic=data["topic"],
            passage=Passage.from_dict(data["passage"]),
            questions=[Question.from_dict(item) for item in data["questions"]],
            answer_key=list(data["answer_key"]),
            analysis=AnalysisReport.from_dict(data["analysis"]),
            vocabulary=[VocabularyItem.from_dict(item) for item in data.get("vocabulary", [])],
            shared_options=list(data.get("shared_options", [])),
            task_prompt=data.get("task_prompt", ""),
            reference_answer=data.get("reference_answer", ""),
            rubric_focus=list(data.get("rubric_focus", [])),
            min_response_words=int(data.get("min_response_words", 0)),
            max_response_words=int(data.get("max_response_words", 0)),
            slot=data.get("slot"),
            word_count=int(data.get("word_count", 0)),
            created_at=datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else utc_now(),
            generator_model=data.get("generator_model", "mock"),
            source_type=data.get("source_type", "ai"),
        )


@dataclass
class AttemptQuestionResult:
    question_id: str
    user_answer: str
    correct_answer: str
    is_correct: bool
    explanation: str
    skill_tag: str

    def to_dict(self) -> dict[str, Any]:
        return _plain(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AttemptQuestionResult":
        return cls(
            question_id=data["question_id"],
            user_answer=data.get("user_answer", ""),
            correct_answer=data["correct_answer"],
            is_correct=bool(data["is_correct"]),
            explanation=data.get("explanation", ""),
            skill_tag=data.get("skill_tag", "general"),
        )


@dataclass
class AttemptResult:
    id: str
    question_set_id: str
    correct_count: int
    total_count: int
    accuracy: float
    duration_seconds: int
    summary: str
    recommendations: list[str]
    question_results: list[AttemptQuestionResult]
    subjective_evaluation: SubjectiveEvaluation | None = None
    created_at: datetime = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return _plain(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AttemptResult":
        return cls(
            id=data["id"],
            question_set_id=data["question_set_id"],
            correct_count=int(data["correct_count"]),
            total_count=int(data["total_count"]),
            accuracy=float(data["accuracy"]),
            duration_seconds=int(data["duration_seconds"]),
            summary=data.get("summary", ""),
            recommendations=list(data.get("recommendations", [])),
            question_results=[
                AttemptQuestionResult.from_dict(item)
                for item in data.get("question_results", [])
            ],
            subjective_evaluation=(
                SubjectiveEvaluation.from_dict(data["subjective_evaluation"])
                if data.get("subjective_evaluation")
                else None
            ),
            created_at=datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else utc_now(),
        )


@dataclass
class MockExamSectionRecord:
    question_type: QuestionType
    slot: int | None
    question_set: QuestionSet
    answers: dict[str, str]
    result: AttemptResult

    def to_dict(self) -> dict[str, Any]:
        return _plain(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MockExamSectionRecord":
        return cls(
            question_type=QuestionType(data["question_type"]),
            slot=data.get("slot"),
            question_set=QuestionSet.from_dict(data["question_set"]),
            answers={str(key): str(value) for key, value in dict(data.get("answers", {})).items()},
            result=AttemptResult.from_dict(data["result"]),
        )


@dataclass
class MockExamRecord:
    id: str
    level: Level
    started_at: datetime
    submitted_at: datetime
    duration_seconds: int
    total_score: float
    score_breakdown: dict[str, float]
    summary: str
    recommendations: list[str]
    weakness_tags: list[str]
    sections: list[MockExamSectionRecord] = field(default_factory=list)
    created_at: datetime = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return _plain(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MockExamRecord":
        return cls(
            id=data["id"],
            level=Level(data["level"]),
            started_at=datetime.fromisoformat(data["started_at"])
            if data.get("started_at")
            else utc_now(),
            submitted_at=datetime.fromisoformat(data["submitted_at"])
            if data.get("submitted_at")
            else utc_now(),
            duration_seconds=int(data.get("duration_seconds", 0)),
            total_score=float(data.get("total_score", 0.0)),
            score_breakdown={
                str(key): float(value)
                for key, value in dict(data.get("score_breakdown", {})).items()
            },
            summary=data.get("summary", ""),
            recommendations=list(data.get("recommendations", [])),
            weakness_tags=list(data.get("weakness_tags", [])),
            sections=[
                MockExamSectionRecord.from_dict(item)
                for item in data.get("sections", [])
            ],
            created_at=datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else utc_now(),
        )
