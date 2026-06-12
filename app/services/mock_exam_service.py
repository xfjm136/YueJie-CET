from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any

from app.ai.pipelines import SubjectiveEvaluationPipeline
from app.constants import QUESTION_TYPE_LABELS, SKILL_ADVICE, SKILL_LABELS
from app.data.db import Database
from app.domain.enums import Level, QuestionType
from app.domain.scoring import grade_attempt
from app.domain.schemas import (
    AttemptQuestionResult,
    AttemptResult,
    MockExamRecord,
    MockExamSectionRecord,
    VocabularyItem,
    make_id,
)
from app.services.weakness_service import WeaknessService


class MockExamService:
    SECTION_WEIGHTS: dict[tuple[str, int | None], float] = {
        ("writing", None): 15 / 65 * 100,
        ("translation", None): 15 / 65 * 100,
        ("banked_cloze", None): 5 / 65 * 100,
        ("long_reading", None): 10 / 65 * 100,
        ("careful_reading", 1): 10 / 65 * 100,
        ("careful_reading", 2): 10 / 65 * 100,
    }

    def __init__(
        self,
        db: Database,
        weakness_service: WeaknessService,
        subjective_evaluator: SubjectiveEvaluationPipeline | None = None,
    ) -> None:
        self.db = db
        self.weakness_service = weakness_service
        self.subjective_evaluator = subjective_evaluator

    def submit_mock_exam(
        self,
        level: Level,
        sections: list[dict[str, Any]],
        started_at: datetime,
        submitted_at: datetime | None = None,
        duration_seconds_override: int | None = None,
    ) -> MockExamRecord:
        submitted = submitted_at or datetime.now(timezone.utc)
        duration_seconds = max(
            1,
            int(duration_seconds_override)
            if duration_seconds_override is not None
            else int((submitted - started_at).total_seconds()),
        )
        section_records: list[MockExamSectionRecord] = []
        score_breakdown: dict[str, float] = {}
        wrong_skills: Counter[str] = Counter()

        for item in sections:
            question_set_id = str(item.get("question_set_id", "")).strip()
            if not question_set_id:
                raise ValueError("missing question_set_id in mock exam section")
            question_set = self.db.get_question_set(question_set_id)
            if question_set is None:
                raise ValueError(f"question set not found: {question_set_id}")
            answers = {
                str(key): str(value)
                for key, value in dict(item.get("answers", {})).items()
            }
            section_duration_seconds = max(1, int(item.get("duration_seconds", 1) or 1))

            if question_set.is_subjective:
                response_text = answers.get("response_text", "").strip()
                if not response_text:
                    raise ValueError("missing response_text for mock exam subjective section")
                if self.subjective_evaluator is None:
                    raise RuntimeError("subjective evaluator is not configured")
                evaluation = self.subjective_evaluator.evaluate(
                    question_set=question_set,
                    response_text=response_text,
                    duration_seconds=section_duration_seconds,
                    exam_mode=True,
                )
                result = AttemptResult(
                    id=make_id("mock_attempt"),
                    question_set_id=question_set.id,
                    correct_count=0,
                    total_count=0,
                    accuracy=round(evaluation.score_15 / 15.0, 4),
                    duration_seconds=section_duration_seconds,
                    summary=evaluation.overall_feedback_zh,
                    recommendations=[
                        f"{SKILL_LABELS.get(tag, tag)}：{SKILL_ADVICE.get(tag, SKILL_ADVICE['general'])}"
                        for tag in evaluation.weakness_tags
                    ] or ["优先根据批注重写并修正高频语法与词汇问题。"],
                    question_results=[
                        AttemptQuestionResult(
                            question_id=f"subjective-{index + 1}",
                            user_answer=item.original_sentence,
                            correct_answer=item.revised_sentence,
                            is_correct=False,
                            explanation=item.reason_zh,
                            skill_tag=item.skill_tag,
                        )
                        for index, item in enumerate(evaluation.sentence_rewrites)
                    ],
                    subjective_evaluation=evaluation,
                )
                for tag in evaluation.weakness_tags:
                    wrong_skills[tag] += 1
            else:
                result = grade_attempt(question_set, answers, section_duration_seconds)
                for question_result in result.question_results:
                    if not question_result.is_correct:
                        wrong_skills[question_result.skill_tag] += 1

            section_records.append(
                MockExamSectionRecord(
                    question_type=question_set.question_type,
                    slot=question_set.slot,
                    question_set=question_set,
                    answers=answers,
                    result=result,
                )
            )
            score_breakdown[self._section_key(question_set.question_type, question_set.slot)] = round(
                result.accuracy * self._section_weight(question_set.question_type, question_set.slot),
                2,
            )

        total_score = round(sum(score_breakdown.values()), 2)
        weak_tags = [skill for skill, _count in wrong_skills.most_common(3)]
        summary = (
            f"{level.value.upper()} 模拟四六级考试已完成，共 {len(section_records)} 个部分。"
            f" 当前综合得分 {total_score:.1f} / 100。"
        )
        recommendations = [
            f"{SKILL_LABELS.get(tag, tag)}：{SKILL_ADVICE.get(tag, SKILL_ADVICE['general'])}"
            for tag in weak_tags
        ] or ["先完成整套复盘，再按题型回到普通刷题中做针对性强化。"]

        record = MockExamRecord(
            id=make_id("mock_exam"),
            level=level,
            started_at=started_at,
            submitted_at=submitted,
            duration_seconds=duration_seconds,
            total_score=total_score,
            score_breakdown=score_breakdown,
            summary=summary,
            recommendations=recommendations,
            weakness_tags=weak_tags,
            sections=section_records,
        )
        self.db.save_mock_exam_record(record)
        self.weakness_service.refresh_mock_exam_snapshot(level)
        return record

    @staticmethod
    def _section_key(question_type: QuestionType, slot: int | None) -> str:
        if question_type is QuestionType.CAREFUL_READING:
            return f"careful_reading_{slot or 1}"
        return question_type.value

    @classmethod
    def _section_weight(cls, question_type: QuestionType, slot: int | None) -> float:
        return cls.SECTION_WEIGHTS.get(
            (question_type.value, slot),
            cls.SECTION_WEIGHTS.get((question_type.value, None), 0.0),
        )
