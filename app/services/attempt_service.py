from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

from app.constants import SKILL_ADVICE, SKILL_LABELS
from app.data.db import Database
from app.domain.enums import Level, QuestionType
from app.domain.scoring import grade_attempt
from app.domain.schemas import (
    AttemptQuestionResult,
    AttemptResult,
    QuestionSet,
    VocabularyItem,
    make_id,
)
from app.ai.pipelines import SubjectiveEvaluationPipeline
from app.services.weakness_service import WeaknessService


class AttemptService:
    def __init__(
        self,
        db: Database,
        weakness_service: WeaknessService,
        subjective_evaluator: SubjectiveEvaluationPipeline | None = None,
    ) -> None:
        self.db = db
        self.weakness_service = weakness_service
        self.subjective_evaluator = subjective_evaluator

    def submit_attempt(
        self,
        question_set: QuestionSet,
        answers: dict[str, str],
        started_at: datetime,
        is_history_retry: bool = False,
        progress_callback: Callable[[str, str], None] | None = None,
    ) -> AttemptResult:
        submitted_at = datetime.now(timezone.utc)
        duration_seconds = max(1, int((submitted_at - started_at).total_seconds()))
        if question_set.is_subjective:
            response_text = answers.get("response_text", "").strip()
            if not response_text:
                raise ValueError("missing response_text for subjective task")
            if self.subjective_evaluator is None:
                raise RuntimeError("subjective evaluator is not configured")
            if progress_callback is not None:
                progress_callback(
                    "prepare",
                    f"已读取作答内容，约 {len(response_text.split())} 词；正在整理评分维度。",
                )
                progress_callback("score_request", "正在向 DeepSeek 请求评分与批注。")
            evaluation = self.subjective_evaluator.evaluate(
                question_set=question_set,
                response_text=response_text,
                duration_seconds=duration_seconds,
            )
            if progress_callback is not None:
                progress_callback("analysis", "评分结果已返回，正在整理错词、病句和高分版本。")
            result = AttemptResult(
                id=make_id("attempt"),
                question_set_id=question_set.id,
                correct_count=0,
                total_count=0,
                accuracy=round(evaluation.score_15 / 15.0, 4),
                duration_seconds=duration_seconds,
                summary=evaluation.overall_feedback_zh,
                recommendations=[
                    f"{SKILL_LABELS.get(tag, tag)}：{SKILL_ADVICE.get(tag, SKILL_ADVICE['general'])}"
                    for tag in evaluation.weakness_tags
                ] or ["先根据病句改写和错词纠正重写一版。"],
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
            extra_vocab = [
                VocabularyItem(
                    lemma=item.corrected.lower(),
                    surface_form=item.corrected,
                    level_hint=question_set.level.value,
                    meaning_zh=item.meaning_zh,
                    example_en=item.corrected,
                    error_related_score=1,
                )
                for item in evaluation.wrong_words
                if item.corrected.strip()
            ]
        else:
            if progress_callback is not None:
                progress_callback("grade", "正在判分并整理客观题结果。")
            result = grade_attempt(question_set, answers, duration_seconds)
            extra_vocab = []
        if progress_callback is not None:
            progress_callback("save", "正在保存作答记录、词汇与薄弱项。")
        self.db.save_attempt(result, answers, started_at, submitted_at, is_history_retry)
        self.db.upsert_vocabulary_items(question_set.vocabulary)
        if extra_vocab:
            self.db.upsert_vocabulary_items(extra_vocab)
        self.weakness_service.refresh_snapshot(question_set.level, question_set.question_type)
        return result

    def delete_attempt_history(self, attempt_id: str) -> dict[str, str | bool]:
        deleted = self.db.delete_attempt_history(attempt_id)
        if deleted is None:
            raise ValueError("attempt not found")
        self.weakness_service.rebuild_snapshots()
        return deleted
