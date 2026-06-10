from __future__ import annotations

from datetime import datetime, timezone

from app.data.db import Database
from app.domain.enums import Level, QuestionType
from app.domain.scoring import grade_attempt
from app.domain.schemas import AttemptResult, QuestionSet
from app.services.weakness_service import WeaknessService


class AttemptService:
    def __init__(self, db: Database, weakness_service: WeaknessService) -> None:
        self.db = db
        self.weakness_service = weakness_service

    def submit_attempt(
        self,
        question_set: QuestionSet,
        answers: dict[str, str],
        started_at: datetime,
        is_history_retry: bool = False,
    ) -> AttemptResult:
        submitted_at = datetime.now(timezone.utc)
        duration_seconds = max(1, int((submitted_at - started_at).total_seconds()))
        result = grade_attempt(question_set, answers, duration_seconds)
        self.db.save_attempt(result, answers, started_at, submitted_at, is_history_retry)
        self.db.upsert_vocabulary_items(question_set.vocabulary)
        self.weakness_service.refresh_snapshot(question_set.level, question_set.question_type)
        return result

    def delete_attempt_history(self, attempt_id: str) -> dict[str, str | bool]:
        deleted = self.db.delete_attempt_history(attempt_id)
        if deleted is None:
            raise ValueError("attempt not found")
        self.weakness_service.rebuild_snapshots()
        return deleted
