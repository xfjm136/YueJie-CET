from __future__ import annotations

import json
from collections import Counter

from app.constants import QUESTION_TYPE_LABELS, SKILL_LABELS
from app.data.db import Database
from app.domain.enums import Level, QuestionType


class WeaknessService:
    def __init__(self, db: Database) -> None:
        self.db = db

    def refresh_snapshot(
        self,
        level: Level,
        question_type: QuestionType,
    ) -> dict[str, str | int | dict[str, float]] | None:
        payloads = self.db.recent_attempt_payloads(level, question_type, limit=10)
        if len(payloads) < 5:
            return None

        total_by_skill: Counter[str] = Counter()
        wrong_by_skill: Counter[str] = Counter()
        total_duration = 0

        for row in payloads:
            total_duration += int(row["duration_seconds"])
            result = json.loads(row["result_json"])
            for item in result.get("question_results", []):
                skill = item.get("skill_tag", "general")
                total_by_skill[skill] += 1
                if not item.get("is_correct", False):
                    wrong_by_skill[skill] += 1

        dimensions = {
            skill: round(100 * (1 - wrong_by_skill[skill] / total), 2)
            for skill, total in total_by_skill.items()
            if total > 0
        }
        avg_duration = total_duration / len(payloads)
        dimensions["speed"] = 100.0 if avg_duration <= 18 * 60 else max(30.0, 100.0 - (avg_duration - 18 * 60) / 12)

        weakest = sorted(
            ((skill, score) for skill, score in dimensions.items() if skill != "speed"),
            key=lambda item: item[1],
        )[:2]
        weak_labels = [SKILL_LABELS.get(skill, skill) for skill, _score in weakest]
        summary = (
            f"最近 {len(payloads)} 次 {QUESTION_TYPE_LABELS[question_type]} 中，"
            f"薄弱点更集中在：{'、'.join(weak_labels) if weak_labels else '综合理解'}。"
        )

        self.db.save_weakness_snapshot(level, question_type, summary, dimensions, len(payloads))
        return {
            "summary": summary,
            "based_on_attempt_count": len(payloads),
            "dimensions": dimensions,
        }

