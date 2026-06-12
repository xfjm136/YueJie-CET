from __future__ import annotations

import json
from collections import Counter

from app.constants import QUESTION_TYPE_LABELS, SKILL_LABELS
from app.data.db import Database
from app.domain.enums import Level, QuestionType
from app.domain.schemas import MockExamRecord


class WeaknessService:
    def __init__(self, db: Database) -> None:
        self.db = db

    def rebuild_snapshots(self) -> None:
        self.db.clear_weakness_snapshots()
        for level, question_type in self.db.attempt_level_type_pairs():
            self.refresh_snapshot(level, question_type)
        self.db.clear_mock_exam_weakness_snapshots()
        for level in self.db.mock_exam_levels():
            self.refresh_mock_exam_snapshot(level)

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
            subjective = result.get("subjective_evaluation") or {}
            for skill in subjective.get("weakness_tags", []):
                total_by_skill[skill] += 1
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

    def refresh_mock_exam_snapshot(
        self,
        level: Level,
    ) -> dict[str, str | int | dict[str, float]] | None:
        payloads = self.db.recent_mock_exam_payloads(level, limit=10)
        if len(payloads) < 2:
            return None

        total_by_skill: Counter[str] = Counter()
        wrong_by_skill: Counter[str] = Counter()
        total_duration = 0

        for row in payloads:
            total_duration += int(row["duration_seconds"])
            record = MockExamRecord.from_dict(json.loads(row["payload_json"]))
            for section in record.sections:
                for item in section.result.question_results:
                    skill = item.skill_tag or "general"
                    total_by_skill[skill] += 1
                    if not item.is_correct:
                        wrong_by_skill[skill] += 1
                subjective = section.result.subjective_evaluation
                if subjective is None:
                    continue
                for skill in subjective.weakness_tags:
                    total_by_skill[skill] += 1
                    wrong_by_skill[skill] += 1

        dimensions = {
            skill: round(100 * (1 - wrong_by_skill[skill] / total), 2)
            for skill, total in total_by_skill.items()
            if total > 0
        }
        avg_duration = total_duration / len(payloads)
        dimensions["speed"] = 100.0 if avg_duration <= 100 * 60 else max(
            30.0,
            100.0 - (avg_duration - 100 * 60) / 30,
        )

        weakest = sorted(
            ((skill, score) for skill, score in dimensions.items() if skill != "speed"),
            key=lambda item: item[1],
        )[:3]
        weak_labels = [SKILL_LABELS.get(skill, skill) for skill, _score in weakest]
        summary = (
            f"最近 {len(payloads)} 次 {level.value.upper()} 模拟四六级考试中，"
            f"薄弱点更集中在：{'、'.join(weak_labels) if weak_labels else '综合能力'}。"
        )
        self.db.save_mock_exam_weakness_snapshot(level, summary, dimensions, len(payloads))
        return {
            "summary": summary,
            "based_on_exam_count": len(payloads),
            "dimensions": dimensions,
        }
