from __future__ import annotations

from app.constants import LEVEL_LABELS, QUESTION_TYPE_LABELS, QUESTION_TYPE_SHORT_LABELS
from app.data.db import Database
from app.domain.enums import Level, QuestionType


class StatsService:
    def __init__(self, db: Database) -> None:
        self.db = db

    @staticmethod
    def seconds_to_text(seconds: int) -> str:
        minutes, remaining = divmod(int(seconds), 60)
        return f"{minutes:02d}:{remaining:02d}"

    def overview_text(self) -> str:
        stats = self.db.overview_stats()
        common_type = (
            QUESTION_TYPE_LABELS[QuestionType(stats["most_common_type"])]
            if stats["most_common_type"]
            else "暂无"
        )
        updated = stats["latest_weakness_updated_at"] or "暂无"
        performance = f"{stats['recent_performance_percent']:.1f}"
        raw_accuracy = f"{stats['recent_accuracy'] * 100:.1f}%"
        pace = f"{stats['recent_pace_percent']:.1f}"
        return (
            f"总刷题数：{stats['total_attempts']}\n"
            f"四级题数：{stats['total_cet4']}    六级题数：{stats['total_cet6']}\n"
            f"最近 5 次表现指数：{performance}\n"
            f"最近 5 次节奏匹配：{pace}\n"
            f"原始均值：正确率 {raw_accuracy} / 用时 {self.seconds_to_text(stats['recent_duration_seconds'])}\n"
            f"当前最常练题型：{common_type}\n"
            f"最近一次薄弱项更新时间：{updated}"
        )

    def overview_data(self) -> dict:
        stats = self.db.overview_stats()
        common_type = (
            QUESTION_TYPE_LABELS[QuestionType(stats["most_common_type"])]
            if stats["most_common_type"]
            else "暂无"
        )
        return {
            **stats,
            "recent_accuracy_percent": round(stats["recent_accuracy"] * 100, 1),
            "recent_duration_text": self.seconds_to_text(stats["recent_duration_seconds"]),
            "recent_performance_percent": round(stats["recent_performance_percent"], 1),
            "recent_pace_percent": round(stats["recent_pace_percent"], 1),
            "raw_recent_accuracy_percent": round(stats["recent_accuracy"] * 100, 1),
            "raw_recent_duration_text": self.seconds_to_text(stats["recent_duration_seconds"]),
            "most_common_type_label": common_type,
            "cet4_ratio": stats["total_cet4"] / stats["total_attempts"]
            if stats["total_attempts"]
            else 0.0,
            "cet6_ratio": stats["total_cet6"] / stats["total_attempts"]
            if stats["total_attempts"]
            else 0.0,
        }

    def level_type_summary_text(self, level: Level) -> str:
        stats = self.db.type_stats_for_level(level)
        lines = [f"{LEVEL_LABELS[level]}最近数据："]
        keys = [
            ("banked_cloze", "选词填空"),
            ("long_reading", "长篇阅读"),
            ("careful_reading:1", "仔细阅读 1"),
            ("careful_reading:2", "仔细阅读 2"),
            ("writing", "写作"),
            ("translation", "翻译"),
        ]
        for key, label in keys:
            item = stats.get(key)
            if not item:
                lines.append(f"- {label}：暂无记录")
                continue
            lines.append(
                f"- {label}：{item['attempt_count']} 次，最近 5 次正确率 {item['recent_accuracy'] * 100:.1f}% ，"
                f"平均时间 {self.seconds_to_text(item['recent_duration_seconds'])}"
            )
        return "\n".join(lines)

    def level_type_data(self, level: Level) -> list[dict]:
        stats = self.db.type_stats_for_level(level)
        keys = [
            ("banked_cloze", "选词填空"),
            ("long_reading", "长篇阅读"),
            ("careful_reading:1", "仔细阅读 1"),
            ("careful_reading:2", "仔细阅读 2"),
            ("writing", "写作"),
            ("translation", "翻译"),
        ]
        cards = []
        for key, label in keys:
            item = stats.get(key, {})
            cards.append(
                {
                    "key": key,
                    "short_label": QUESTION_TYPE_SHORT_LABELS.get(key, label),
                    "label": label,
                    "attempt_count": item.get("attempt_count", 0),
                    "recent_accuracy_percent": round(item.get("recent_accuracy", 0.0) * 100, 1),
                    "recent_duration_seconds": item.get("recent_duration_seconds", 0),
                    "recent_duration_text": self.seconds_to_text(
                        item.get("recent_duration_seconds", 0)
                    ),
                    "best_recent_accuracy_percent": round(
                        item.get("best_recent_accuracy", 0.0) * 100, 1
                    ),
                    "recent_accuracy_series": item.get("recent_accuracy_series", []),
                    "recent_duration_series": item.get("recent_duration_series", []),
                }
            )
        return cards

    def list_history(self, limit: int = 20) -> list[dict]:
        return self.db.list_history(limit=limit)

    def list_vocabulary(self, limit: int = 100) -> list[dict]:
        return self.db.list_vocabulary(limit=limit)

    def list_weakness_snapshots(self, limit: int = 20) -> list[dict]:
        return self.db.list_weakness_snapshots(limit=limit)

    def list_mock_exam_history(self, limit: int = 20) -> list[dict]:
        return self.db.list_mock_exam_history(limit=limit)

    def list_mock_exam_weakness_snapshots(self, limit: int = 20) -> list[dict]:
        return self.db.list_mock_exam_weakness_snapshots(limit=limit)
