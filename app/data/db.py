from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.domain.enums import Level, QuestionType
from app.domain.schemas import AttemptResult, QuestionSet, VocabularyItem


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def managed_connection(self):
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_schema(self) -> None:
        with self.managed_connection() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;

                CREATE TABLE IF NOT EXISTS question_sets (
                    id TEXT PRIMARY KEY,
                    level TEXT NOT NULL,
                    question_type TEXT NOT NULL,
                    slot INTEGER,
                    title TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    word_count INTEGER NOT NULL,
                    generator_model TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS attempts (
                    id TEXT PRIMARY KEY,
                    question_set_id TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    submitted_at TEXT NOT NULL,
                    duration_seconds INTEGER NOT NULL,
                    correct_count INTEGER NOT NULL,
                    total_count INTEGER NOT NULL,
                    accuracy REAL NOT NULL,
                    answers_json TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    is_history_retry INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY(question_set_id) REFERENCES question_sets(id)
                );

                CREATE TABLE IF NOT EXISTS vocabulary_items (
                    lemma TEXT PRIMARY KEY,
                    surface_form TEXT NOT NULL,
                    level_hint TEXT NOT NULL,
                    meaning_zh TEXT NOT NULL,
                    example_en TEXT NOT NULL,
                    frequency_score INTEGER NOT NULL DEFAULT 0,
                    error_related_score INTEGER NOT NULL DEFAULT 0,
                    last_seen_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS weakness_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    level TEXT NOT NULL,
                    question_type TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    dimensions_json TEXT NOT NULL,
                    based_on_attempt_count INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS app_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

    def save_question_set(self, question_set: QuestionSet) -> None:
        with self.managed_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO question_sets (
                    id, level, question_type, slot, title, topic, word_count,
                    generator_model, source_type, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    question_set.id,
                    question_set.level.value,
                    question_set.question_type.value,
                    question_set.slot,
                    question_set.title,
                    question_set.topic,
                    question_set.word_count,
                    question_set.generator_model,
                    question_set.source_type,
                    json.dumps(question_set.to_dict(), ensure_ascii=False),
                    question_set.created_at.isoformat(),
                ),
            )

    def get_question_set(self, question_set_id: str) -> QuestionSet | None:
        with self.managed_connection() as conn:
            row = conn.execute(
                "SELECT payload_json FROM question_sets WHERE id = ?",
                (question_set_id,),
            ).fetchone()
        if not row:
            return None
        return QuestionSet.from_dict(json.loads(row["payload_json"]))

    def save_attempt(
        self,
        result: AttemptResult,
        answers: dict[str, str],
        started_at: datetime,
        submitted_at: datetime,
        is_history_retry: bool,
    ) -> None:
        with self.managed_connection() as conn:
            conn.execute(
                """
                INSERT INTO attempts (
                    id, question_set_id, started_at, submitted_at, duration_seconds,
                    correct_count, total_count, accuracy, answers_json, result_json,
                    is_history_retry
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.id,
                    result.question_set_id,
                    started_at.isoformat(),
                    submitted_at.isoformat(),
                    result.duration_seconds,
                    result.correct_count,
                    result.total_count,
                    result.accuracy,
                    json.dumps(answers, ensure_ascii=False),
                    json.dumps(result.to_dict(), ensure_ascii=False),
                    1 if is_history_retry else 0,
                ),
            )

    def delete_attempt_history(self, attempt_id: str) -> dict[str, Any] | None:
        with self.managed_connection() as conn:
            row = conn.execute(
                """
                SELECT a.id AS attempt_id, a.question_set_id, q.level, q.question_type, q.title, q.topic
                FROM attempts a
                JOIN question_sets q ON q.id = a.question_set_id
                WHERE a.id = ?
                """,
                (attempt_id,),
            ).fetchone()
            if not row:
                return None
            row_data = dict(row)

            question_set_id = row_data["question_set_id"]
            conn.execute("DELETE FROM attempts WHERE id = ?", (attempt_id,))
            remaining = conn.execute(
                "SELECT COUNT(*) AS total FROM attempts WHERE question_set_id = ?",
                (question_set_id,),
            ).fetchone()
            question_set_deleted = False
            if int(remaining["total"]) == 0:
                conn.execute("DELETE FROM question_sets WHERE id = ?", (question_set_id,))
                question_set_deleted = True

            self._rebuild_vocabulary_items_from_attempts(conn)
            return {
                "attempt_id": row_data["attempt_id"],
                "question_set_id": question_set_id,
                "question_set_deleted": question_set_deleted,
                "level": row_data["level"],
                "question_type": row_data["question_type"],
                "title": row_data["title"],
                "topic": row_data["topic"],
            }

    def list_history(self, limit: int = 20) -> list[dict[str, Any]]:
        with self.managed_connection() as conn:
            rows = conn.execute(
                """
                SELECT a.id AS attempt_id, a.submitted_at, a.duration_seconds, a.accuracy,
                       a.correct_count, a.total_count, a.is_history_retry,
                       q.id AS question_set_id, q.level, q.question_type, q.slot,
                       q.title, q.topic
                FROM attempts a
                JOIN question_sets q ON q.id = a.question_set_id
                ORDER BY a.submitted_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_attempt_bundle(self, attempt_id: str) -> dict[str, Any] | None:
        with self.managed_connection() as conn:
            row = conn.execute(
                """
                SELECT a.*, q.payload_json, q.level, q.question_type, q.slot, q.title, q.topic
                FROM attempts a
                JOIN question_sets q ON q.id = a.question_set_id
                WHERE a.id = ?
                """,
                (attempt_id,),
            ).fetchone()
        if not row:
            return None
        return {
            "attempt": dict(row),
            "question_set": QuestionSet.from_dict(json.loads(row["payload_json"])),
            "result": AttemptResult.from_dict(json.loads(row["result_json"])),
            "answers": json.loads(row["answers_json"]),
        }

    def upsert_vocabulary_items(self, items: list[VocabularyItem]) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        with self.managed_connection() as conn:
            for item in items:
                row = conn.execute(
                    "SELECT frequency_score, error_related_score FROM vocabulary_items WHERE lemma = ?",
                    (item.lemma,),
                ).fetchone()
                if row:
                    conn.execute(
                        """
                        UPDATE vocabulary_items
                        SET surface_form = ?, level_hint = ?, meaning_zh = ?, example_en = ?,
                            frequency_score = ?, error_related_score = ?, last_seen_at = ?
                        WHERE lemma = ?
                        """,
                        (
                            item.surface_form,
                            item.level_hint,
                            item.meaning_zh,
                            item.example_en,
                            int(row["frequency_score"]) + max(item.frequency_score, 1),
                            int(row["error_related_score"]) + item.error_related_score,
                            timestamp,
                            item.lemma,
                        ),
                    )
                else:
                    conn.execute(
                        """
                        INSERT INTO vocabulary_items (
                            lemma, surface_form, level_hint, meaning_zh, example_en,
                            frequency_score, error_related_score, last_seen_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            item.lemma,
                            item.surface_form,
                            item.level_hint,
                            item.meaning_zh,
                            item.example_en,
                            max(item.frequency_score, 1),
                            item.error_related_score,
                            timestamp,
                        ),
                    )

    def _rebuild_vocabulary_items_from_attempts(self, conn: sqlite3.Connection) -> None:
        conn.execute("DELETE FROM vocabulary_items")
        rows = conn.execute(
            """
            SELECT a.submitted_at, a.result_json, q.payload_json
            FROM attempts a
            JOIN question_sets q ON q.id = a.question_set_id
            ORDER BY a.submitted_at ASC
            """
        ).fetchall()
        aggregates: dict[str, dict[str, Any]] = {}
        for row in rows:
            question_set = QuestionSet.from_dict(json.loads(row["payload_json"]))
            result_json = json.loads(row["result_json"])
            seen_at = row["submitted_at"]
            for item in question_set.vocabulary:
                entry = aggregates.get(item.lemma)
                if entry is None:
                    aggregates[item.lemma] = {
                        "lemma": item.lemma,
                        "surface_form": item.surface_form,
                        "level_hint": item.level_hint,
                        "meaning_zh": item.meaning_zh,
                        "example_en": item.example_en,
                        "frequency_score": max(int(item.frequency_score), 1),
                        "error_related_score": int(item.error_related_score),
                        "last_seen_at": seen_at,
                    }
                else:
                    entry["surface_form"] = item.surface_form
                    entry["level_hint"] = item.level_hint
                    entry["meaning_zh"] = item.meaning_zh
                    entry["example_en"] = item.example_en
                    entry["frequency_score"] += max(int(item.frequency_score), 1)
                    entry["error_related_score"] += int(item.error_related_score)
                    entry["last_seen_at"] = seen_at
            subjective = result_json.get("subjective_evaluation") or {}
            for item in subjective.get("wrong_words", []):
                corrected = str(item.get("corrected", "")).strip()
                if not corrected:
                    continue
                lemma = corrected.lower()
                entry = aggregates.get(lemma)
                if entry is None:
                    aggregates[lemma] = {
                        "lemma": lemma,
                        "surface_form": corrected,
                        "level_hint": question_set.level.value,
                        "meaning_zh": str(item.get("meaning_zh", "")).strip(),
                        "example_en": corrected,
                        "frequency_score": 1,
                        "error_related_score": 1,
                        "last_seen_at": seen_at,
                    }
                else:
                    entry["surface_form"] = corrected
                    if not entry["meaning_zh"]:
                        entry["meaning_zh"] = str(item.get("meaning_zh", "")).strip()
                    entry["error_related_score"] += 1
                    entry["frequency_score"] += 1
                    entry["last_seen_at"] = seen_at

        for item in aggregates.values():
            conn.execute(
                """
                INSERT INTO vocabulary_items (
                    lemma, surface_form, level_hint, meaning_zh, example_en,
                    frequency_score, error_related_score, last_seen_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item["lemma"],
                    item["surface_form"],
                    item["level_hint"],
                    item["meaning_zh"],
                    item["example_en"],
                    item["frequency_score"],
                    item["error_related_score"],
                    item["last_seen_at"],
                ),
            )

    def list_vocabulary(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.managed_connection() as conn:
            rows = conn.execute(
                """
                SELECT lemma, surface_form, level_hint, meaning_zh, example_en,
                       frequency_score, error_related_score, last_seen_at
                FROM vocabulary_items
                ORDER BY frequency_score DESC, last_seen_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def save_weakness_snapshot(
        self,
        level: Level,
        question_type: QuestionType,
        summary: str,
        dimensions: dict[str, float],
        based_on_attempt_count: int,
    ) -> None:
        with self.managed_connection() as conn:
            conn.execute(
                """
                INSERT INTO weakness_snapshots (
                    level, question_type, summary, dimensions_json,
                    based_on_attempt_count, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    level.value,
                    question_type.value,
                    summary,
                    json.dumps(dimensions, ensure_ascii=False),
                    based_on_attempt_count,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def clear_weakness_snapshots(self) -> None:
        with self.managed_connection() as conn:
            conn.execute("DELETE FROM weakness_snapshots")

    def attempt_level_type_pairs(self) -> list[tuple[Level, QuestionType]]:
        with self.managed_connection() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT q.level, q.question_type
                FROM attempts a
                JOIN question_sets q ON q.id = a.question_set_id
                ORDER BY q.level, q.question_type
                """
            ).fetchall()
        return [(Level(row["level"]), QuestionType(row["question_type"])) for row in rows]

    def list_weakness_snapshots(self, limit: int = 20) -> list[dict[str, Any]]:
        with self.managed_connection() as conn:
            rows = conn.execute(
                """
                SELECT id, level, question_type, summary, dimensions_json,
                       based_on_attempt_count, updated_at
                FROM weakness_snapshots
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def latest_weakness_updated_at(self) -> str | None:
        with self.managed_connection() as conn:
            row = conn.execute(
                "SELECT updated_at FROM weakness_snapshots ORDER BY updated_at DESC LIMIT 1"
            ).fetchone()
        return row["updated_at"] if row else None

    def latest_weakness_summary(
        self,
        level: Level,
        question_type: QuestionType,
    ) -> str | None:
        with self.managed_connection() as conn:
            row = conn.execute(
                """
                SELECT summary
                FROM weakness_snapshots
                WHERE level = ? AND question_type = ?
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (level.value, question_type.value),
            ).fetchone()
        return row["summary"] if row else None

    def get_app_setting(self, key: str, default: str | None = None) -> str | None:
        with self.managed_connection() as conn:
            row = conn.execute(
                "SELECT value FROM app_settings WHERE key = ?",
                (key,),
            ).fetchone()
        return row["value"] if row else default

    def set_app_setting(self, key: str, value: str) -> None:
        with self.managed_connection() as conn:
            conn.execute(
                """
                INSERT INTO app_settings (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (key, value, datetime.now(timezone.utc).isoformat()),
            )

    def overview_stats(self) -> dict[str, Any]:
        history = self.list_history(limit=200)
        total_attempts = len(history)
        total_cet4 = sum(1 for row in history if row["level"] == Level.CET4.value)
        total_cet6 = sum(1 for row in history if row["level"] == Level.CET6.value)
        recent = history[:5]
        avg_accuracy = round(sum(row["accuracy"] for row in recent) / len(recent), 4) if recent else 0.0
        avg_duration = int(sum(row["duration_seconds"] for row in recent) / len(recent)) if recent else 0
        type_counter = Counter(row["question_type"] for row in history)
        most_common_type = type_counter.most_common(1)[0][0] if type_counter else None
        accuracy_series = [round(row["accuracy"] * 100, 2) for row in reversed(recent)]
        duration_series = [int(row["duration_seconds"]) for row in reversed(recent)]
        return {
            "total_attempts": total_attempts,
            "total_cet4": total_cet4,
            "total_cet6": total_cet6,
            "recent_accuracy": avg_accuracy,
            "recent_duration_seconds": avg_duration,
            "recent_accuracy_series": accuracy_series,
            "recent_duration_series": duration_series,
            "most_common_type": most_common_type,
            "latest_weakness_updated_at": self.latest_weakness_updated_at(),
        }

    def type_stats_for_level(self, level: Level) -> dict[str, dict[str, Any]]:
        history = [row for row in self.list_history(limit=200) if row["level"] == level.value]
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in history:
            key = f"{row['question_type']}:{row['slot']}" if row["slot"] else row["question_type"]
            grouped[key].append(row)
        stats: dict[str, dict[str, Any]] = {}
        for key, rows in grouped.items():
            recent = rows[:5]
            stats[key] = {
                "attempt_count": len(rows),
                "recent_accuracy": round(
                    sum(item["accuracy"] for item in recent) / len(recent), 4
                )
                if recent
                else 0.0,
                "recent_duration_seconds": int(
                    sum(item["duration_seconds"] for item in recent) / len(recent)
                )
                if recent
                else 0,
                "best_recent_accuracy": max((item["accuracy"] for item in recent), default=0.0),
                "recent_accuracy_series": [
                    round(item["accuracy"] * 100, 2) for item in reversed(recent)
                ],
                "recent_duration_series": [
                    int(item["duration_seconds"]) for item in reversed(recent)
                ],
            }
        return stats

    def recent_attempt_payloads(
        self,
        level: Level,
        question_type: QuestionType,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        with self.managed_connection() as conn:
            rows = conn.execute(
                """
                SELECT a.result_json, a.duration_seconds
                FROM attempts a
                JOIN question_sets q ON q.id = a.question_set_id
                WHERE q.level = ? AND q.question_type = ?
                ORDER BY a.submitted_at DESC
                LIMIT ?
                """,
                (level.value, question_type.value, limit),
            ).fetchall()
        return [dict(row) for row in rows]
