import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from app.data.db import Database
from app.domain.enums import Level, QuestionType
from app.domain.schemas import (
    AnalysisReport,
    ItemExplanation,
    Passage,
    Question,
    QuestionSet,
    ScoreDimension,
    SentenceRewrite,
    SubjectiveEvaluation,
    VocabularyItem,
    WordCorrection,
)
from app.services.attempt_service import AttemptService
from app.services.weakness_service import WeaknessService


def build_question_set() -> QuestionSet:
    return QuestionSet(
        id="qs_demo",
        level=Level.CET4,
        question_type=QuestionType.BANKED_CLOZE,
        title="Demo",
        topic="demo",
        passage=Passage(title="Demo", paragraphs=["Paragraph one."]),
        questions=[Question(id="q1", prompt="Question 1", skill_tag="vocabulary")],
        answer_key=["A"],
        analysis=AnalysisReport(
            overall_strategy="demo",
            overall_summary="demo",
            item_explanations=[ItemExplanation("q1", "A", "exp1", "vocabulary")],
            test_tips=[],
        ),
        vocabulary=[VocabularyItem("demo", "demo", "cet4", "示例")],
        shared_options=["A. demo", "B. test"],
    )


def build_subjective_question_set(question_type: QuestionType = QuestionType.WRITING) -> QuestionSet:
    passage_lines = (
        [
            "In recent years, digital tools have become part of campus life.",
            "Write an essay on balancing efficiency and independent thinking.",
        ]
        if question_type is QuestionType.WRITING
        else [
            "如今，越来越多的大学生开始使用人工智能工具辅助学习，但真正的进步仍然离不开独立思考、持续练习与自我反思。",
            "请将上文翻译成英文。",
        ]
    )
    return QuestionSet(
        id=f"qs_{question_type.value}",
        level=Level.CET4,
        question_type=question_type,
        title="Subjective Demo",
        topic="ai and learning",
        passage=Passage(title="Subjective Demo", paragraphs=passage_lines),
        questions=[],
        answer_key=[],
        analysis=AnalysisReport(
            overall_strategy="demo",
            overall_summary="demo",
            item_explanations=[],
            test_tips=[],
        ),
        vocabulary=[VocabularyItem("reflection", "reflection", "cet4", "反思")],
        task_prompt=(
            "Write an essay in no less than 120 words."
            if question_type is QuestionType.WRITING
            else "Translate the Chinese passage into English."
        ),
        reference_answer="Sample reference answer.",
        rubric_focus=(
            ["content_relevance", "coherence", "grammar", "lexical_accuracy"]
            if question_type is QuestionType.WRITING
            else ["translation_accuracy", "translation_fluency", "grammar", "lexical_accuracy"]
        ),
        min_response_words=120,
        max_response_words=180 if question_type is QuestionType.WRITING else 220,
    )


class FakeSubjectiveEvaluator:
    def evaluate(
        self,
        question_set: QuestionSet,
        response_text: str,
        duration_seconds: int,
    ) -> SubjectiveEvaluation:
        return SubjectiveEvaluation(
            score_15=10.5,
            estimated_reported_score=78.0,
            grade_band="中等",
            overall_feedback_zh=f"{question_set.question_type.value} 评阅已完成",
            score_dimensions=[
                ScoreDimension("grammar", 2.5, 4.0, "语法仍需加强。"),
                ScoreDimension("lexical_accuracy", 2.5, 3.0, "词汇准确性一般。"),
                ScoreDimension(
                    "content_relevance"
                    if question_set.question_type is QuestionType.WRITING
                    else "translation_accuracy",
                    3.0,
                    4.0,
                    "内容基本达标。",
                ),
                ScoreDimension(
                    "coherence"
                    if question_set.question_type is QuestionType.WRITING
                    else "translation_fluency",
                    2.5,
                    4.0,
                    "连贯性仍可提升。",
                ),
            ],
            wrong_words=[
                WordCorrection(
                    original="useing",
                    corrected="using",
                    reason_zh="拼写错误。",
                    meaning_zh="使用",
                    skill_tag="lexical_accuracy",
                )
            ],
            sentence_rewrites=[
                SentenceRewrite(
                    original_sentence="This make students lazy.",
                    revised_sentence="This makes some students overly dependent on shortcuts.",
                    reason_zh="主谓一致错误。",
                    skill_tag="grammar",
                )
            ],
            high_score_version="A polished high-score version.",
            weakness_tags=["grammar", "lexical_accuracy"],
        )


class DatabaseTests(unittest.TestCase):
    def test_overview_stats_include_normalized_home_metrics(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            db = Database(Path(tmp_dir) / "test.db")
            db.init_schema()

            banked = build_question_set()
            long_reading = build_question_set()
            long_reading.id = "qs_long"
            long_reading.question_type = QuestionType.LONG_READING
            long_reading.title = "Long Demo"
            long_reading.topic = "long demo"

            db.save_question_set(banked)
            db.save_question_set(long_reading)

            attempt_service = AttemptService(db, WeaknessService(db))
            started_at = datetime.now(timezone.utc) - timedelta(minutes=40)
            attempt_service.submit_attempt(
                question_set=banked,
                answers={"q1": "A"},
                started_at=started_at,
                is_history_retry=False,
            )
            attempt_service.submit_attempt(
                question_set=long_reading,
                answers={"q1": "A"},
                started_at=started_at + timedelta(minutes=5),
                is_history_retry=False,
            )

            overview = db.overview_stats()

            self.assertIn("recent_performance_percent", overview)
            self.assertIn("recent_pace_percent", overview)
            self.assertEqual(len(overview["recent_performance_series"]), 2)
            self.assertEqual(len(overview["recent_pace_series"]), 2)
            self.assertLessEqual(max(overview["recent_pace_series"]), 100.0)
            self.assertNotEqual(overview["recent_pace_series"][0], overview["recent_duration_series"][0])

    def test_database_round_trip(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            db = Database(Path(tmp_dir) / "test.db")
            db.init_schema()
            question_set = build_question_set()
            db.save_question_set(question_set)
            db.upsert_vocabulary_items(question_set.vocabulary)

            loaded = db.get_question_set(question_set.id)
            vocab = db.list_vocabulary(limit=10)

            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.id, question_set.id)
            self.assertEqual(loaded.answer_key, ["A"])
            self.assertEqual(vocab[0]["lemma"], "demo")

    def test_app_settings_round_trip(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            db = Database(Path(tmp_dir) / "test.db")
            db.init_schema()
            db.set_app_setting("theme_mode", "dark")
            db.set_app_setting("background_mode", "opaque")

            self.assertEqual(db.get_app_setting("theme_mode"), "dark")
            self.assertEqual(db.get_app_setting("background_mode"), "opaque")

    def test_delete_attempt_history_rebuilds_history_and_vocabulary(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            db = Database(Path(tmp_dir) / "test.db")
            db.init_schema()
            question_set = build_question_set()
            db.save_question_set(question_set)
            db.upsert_vocabulary_items(question_set.vocabulary)

            attempt_service = AttemptService(db, WeaknessService(db))
            started_at = datetime.now(timezone.utc) - timedelta(minutes=5)
            result = attempt_service.submit_attempt(
                question_set=question_set,
                answers={"q1": "A"},
                started_at=started_at,
                is_history_retry=False,
            )

            deleted = attempt_service.delete_attempt_history(result.id)

            self.assertEqual(deleted["attempt_id"], result.id)
            self.assertTrue(deleted["question_set_deleted"])
            self.assertEqual(db.list_history(limit=10), [])
            self.assertEqual(db.list_vocabulary(limit=10), [])
            self.assertIsNone(db.get_question_set(question_set.id))

    def test_delete_attempt_history_rebuilds_remaining_vocabulary_without_missing_keys(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            db = Database(Path(tmp_dir) / "test.db")
            db.init_schema()

            first = build_question_set()
            second = build_subjective_question_set(QuestionType.WRITING)
            second.id = "qs_subjective_keep"
            second.vocabulary = [
                VocabularyItem("balance", "balance", "cet4", "平衡"),
            ]

            db.save_question_set(first)
            db.save_question_set(second)

            attempt_service = AttemptService(
                db,
                WeaknessService(db),
                subjective_evaluator=FakeSubjectiveEvaluator(),
            )
            started_at = datetime.now(timezone.utc) - timedelta(minutes=15)
            deleted_attempt = attempt_service.submit_attempt(
                question_set=first,
                answers={"q1": "A"},
                started_at=started_at,
                is_history_retry=False,
            )
            kept_attempt = attempt_service.submit_attempt(
                question_set=second,
                answers={"response_text": "A subjective answer."},
                started_at=started_at + timedelta(minutes=5),
                is_history_retry=False,
            )

            deleted = attempt_service.delete_attempt_history(deleted_attempt.id)
            history = db.list_history(limit=10)
            vocab = db.list_vocabulary(limit=20)
            vocab_by_lemma = {item["lemma"]: item for item in vocab}

            self.assertEqual(deleted["attempt_id"], deleted_attempt.id)
            self.assertTrue(deleted["question_set_deleted"])
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0]["attempt_id"], kept_attempt.id)
            self.assertIn("balance", vocab_by_lemma)
            self.assertIn("using", vocab_by_lemma)

    def test_subjective_attempt_adds_wrong_words_to_vocabulary(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            db = Database(Path(tmp_dir) / "test.db")
            db.init_schema()
            question_set = build_subjective_question_set(QuestionType.WRITING)
            db.save_question_set(question_set)

            attempt_service = AttemptService(
                db,
                WeaknessService(db),
                subjective_evaluator=FakeSubjectiveEvaluator(),
            )
            started_at = datetime.now(timezone.utc) - timedelta(minutes=12)
            result = attempt_service.submit_attempt(
                question_set=question_set,
                answers={"response_text": "A short essay draft."},
                started_at=started_at,
                is_history_retry=False,
            )

            vocab = db.list_vocabulary(limit=20)
            vocab_by_lemma = {item["lemma"]: item for item in vocab}

            self.assertIsNotNone(result.subjective_evaluation)
            self.assertEqual(result.question_results[0]["question_id"] if isinstance(result.question_results[0], dict) else result.question_results[0].question_id, "subjective-1")
            self.assertIn("reflection", vocab_by_lemma)
            self.assertIn("using", vocab_by_lemma)
            self.assertEqual(vocab_by_lemma["using"]["meaning_zh"], "使用")
            self.assertGreaterEqual(vocab_by_lemma["using"]["error_related_score"], 1)

    def test_subjective_attempts_build_weakness_snapshot(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            db = Database(Path(tmp_dir) / "test.db")
            db.init_schema()
            question_set = build_subjective_question_set(QuestionType.TRANSLATION)
            db.save_question_set(question_set)

            attempt_service = AttemptService(
                db,
                WeaknessService(db),
                subjective_evaluator=FakeSubjectiveEvaluator(),
            )
            for offset in range(5):
                started_at = datetime.now(timezone.utc) - timedelta(minutes=10 + offset)
                attempt_service.submit_attempt(
                    question_set=question_set,
                    answers={"response_text": "A translated response."},
                    started_at=started_at,
                    is_history_retry=False,
                )

            snapshots = db.list_weakness_snapshots(limit=10)

            self.assertTrue(snapshots)
            latest = snapshots[0]
            self.assertEqual(latest["question_type"], QuestionType.TRANSLATION.value)
            self.assertEqual(latest["based_on_attempt_count"], 5)
            self.assertIn("grammar", latest["dimensions_json"])
            self.assertIn("翻译", latest["summary"])


if __name__ == "__main__":
    unittest.main()
