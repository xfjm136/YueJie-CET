import unittest
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
    VocabularyItem,
)


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


class DatabaseTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
