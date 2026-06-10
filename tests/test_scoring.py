import unittest

from app.domain.enums import Level, QuestionType
from app.domain.scoring import grade_attempt
from app.domain.schemas import AnalysisReport, ItemExplanation, Passage, Question, QuestionSet


def build_question_set() -> QuestionSet:
    return QuestionSet(
        id="qs_demo",
        level=Level.CET4,
        question_type=QuestionType.CAREFUL_READING,
        slot=1,
        title="Demo",
        topic="demo",
        passage=Passage(title="Demo", paragraphs=["Paragraph one."]),
        questions=[
            Question(id="q1", prompt="Question 1", options=["A", "B"], skill_tag="detail"),
            Question(id="q2", prompt="Question 2", options=["A", "B"], skill_tag="main_idea"),
        ],
        answer_key=["A", "B"],
        analysis=AnalysisReport(
            overall_strategy="demo",
            overall_summary="demo",
            item_explanations=[
                ItemExplanation("q1", "A", "exp1", "detail"),
                ItemExplanation("q2", "B", "exp2", "main_idea"),
            ],
            test_tips=[],
        ),
        vocabulary=[],
    )


class ScoringTests(unittest.TestCase):
    def test_grade_attempt_counts_accuracy(self) -> None:
        result = grade_attempt(build_question_set(), {"q1": "A", "q2": "A"}, 120)
        self.assertEqual(result.correct_count, 1)
        self.assertEqual(result.total_count, 2)
        self.assertEqual(result.accuracy, 0.5)
        self.assertEqual(len(result.question_results), 2)

    def test_subjective_types_require_ai_evaluation_pipeline(self) -> None:
        question_set = QuestionSet(
            id="qs_writing",
            level=Level.CET4,
            question_type=QuestionType.WRITING,
            title="Writing Task",
            topic="writing",
            passage=Passage(title="Task", paragraphs=["Task line one."]),
            questions=[],
            answer_key=[],
            analysis=AnalysisReport(
                overall_strategy="demo",
                overall_summary="demo",
                item_explanations=[],
                test_tips=[],
            ),
            vocabulary=[],
            task_prompt="Write an essay.",
            reference_answer="A sample essay.",
            rubric_focus=["content_relevance", "grammar"],
            min_response_words=120,
            max_response_words=180,
        )
        self.assertEqual(question_set.layout_mode, "two")
        self.assertTrue(question_set.is_subjective)
        with self.assertRaises(ValueError):
            grade_attempt(question_set, {"response_text": "demo"}, 600)


if __name__ == "__main__":
    unittest.main()
