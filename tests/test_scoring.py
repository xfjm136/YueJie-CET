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


if __name__ == "__main__":
    unittest.main()
