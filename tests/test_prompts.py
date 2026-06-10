import unittest

from app.ai.pipelines import QuestionGenerationPipeline
from app.domain.enums import Level, QuestionType


class PromptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = QuestionGenerationPipeline(client=None, default_model="deepseek-v4-pro")

    def test_generate_requires_real_client_instead_of_silent_mock(self) -> None:
        with self.assertRaises(RuntimeError):
            self.pipeline.generate(Level.CET4, QuestionType.BANKED_CLOZE)

    def test_question_spec_banked_cloze_matches_cet_requirements(self) -> None:
        spec = self.pipeline._question_spec(Level.CET4, QuestionType.BANKED_CLOZE, None)
        self.assertIn("CET4 200-250 words", spec)
        self.assertIn("10 blanks", spec)
        self.assertIn("15 shared options", spec)
        self.assertIn("[1]...[10]", spec)
        self.assertIn("one English word only", spec)

    def test_question_spec_long_reading_matches_cet_requirements(self) -> None:
        spec = self.pipeline._question_spec(Level.CET6, QuestionType.LONG_READING, None)
        self.assertIn("CET6 1050-1350 words", spec)
        self.assertIn("10 statements", spec)
        self.assertIn("Paragraphs must be labeled", spec)

    def test_generation_prompt_includes_exam_constraints(self) -> None:
        prompt = self.pipeline._generation_user_prompt(
            Level.CET4,
            QuestionType.CAREFUL_READING,
            1,
            {
                "topic": "education",
                "genre": "expository",
                "register": "exam",
                "target_word_count": 320,
                "skill_focus": ["main idea", "detail"],
                "difficulty_controls": ["familiar topic"],
                "structure_plan": ["5 questions"],
                "option_strategy": ["balanced distractors"],
            },
        )
        self.assertIn("Mandatory spec", prompt)
        self.assertIn("5 four-option multiple-choice questions", prompt)
        self.assertIn("Ensure every answer is uniquely supported", prompt)
        self.assertIn("Use concise Chinese explanations", prompt)
        self.assertIn('"options": ["A. ...", "B. ...", "C. ...", "D. ..."]', prompt)
        self.assertIn("All of the above", prompt)

    def test_careful_reading_spec_mentions_core_question_types(self) -> None:
        spec = self.pipeline._question_spec(Level.CET6, QuestionType.CAREFUL_READING, 2)
        self.assertIn("CET6 390-470 words", spec)
        self.assertIn("exactly 5 four-option multiple-choice questions", spec)
        self.assertIn("main idea, detail, inference, vocabulary in context, and attitude/tone", spec)

    def test_long_reading_generation_prompt_mentions_paraphrase_matching(self) -> None:
        prompt = self.pipeline._generation_user_prompt(
            Level.CET6,
            QuestionType.LONG_READING,
            None,
            {
                "topic": "technology and education",
                "genre": "expository",
                "register": "exam",
                "target_word_count": 1180,
                "skill_focus": ["matching", "scanning"],
                "difficulty_controls": ["mild academic tone"],
                "structure_plan": ["10 matching items"],
                "option_strategy": ["paraphrase statements"],
            },
        )
        self.assertIn("10 statements for paragraph matching", prompt)
        self.assertIn("statements must be paraphrases", prompt)
        self.assertIn("Paragraphs must be labeled", prompt)
        self.assertIn('"shared_options": []', prompt)
        self.assertIn('"prompt": "English statement to match"', prompt)
        self.assertIn("statement rather than a question", prompt)

    def test_generation_prompt_mentions_cet_source_style_and_background_limits(self) -> None:
        prompt = self.pipeline._generation_user_prompt(
            Level.CET4,
            QuestionType.BANKED_CLOZE,
            None,
            {
                "topic": "public health",
                "genre": "expository",
                "register": "exam",
                "target_word_count": 230,
                "skill_focus": ["vocabulary", "logic"],
                "difficulty_controls": ["familiar topic"],
                "structure_plan": ["10 blanks"],
                "option_strategy": ["same part of speech distractors"],
            },
        )
        self.assertIn("source materials", prompt)
        self.assertIn("Background knowledge must be common or recoverable", prompt)
        self.assertIn("Do not write in a dramatic, fictional, or conversational blog style", prompt)
        self.assertIn('"prompt": "Blank 1"', prompt)
        self.assertIn("one English word", prompt)
        self.assertIn("Do not place Chinese characters", prompt)

    def test_blueprint_builds_target_counts_and_skill_focus(self) -> None:
        blueprint = self.pipeline._build_blueprint(
            Level.CET6,
            QuestionType.CAREFUL_READING,
            2,
            "推理判断不稳定",
        )
        self.assertEqual(blueprint["target_word_count"], 430)
        self.assertEqual(blueprint["vocabulary_target_count"], 5)
        self.assertIn("inference", blueprint["skill_focus"])
        self.assertIn("推理判断不稳定", blueprint["weakness_focus"])

    def test_postprocess_normalizes_careful_reading_options_and_skill_tags(self) -> None:
        payload = {
            "title": "Demo",
            "topic": "education",
            "passage": {"title": "Demo", "paragraphs": ["Paragraph one.", "Paragraph two."]},
            "questions": [
                {
                    "id": "wrong-id",
                    "prompt": "The word \"critical\" most likely means",
                    "options": ["A) important", "B) easy", "C) large", "D) modern"],
                    "skill_tag": "vocabulary",
                    "hint": None,
                }
            ],
            "answer_key": ["A."],
            "analysis": {"item_explanations": [{"question_id": "bad", "correct_answer": "A.", "skill_tag": "tone"}], "test_tips": ["tip1"]},
            "vocabulary": [],
        }
        normalized = self.pipeline._postprocess_payload(
            payload,
            Level.CET4,
            QuestionType.CAREFUL_READING,
            1,
        )
        self.assertEqual(normalized["questions"][0]["id"], "q1")
        self.assertEqual(normalized["questions"][0]["options"][0], "A. important")
        self.assertEqual(normalized["questions"][0]["skill_tag"], "vocabulary_in_context")
        self.assertEqual(normalized["analysis"]["item_explanations"][0]["question_id"], "q1")
        self.assertEqual(len(normalized["analysis"]["test_tips"]), 3)

    def test_postprocess_relabels_long_reading_paragraphs(self) -> None:
        payload = {
            "title": "Demo",
            "topic": "technology",
            "passage": {
                "title": "Demo",
                "paragraphs": ["A) first paragraph", "second paragraph without label"],
            },
            "questions": [],
            "answer_key": [],
            "analysis": {"item_explanations": [], "test_tips": []},
            "vocabulary": [],
        }
        normalized = self.pipeline._postprocess_payload(
            payload,
            Level.CET4,
            QuestionType.LONG_READING,
            None,
        )
        self.assertEqual(normalized["passage"]["paragraphs"][0], "A. first paragraph")
        self.assertEqual(normalized["passage"]["paragraphs"][1], "B. second paragraph without label")


if __name__ == "__main__":
    unittest.main()
