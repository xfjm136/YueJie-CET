import unittest

from app.ai.pipelines import QuestionGenerationPipeline
from app.domain.enums import Level, QuestionType


class PromptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = QuestionGenerationPipeline(client=None, default_model="deepseek-v4-flash")

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
        self.assertIn("Structured output contract", prompt)
        self.assertIn("objective sets", prompt)
        self.assertIn("All of the above", prompt)

    def test_careful_reading_spec_mentions_core_question_types(self) -> None:
        spec = self.pipeline._question_spec(Level.CET6, QuestionType.CAREFUL_READING, 2)
        self.assertIn("CET6 400-450 words", spec)
        self.assertIn("exactly 5 four-option multiple-choice questions", spec)
        self.assertIn("Question stems may be either direct questions or unfinished statements", spec)
        self.assertIn("do not force exactly one of each", spec)
        self.assertIn("social issues, ethics, psychology, technology", spec)

    def test_writing_spec_mentions_minimum_words(self) -> None:
        spec = self.pipeline._question_spec(Level.CET4, QuestionType.WRITING, None)
        self.assertIn("at least 120 words but no more than 180 words", spec)
        self.assertIn("high-scoring sample essay", spec)

    def test_translation_spec_mentions_chinese_source_length(self) -> None:
        spec = self.pipeline._question_spec(Level.CET6, QuestionType.TRANSLATION, None)
        self.assertIn("180-200 Chinese characters", spec)
        self.assertIn("reference translation", spec)

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
        self.assertIn("rubric_focus must be an empty array", prompt)
        self.assertIn("10 English statements for matching", prompt)
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
        self.assertIn("task_prompt and reference_answer must be empty strings", prompt)
        self.assertIn("one English word", prompt)
        self.assertIn("Do not place Chinese characters", prompt)

    def test_blueprint_builds_target_counts_and_skill_focus(self) -> None:
        blueprint = self.pipeline._build_blueprint(
            Level.CET6,
            QuestionType.CAREFUL_READING,
            2,
            "推理判断不稳定",
        )
        self.assertEqual(blueprint["target_word_count"], 425)
        self.assertEqual(blueprint["vocabulary_target_count"], 5)
        self.assertTrue(any("infer" in item.lower() or "implied" in item.lower() for item in blueprint["skill_focus"]))
        self.assertIn("推理判断不稳定", blueprint["weakness_focus"])
        self.assertIn("Section C Passage Two", blueprint["exam_profile"])
        self.assertTrue(blueprint["prompt_style_anchors"])
        self.assertTrue(blueprint["authenticity_guardrails"])

    def test_careful_reading_blueprints_differ_by_slot_and_level(self) -> None:
        cet4_slot1 = self.pipeline._build_blueprint(
            Level.CET4,
            QuestionType.CAREFUL_READING,
            1,
            None,
        )
        cet4_slot2 = self.pipeline._build_blueprint(
            Level.CET4,
            QuestionType.CAREFUL_READING,
            2,
            None,
        )
        cet6_slot1 = self.pipeline._build_blueprint(
            Level.CET6,
            QuestionType.CAREFUL_READING,
            1,
            None,
        )
        cet6_slot2 = self.pipeline._build_blueprint(
            Level.CET6,
            QuestionType.CAREFUL_READING,
            2,
            None,
        )
        self.assertIn("explanatory", cet4_slot1["register"])
        self.assertIn("interpretive", cet4_slot2["register"])
        self.assertIn("analytical", cet6_slot1["register"])
        self.assertIn("analytical or reflective", cet6_slot2["register"])
        self.assertIn("original English materials", cet4_slot1["source_material_hint"])
        self.assertIn("reflective passage", cet6_slot2["source_material_hint"])
        self.assertIn("Section C Passage One", cet4_slot1["exam_profile"])
        self.assertIn("Section C Passage Two", cet4_slot2["exam_profile"])
        self.assertIn("Section C Passage One", cet6_slot1["exam_profile"])
        self.assertIn("Section C Passage Two", cet6_slot2["exam_profile"])
        self.assertTrue(any("what do we learn" in item.lower() or "what does the author say" in item.lower() for item in cet4_slot1["prompt_style_anchors"]))
        self.assertTrue(any("attitude" in item.lower() or "why does the author" in item.lower() for item in cet6_slot2["prompt_style_anchors"]))

    def test_blueprint_avoids_recent_topic_keyword_repetition(self) -> None:
        blueprint = self.pipeline._build_blueprint(
            Level.CET4,
            QuestionType.CAREFUL_READING,
            2,
            None,
            recent_topics=[
                "second-hand shopping among young consumers",
                "young people's digital behavior and attention",
                "changing consumer preferences among students",
            ],
        )
        self.assertIn("recent themes", blueprint["novelty_requirement"])
        self.assertTrue(blueprint["avoid_theme_keywords"])
        self.assertNotIn(blueprint["topic"], {
            "second-hand shopping among young consumers",
            "young people's digital behavior and attention",
            "changing consumer preferences among students",
        })

    def test_writing_and_translation_blueprints_keep_cet_style_hints(self) -> None:
        writing = self.pipeline._build_blueprint(
            Level.CET4,
            QuestionType.WRITING,
            None,
            None,
        )
        translation = self.pipeline._build_blueprint(
            Level.CET6,
            QuestionType.TRANSLATION,
            None,
            None,
        )
        self.assertIn("official exam instructions", writing["writing_mode"])
        self.assertIn("Chinese culture", translation["translation_domain"])

    def test_writing_prompt_examples_follow_recent_real_paper_shapes(self) -> None:
        cet4_examples = self.pipeline._writing_prompt_examples(Level.CET4, QuestionType.WRITING)
        cet6_examples = self.pipeline._writing_prompt_examples(Level.CET6, QuestionType.WRITING)
        self.assertTrue(all("Suppose" in item and "You are now to write" in item for item in cet4_examples))
        self.assertTrue(all("begins with the sentence" in item for item in cet6_examples))
        self.assertTrue(all("You should copy the sentence given in quotes" in item for item in cet6_examples))

    def test_generation_prompt_mentions_exam_profile_and_novelty_rules(self) -> None:
        prompt = self.pipeline._generation_user_prompt(
            Level.CET6,
            QuestionType.CAREFUL_READING,
            1,
            {
                "topic": "pricing strategy and market competition",
                "genre": "expository",
                "register": "business",
                "source_material_hint": "adapted business report",
                "exam_profile": "CET6 careful reading passage one: fact-driven business, economics, workplace, or market-report article.",
                "passage_flow": ["market background", "strategy", "consequence"],
                "prompt_style_anchors": ["What is the underlying motive behind ...?"],
                "authenticity_guardrails": ["do not sound like a coaching handout"],
                "writing_prompt_examples": [],
                "target_word_count": 425,
                "word_count_guidance": "Keep the passage paragraphs between 400 and 450 words.",
                "skill_focus": ["market finding or consequence interpretation"],
                "difficulty_controls": ["distractors must be plausible"],
                "structure_plan": ["5 four-option questions"],
                "option_strategy": ["distractors should reflect confusion among business motive and consequence"],
                "writing_mode": "",
                "translation_domain": "",
                "weakness_focus": "none",
                "question_id_pattern": "q1 to q5",
                "vocabulary_target_count": 5,
                "analysis_style": "Chinese only",
                "anti_repeat_topics": ["platform expansion and local business pressure"],
                "avoid_theme_keywords": ["platform", "business"],
                "novelty_requirement": "Avoid recycling recent themes from other recently generated sets.",
            },
        )
        self.assertIn("blueprint.exam_profile", prompt)
        self.assertIn("prompt_style_anchors", prompt)
        self.assertIn("Novelty rule", prompt)

    def test_objective_schema_zeros_subjective_fields(self) -> None:
        schema = self.pipeline._tool_parameters_schema(Level.CET4, QuestionType.BANKED_CLOZE)
        props = schema["properties"]
        self.assertEqual(props["task_prompt"]["enum"], [""])
        self.assertEqual(props["reference_answer"]["enum"], [""])
        self.assertEqual(props["rubric_focus"]["maxItems"], 0)
        self.assertEqual(props["min_response_words"]["enum"], [0])
        self.assertEqual(props["max_response_words"]["enum"], [0])

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
        self.assertEqual(normalized["analysis"]["item_explanations"][0]["correct_answer"], "A")
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

    def test_postprocess_compacts_writing_prompt_lines(self) -> None:
        payload = {
            "title": "Demo Writing",
            "topic": "teamwork",
            "task_prompt": "Write about teamwork.",
            "reference_answer": "This is a sample essay with enough words to pass the length requirement. " * 4,
            "rubric_focus": ["content_relevance", "coherence", "grammar", "lexical_accuracy"],
            "min_response_words": 120,
            "max_response_words": 180,
            "shared_options": [],
            "passage": {
                "title": "Demo Writing",
                "paragraphs": [
                    "1. Suppose your class is doing a group project.",
                    "2. Write an essay on the importance of teamwork.",
                    "3. You may include examples from study and daily life.",
                ],
            },
            "questions": [],
            "answer_key": [],
            "analysis": {"item_explanations": [], "test_tips": []},
            "vocabulary": [],
        }
        normalized = self.pipeline._postprocess_payload(
            payload,
            Level.CET4,
            QuestionType.WRITING,
            None,
        )
        self.assertEqual(len(normalized["passage"]["paragraphs"]), 1)
        self.assertFalse(normalized["passage"]["paragraphs"][0].startswith("1."))

    def test_postprocess_deduplicates_cet4_time_sentence(self) -> None:
        payload = {
            "title": "Demo Writing",
            "topic": "academic writing",
            "task_prompt": "Directions:",
            "reference_answer": "This is a sample essay with enough words to pass the length requirement. " * 4,
            "rubric_focus": ["content_relevance", "coherence", "grammar", "lexical_accuracy"],
            "min_response_words": 120,
            "max_response_words": 180,
            "shared_options": [],
            "passage": {
                "title": "Demo Writing",
                "paragraphs": [
                    "Suppose your university is considering whether academic writing should be a required course for all first-year students. You are now to write an essay stating your view on this issue. You will have 30 minutes for this task. You will have 30 minutes to write the essay. You should write at least 120 words but no more than 180 words."
                ],
            },
            "questions": [],
            "answer_key": [],
            "analysis": {"item_explanations": [], "test_tips": []},
            "vocabulary": [],
        }
        normalized = self.pipeline._postprocess_payload(
            payload,
            Level.CET4,
            QuestionType.WRITING,
            None,
        )
        text = normalized["passage"]["paragraphs"][0]
        self.assertEqual(text.lower().count("you will have 30 minutes"), 1)

    def test_postprocess_objective_payload_clears_subjective_fields(self) -> None:
        payload = {
            "title": "Demo Cloze",
            "topic": "transport",
            "task_prompt": "Ignore me",
            "reference_answer": "Ignore me too",
            "rubric_focus": ["x", "y", "z", "w"],
            "min_response_words": 120,
            "max_response_words": 180,
            "shared_options": ["A. test"] * 15,
            "passage": {"title": "Demo", "paragraphs": ["Text with [1] [2] [3] [4] [5] [6] [7] [8] [9] [10]."]},
            "questions": [{"id": "q1", "prompt": "Blank 1", "skill_tag": "logic"}],
            "answer_key": ["A"],
            "analysis": {"item_explanations": [{"question_id": "q1", "correct_answer": "A", "explanation": "中文"}], "test_tips": ["a", "b", "c"]},
            "vocabulary": [{"lemma": "test", "surface_form": "test", "level_hint": "cet4", "meaning_zh": "测试", "example_en": "This is a test."}],
        }
        normalized = self.pipeline._postprocess_payload(
            payload,
            Level.CET4,
            QuestionType.BANKED_CLOZE,
            None,
        )
        self.assertEqual(normalized["task_prompt"], "")
        self.assertEqual(normalized["reference_answer"], "")
        self.assertEqual(normalized["rubric_focus"], [])
        self.assertEqual(normalized["min_response_words"], 0)
        self.assertEqual(normalized["max_response_words"], 0)


if __name__ == "__main__":
    unittest.main()
