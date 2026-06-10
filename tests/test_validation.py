import unittest

from app.ai.validators import CETQuestionValidator, QuestionSetValidationError
from app.domain.enums import Level, QuestionType


class ValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.validator = CETQuestionValidator()

    def test_banked_cloze_payload_passes_validation(self) -> None:
        payload = {
            "title": "Study Habits",
            "topic": "study",
            "shared_options": [
                "A. access",
                "B. adapt",
                "C. benefit",
                "D. challenge",
                "E. connect",
                "F. efficient",
                "G. flexible",
                "H. maintain",
                "I. pressure",
                "J. routine",
                "K. source",
                "L. strategy",
                "M. transform",
                "N. value",
                "O. virtual",
            ],
            "passage": {
                "title": "Study Habits",
                "paragraphs": [
                    "Online learning became a more [1] part of student life in recent years. Many learners had to [2] with new tools quickly, but the shift also created a [3] when planning was weak. Teachers noticed that students often logged in on time yet still felt lost because they had not formed stable reading habits. In many classrooms, digital platforms gave learners more choices, but they also demanded greater self-control, clearer weekly goals, and more careful review of notes after class. Students who lacked these habits usually spent more time online without making much progress in understanding difficult materials.",
                    "A useful [4] is to divide study time into short blocks. This makes work more [5] and reduces the [6] that comes from trying to finish everything at once. A regular [7] also helps students stay focused. Some learners prepare a list of small daily tasks, while others create a weekly reading calendar and check their progress every evening. The specific method may differ, but the general principle is the same: students need a predictable structure if they want to read carefully, remember details, and avoid the anxiety that comes from rushed preparation before an important test.",
                    "Still, a good plan should remain [8]. Different tasks require different levels of [9], and students need to remember the long-term [10] of steady reading rather than last-minute effort. When a text contains abstract ideas, students should slow down, review key sentences, and connect new information with what they have learned before.",
                ],
            },
            "questions": [
                {"id": f"q{i}", "prompt": f"Blank {i}", "options": [], "skill_tag": "vocabulary", "hint": None}
                for i in range(1, 11)
            ],
            "answer_key": ["O", "E", "D", "L", "F", "I", "J", "G", "A", "N"],
            "analysis": {
                "overall_strategy": "先判断逻辑和词性。",
                "overall_summary": "考查语境词义和搭配。",
                "item_explanations": [
                    {
                        "question_id": f"q{i}",
                        "correct_answer": answer,
                        "explanation": "结合上下文和词性可确定答案。",
                        "skill_tag": "vocabulary",
                    }
                    for i, answer in enumerate(["O", "E", "D", "L", "F", "I", "J", "G", "A", "N"], start=1)
                ],
                "test_tips": ["先易后难", "先看词性", "注意上下文逻辑"],
            },
            "vocabulary": [
                {"lemma": "virtual", "surface_form": "virtual", "level_hint": "cet4", "meaning_zh": "虚拟的", "example_en": ""},
                {"lemma": "strategy", "surface_form": "strategy", "level_hint": "cet4", "meaning_zh": "策略", "example_en": ""},
                {"lemma": "routine", "surface_form": "routine", "level_hint": "cet4", "meaning_zh": "惯例", "example_en": ""},
            ],
        }
        validated = self.validator.validate(payload, Level.CET4, QuestionType.BANKED_CLOZE, None)
        self.assertEqual(len(validated["questions"]), 10)
        self.assertEqual(validated["answer_key"][0], "O")

    def test_careful_reading_requires_exact_skill_distribution(self) -> None:
        payload = {
            "title": "Urban Noise",
            "topic": "city life",
            "shared_options": [],
            "passage": {
                "title": "Urban Noise",
                "paragraphs": [
                    "Many city residents have become so used to background noise that they rarely notice it until they visit a quieter place. Researchers say this constant exposure can affect concentration, sleep, and even emotional well-being. Although not every loud sound is harmful, the problem becomes serious when noise is frequent and unpredictable. Because city life is fast and crowded, people often accept the sounds of engines, construction work, and late-night social activity as an unavoidable part of daily experience.",
                    "A recent study found that people working near construction sites made more mistakes on memory tasks than those in quieter offices. The researchers argue that the brain must spend energy filtering out irrelevant sound, leaving fewer mental resources for complex thinking. Over time, this can reduce productivity and increase fatigue. In addition, workers may not realize that their performance is declining, because the change is gradual rather than sudden.",
                    "The effects are not limited to work. At home, noise from traffic or late-night activity may interrupt sleep cycles, even if sleepers do not fully wake up. Poor sleep can then influence mood, health, and decision-making the next day. This partly explains why some residents report stress even when they claim they have adapted to city life. Children and older adults may be especially sensitive, since they often have less control over their environment and fewer ways to avoid disturbance.",
                    "Urban planners have started to respond by adding green spaces, improving building insulation, and limiting heavy traffic in residential areas. These changes cannot remove all noise, but they may reduce the most damaging kinds. Experts therefore suggest that noise should be treated as a public-health issue rather than a minor inconvenience. They also encourage residents to support practical measures, such as quieter public transport and better-designed housing, instead of assuming that noise is only a private problem.",
                ],
            },
            "questions": [
                {"id": "q1", "prompt": "What is the main idea of the passage?", "options": ["A. a", "B. b", "C. c", "D. d"], "skill_tag": "main_idea", "hint": None},
                {"id": "q2", "prompt": "According to the study, why did workers perform worse?", "options": ["A. a", "B. b", "C. c", "D. d"], "skill_tag": "detail", "hint": None},
                {"id": "q3", "prompt": "What can be inferred about sleep interruption?", "options": ["A. a", "B. b", "C. c", "D. d"], "skill_tag": "inference", "hint": None},
                {"id": "q4", "prompt": "The word 'adapted' most nearly means", "options": ["A. a", "B. b", "C. c", "D. d"], "skill_tag": "vocabulary_in_context", "hint": None},
                {"id": "q5", "prompt": "What is the author's attitude toward urban-noise control?", "options": ["A. a", "B. b", "C. c", "D. d"], "skill_tag": "attitude", "hint": None},
            ],
            "answer_key": ["A", "B", "C", "D", "A"],
            "analysis": {
                "overall_strategy": "先看首段和各段主题句。",
                "overall_summary": "文章讨论城市噪音的影响与治理。",
                "item_explanations": [
                    {"question_id": "q1", "correct_answer": "A", "explanation": "对应主旨。", "skill_tag": "main_idea"},
                    {"question_id": "q2", "correct_answer": "B", "explanation": "对应细节。", "skill_tag": "detail"},
                    {"question_id": "q3", "correct_answer": "C", "explanation": "对应推断。", "skill_tag": "inference"},
                    {"question_id": "q4", "correct_answer": "D", "explanation": "对应词义。", "skill_tag": "vocabulary_in_context"},
                    {"question_id": "q5", "correct_answer": "A", "explanation": "对应态度。", "skill_tag": "attitude"},
                ],
                "test_tips": ["关注主旨句", "回文定位", "结合上下文"],
            },
            "vocabulary": [
                {"lemma": "concentration", "surface_form": "concentration", "level_hint": "cet4", "meaning_zh": "专注力", "example_en": ""},
                {"lemma": "filter", "surface_form": "filtering", "level_hint": "cet4", "meaning_zh": "过滤", "example_en": ""},
                {"lemma": "insulation", "surface_form": "insulation", "level_hint": "cet4", "meaning_zh": "隔音材料", "example_en": ""},
            ],
        }
        validated = self.validator.validate(payload, Level.CET4, QuestionType.CAREFUL_READING, 1)
        self.assertEqual(validated["questions"][3]["skill_tag"], "vocabulary_in_context")

    def test_invalid_careful_reading_raises(self) -> None:
        payload = {
            "title": "Bad Payload",
            "topic": "demo",
            "shared_options": [],
            "passage": {"title": "Bad", "paragraphs": ["Too short."]},
            "questions": [{"id": "q1", "prompt": "bad", "options": ["A"], "skill_tag": "main_idea", "hint": None}],
            "answer_key": ["E"],
            "analysis": {
                "overall_strategy": "",
                "overall_summary": "",
                "item_explanations": [],
                "test_tips": [],
            },
            "vocabulary": [],
        }
        with self.assertRaises(QuestionSetValidationError):
            self.validator.validate(payload, Level.CET4, QuestionType.CAREFUL_READING, 1)

    def test_banked_cloze_rejects_duplicate_answer_letters(self) -> None:
        payload = {
            "title": "Study Habits",
            "topic": "study",
            "shared_options": [
                "A. access", "B. adapt", "C. benefit", "D. challenge", "E. connect",
                "F. efficient", "G. flexible", "H. maintain", "I. pressure", "J. routine",
                "K. source", "L. strategy", "M. transform", "N. value", "O. virtual",
            ],
            "passage": {
                "title": "Study Habits",
                "paragraphs": [
                    "Online learning became a more [1] part of student life in recent years. Many learners had to [2] with new tools quickly, but the shift also created a [3] when planning was weak. Teachers noticed that students often logged in on time yet still felt lost because they had not formed stable reading habits. In many classrooms, digital platforms gave learners more choices, but they also demanded greater self-control, clearer weekly goals, and more careful review of notes after class. Students who lacked these habits usually spent more time online without making much progress in understanding difficult materials.",
                    "A useful [4] is to divide study time into short blocks. This makes work more [5] and reduces the [6] that comes from trying to finish everything at once. A regular [7] also helps students stay focused. Some learners prepare a list of small daily tasks, while others create a weekly reading calendar and check their progress every evening. The specific method may differ, but the general principle is the same: students need a predictable structure if they want to read carefully, remember details, and avoid the anxiety that comes from rushed preparation before an important test.",
                    "Still, a good plan should remain [8]. Different tasks require different levels of [9], and students need to remember the long-term [10] of steady reading rather than last-minute effort. When a text contains abstract ideas, students should slow down, review key sentences, and connect new information with what they have learned before.",
                ],
            },
            "questions": [
                {"id": f"q{i}", "prompt": f"Blank {i}", "options": [], "skill_tag": "vocabulary", "hint": None}
                for i in range(1, 11)
            ],
            "answer_key": ["O", "E", "D", "L", "F", "I", "J", "J", "A", "N"],
            "analysis": {
                "overall_strategy": "先判断逻辑和词性。",
                "overall_summary": "考查语境词义和搭配。",
                "item_explanations": [
                    {
                        "question_id": f"q{i}",
                        "correct_answer": answer,
                        "explanation": "结合上下文和词性可确定答案。",
                        "skill_tag": "vocabulary",
                    }
                    for i, answer in enumerate(["O", "E", "D", "L", "F", "I", "J", "J", "A", "N"], start=1)
                ],
                "test_tips": ["先易后难", "先看词性", "注意上下文逻辑"],
            },
            "vocabulary": [
                {"lemma": "virtual", "surface_form": "virtual", "level_hint": "cet4", "meaning_zh": "虚拟的", "example_en": ""},
                {"lemma": "strategy", "surface_form": "strategy", "level_hint": "cet4", "meaning_zh": "策略", "example_en": ""},
                {"lemma": "routine", "surface_form": "routine", "level_hint": "cet4", "meaning_zh": "惯例", "example_en": ""},
            ],
        }
        with self.assertRaises(QuestionSetValidationError):
            self.validator.validate(payload, Level.CET4, QuestionType.BANKED_CLOZE, None)

    def test_banked_cloze_rejects_phrase_shared_option(self) -> None:
        payload = {
            "title": "Study Habits",
            "topic": "study",
            "shared_options": [
                "A. in fact",
                "B. adapt",
                "C. benefit",
                "D. challenge",
                "E. connect",
                "F. efficient",
                "G. flexible",
                "H. maintain",
                "I. pressure",
                "J. routine",
                "K. source",
                "L. strategy",
                "M. transform",
                "N. value",
                "O. virtual",
            ],
            "passage": {
                "title": "Study Habits",
                "paragraphs": [
                    "Online learning became a more [1] part of student life in recent years. Many learners had to [2] with new tools quickly, but the shift also created a [3] when planning was weak. Teachers noticed that students often logged in on time yet still felt lost because they had not formed stable reading habits. In many classrooms, digital platforms gave learners more choices, but they also demanded greater self-control, clearer weekly goals, and more careful review of notes after class. Students who lacked these habits usually spent more time online without making much progress in understanding difficult materials.",
                    "A useful [4] is to divide study time into short blocks. This makes work more [5] and reduces the [6] that comes from trying to finish everything at once. A regular [7] also helps students stay focused. Some learners prepare a list of small daily tasks, while others create a weekly reading calendar and check their progress every evening. The specific method may differ, but the general principle is the same: students need a predictable structure if they want to read carefully, remember details, and avoid the anxiety that comes from rushed preparation before an important test.",
                    "Still, a good plan should remain [8]. Different tasks require different levels of [9], and students need to remember the long-term [10] of steady reading rather than last-minute effort. When a text contains abstract ideas, students should slow down, review key sentences, and connect new information with what they have learned before.",
                ],
            },
            "questions": [
                {"id": f"q{i}", "prompt": f"Blank {i}", "options": [], "skill_tag": "vocabulary", "hint": None}
                for i in range(1, 11)
            ],
            "answer_key": ["O", "E", "D", "L", "F", "I", "J", "G", "A", "N"],
            "analysis": {
                "overall_strategy": "先判断逻辑和词性。",
                "overall_summary": "考查语境词义和搭配。",
                "item_explanations": [
                    {
                        "question_id": f"q{i}",
                        "correct_answer": answer,
                        "explanation": "结合上下文和词性可确定答案。",
                        "skill_tag": "vocabulary",
                    }
                    for i, answer in enumerate(["O", "E", "D", "L", "F", "I", "J", "G", "A", "N"], start=1)
                ],
                "test_tips": ["先易后难", "先看词性", "注意上下文逻辑"],
            },
            "vocabulary": [
                {"lemma": "virtual", "surface_form": "virtual", "level_hint": "cet4", "meaning_zh": "虚拟的", "example_en": ""},
                {"lemma": "strategy", "surface_form": "strategy", "level_hint": "cet4", "meaning_zh": "策略", "example_en": ""},
                {"lemma": "routine", "surface_form": "routine", "level_hint": "cet4", "meaning_zh": "惯例", "example_en": ""},
            ],
        }
        with self.assertRaises(QuestionSetValidationError):
            self.validator.validate(payload, Level.CET4, QuestionType.BANKED_CLOZE, None)

    def test_validator_rejects_chinese_in_english_fields(self) -> None:
        payload = {
            "title": "Study Habits",
            "topic": "学习",
            "shared_options": [
                "A. access", "B. adapt", "C. benefit", "D. challenge", "E. connect",
                "F. efficient", "G. flexible", "H. maintain", "I. pressure", "J. routine",
                "K. source", "L. strategy", "M. transform", "N. value", "O. virtual",
            ],
            "passage": {
                "title": "Study Habits",
                "paragraphs": [
                    "Online learning became a more [1] part of student life in recent years. Many learners had to [2] with new tools quickly, but the shift also created a [3] when planning was weak. Teachers noticed that students often logged in on time yet still felt lost because they had not formed stable reading habits. In many classrooms, digital platforms gave learners more choices, but they also demanded greater self-control, clearer weekly goals, and more careful review of notes after class. Students who lacked these habits usually spent more time online without making much progress in understanding difficult materials.",
                    "A useful [4] is to divide study time into short blocks. This makes work more [5] and reduces the [6] that comes from trying to finish everything at once. A regular [7] also helps students stay focused. Some learners prepare a list of small daily tasks, while others create a weekly reading calendar and check their progress every evening. The specific method may differ, but the general principle is the same: students need a predictable structure if they want to read carefully, remember details, and avoid the anxiety that comes from rushed preparation before an important test.",
                    "Still, a good plan should remain [8]. Different tasks require different levels of [9], and students need to remember the long-term [10] of steady reading rather than last-minute effort. When a text contains abstract ideas, students should slow down, review key sentences, and connect new information with what they have learned before.",
                ],
            },
            "questions": [
                {"id": f"q{i}", "prompt": f"Blank {i}", "options": [], "skill_tag": "vocabulary", "hint": None}
                for i in range(1, 11)
            ],
            "answer_key": ["O", "E", "D", "L", "F", "I", "J", "G", "A", "N"],
            "analysis": {
                "overall_strategy": "先判断逻辑和词性。",
                "overall_summary": "考查语境词义和搭配。",
                "item_explanations": [
                    {
                        "question_id": f"q{i}",
                        "correct_answer": answer,
                        "explanation": "结合上下文和词性可确定答案。",
                        "skill_tag": "vocabulary",
                    }
                    for i, answer in enumerate(["O", "E", "D", "L", "F", "I", "J", "G", "A", "N"], start=1)
                ],
                "test_tips": ["先易后难", "先看词性", "注意上下文逻辑"],
            },
            "vocabulary": [
                {"lemma": "virtual", "surface_form": "virtual", "level_hint": "cet4", "meaning_zh": "虚拟的", "example_en": ""},
                {"lemma": "strategy", "surface_form": "strategy", "level_hint": "cet4", "meaning_zh": "策略", "example_en": ""},
                {"lemma": "routine", "surface_form": "routine", "level_hint": "cet4", "meaning_zh": "惯例", "example_en": ""},
            ],
        }
        with self.assertRaises(QuestionSetValidationError):
            self.validator.validate(payload, Level.CET4, QuestionType.BANKED_CLOZE, None)

    def test_long_reading_rejects_question_like_prompts(self) -> None:
        payload = {
            "title": "Urban Planning and Green Design",
            "topic": "urban planning",
            "shared_options": [],
            "passage": {
                "title": "Urban Planning and Green Design",
                "paragraphs": [
                    "A. City planners once focused mainly on traffic flow and housing supply, but many now consider how public spaces shape emotional well-being and social trust among residents.",
                    "B. One reason for this change is that green design often improves not only visual quality but also daily comfort, especially in neighborhoods with limited private space.",
                    "C. Researchers have found that tree cover can reduce local temperatures, soften traffic noise, and encourage longer outdoor stays, all of which may affect social contact.",
                    "D. However, simply adding plants does not guarantee success, because design must match how people actually move, rest, and gather within a neighborhood.",
                    "E. Some cities have redesigned school routes so that children walk through safer and greener streets, hoping to build healthier habits early in life.",
                    "F. In commercial districts, planners often face the opposite challenge: they need lively public areas without creating crowding that discourages older residents from using them.",
                    "G. This is why planners increasingly collect local feedback before redesigning a square, a path, or a transport hub that affects many routines at once.",
                    "H. Digital tools also help, since simulation models can predict where shade, seating, and open sight lines may improve public use at different hours.",
                    "I. Even so, experts warn that models should guide rather than replace observation, because local habits sometimes produce outcomes that data alone cannot anticipate.",
                    "J. In the end, successful green design depends on balancing environmental goals with how residents actually experience streets, parks, and shared facilities.",
                    "K. A well-used public area is usually the result of many small design choices working together rather than one dramatic architectural gesture.",
                ],
            },
            "questions": [
                {"id": "q1", "prompt": "What has changed in the priorities of city planners?", "options": [], "skill_tag": "matching", "hint": None},
                {"id": "q2", "prompt": "Why is green design useful in dense neighborhoods?", "options": [], "skill_tag": "matching", "hint": None},
                {"id": "q3", "prompt": "What can tree cover influence in cities?", "options": [], "skill_tag": "matching", "hint": None},
                {"id": "q4", "prompt": "Why might planting alone be insufficient?", "options": [], "skill_tag": "matching", "hint": None},
                {"id": "q5", "prompt": "How have some cities changed school routes?", "options": [], "skill_tag": "matching", "hint": None},
                {"id": "q6", "prompt": "What challenge exists in busy commercial districts?", "options": [], "skill_tag": "matching", "hint": None},
                {"id": "q7", "prompt": "Why is local feedback increasingly collected?", "options": [], "skill_tag": "matching", "hint": None},
                {"id": "q8", "prompt": "How can digital tools support green design?", "options": [], "skill_tag": "matching", "hint": None},
                {"id": "q9", "prompt": "Why should models not replace real observation?", "options": [], "skill_tag": "matching", "hint": None},
                {"id": "q10", "prompt": "What determines successful green design in the end?", "options": [], "skill_tag": "matching", "hint": None},
            ],
            "answer_key": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
            "analysis": {
                "overall_strategy": "先看关键词，再定位同义改写。",
                "overall_summary": "文章讨论城市绿色设计与公共空间规划。",
                "item_explanations": [
                    {"question_id": f"q{i}", "correct_answer": chr(ord('A') + i - 1), "explanation": "对应原文信息。", "skill_tag": "matching"}
                    for i in range(1, 11)
                ],
                "test_tips": ["先扫题干", "再回文定位", "关注同义改写"],
            },
            "vocabulary": [
                {"lemma": "resident", "surface_form": "residents", "level_hint": "cet6", "meaning_zh": "居民", "example_en": ""},
                {"lemma": "gesture", "surface_form": "gesture", "level_hint": "cet6", "meaning_zh": "姿态；举措", "example_en": ""},
                {"lemma": "simulation", "surface_form": "simulation", "level_hint": "cet6", "meaning_zh": "模拟", "example_en": ""},
            ],
        }
        with self.assertRaises(QuestionSetValidationError):
            self.validator.validate(payload, Level.CET6, QuestionType.LONG_READING, None)


if __name__ == "__main__":
    unittest.main()
