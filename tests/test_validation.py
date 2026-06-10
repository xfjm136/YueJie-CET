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

    def test_writing_payload_passes_validation(self) -> None:
        payload = {
            "title": "Writing Task",
            "topic": "independent thinking",
            "task_prompt": "For this part, you are allowed 30 minutes to write an essay of no less than 120 words according to the outline given below.",
            "reference_answer": "Independent thinking remains essential in modern learning. Although digital tools can save time and improve access to information, students still need to judge evidence, organize ideas, and form their own conclusions. In classroom practice, learners benefit most when they use technology as support rather than as a substitute for reflection. For example, an online summary may help a student review faster, but deep understanding still requires careful reading, note-taking, and personal reflection on evidence. Universities should therefore teach students how to work with new tools while preserving habits of questioning, comparison, and self-correction. In this way, technology can increase efficiency without weakening the human ability to think clearly, argue responsibly, and learn for the long term. It also helps students remain confident when they face new information and unfamiliar opinions.",
            "rubric_focus": ["content_relevance", "coherence", "grammar", "lexical_accuracy"],
            "min_response_words": 120,
            "max_response_words": 220,
            "shared_options": [],
            "passage": {
                "title": "Writing Task",
                "paragraphs": [
                    "Suppose your university is organizing a reading campaign.",
                    "Write an essay on why independent thinking still matters in the age of AI.",
                    "You may include examples from study and daily life.",
                ],
            },
            "questions": [],
            "answer_key": [],
            "analysis": {
                "overall_strategy": "先审题，再列提纲，最后检查语言错误。",
                "overall_summary": "这道写作题强调切题、结构和语言准确度。",
                "item_explanations": [],
                "test_tips": ["先定结构", "再展开主点", "最后检查语法和拼写"],
            },
            "vocabulary": [
                {"lemma": "independent", "surface_form": "independent", "level_hint": "cet4", "meaning_zh": "独立的", "example_en": "Independent thinking helps students judge ideas."},
                {"lemma": "reflection", "surface_form": "reflection", "level_hint": "cet4", "meaning_zh": "反思", "example_en": "Reflection improves long-term learning."},
                {"lemma": "substitute", "surface_form": "substitute", "level_hint": "cet4", "meaning_zh": "替代品", "example_en": "A machine cannot be a full substitute for human judgment."},
            ],
        }
        validated = self.validator.validate(payload, Level.CET4, QuestionType.WRITING, None)
        self.assertEqual(validated["questions"], [])
        self.assertGreaterEqual(validated["word_count"], 120)

    def test_translation_payload_passes_validation(self) -> None:
        payload = {
            "title": "Translation Task",
            "topic": "public libraries",
            "task_prompt": "For this part, you are allowed 30 minutes to translate the following Chinese passage into English.",
            "reference_answer": "In recent years, many Chinese cities have actively promoted the development of public libraries. In addition to lending services, these libraries often organize lectures, reading clubs, and family activities. They have not only enriched the cultural life of residents but also provided a more open and comfortable public space for community interaction.",
            "rubric_focus": ["translation_accuracy", "translation_fluency", "grammar", "lexical_accuracy"],
            "min_response_words": 120,
            "max_response_words": 220,
            "shared_options": [],
            "passage": {
                "title": "Translation Task",
                "paragraphs": [
                    "中国许多城市近年积极推动公共图书馆建设。除了提供借阅服务，这些图书馆还经常举办讲座、读书会和亲子活动，吸引不同年龄的居民主动参加。它们不仅丰富了市民的文化生活，也为社区交流提供了更加开放和舒适的公共空间，使更多家庭能够在日常生活中接触阅读、分享经验并建立更紧密的联系，同时也让社区公共服务显得更加温暖而有活力。"
                ],
            },
            "questions": [],
            "answer_key": [],
            "analysis": {
                "overall_strategy": "先分句，再整理逻辑关系，最后优化英语表达。",
                "overall_summary": "这道翻译题强调信息准确、表达通顺和语法稳定。",
                "item_explanations": [],
                "test_tips": ["先保信息", "再调句式", "最后查语法和搭配"],
            },
            "vocabulary": [
                {"lemma": "community", "surface_form": "community", "level_hint": "cet4", "meaning_zh": "社区", "example_en": "Libraries can strengthen community ties."},
                {"lemma": "cultural", "surface_form": "cultural", "level_hint": "cet4", "meaning_zh": "文化的", "example_en": "The city offers rich cultural resources."},
                {"lemma": "interaction", "surface_form": "interaction", "level_hint": "cet4", "meaning_zh": "交流", "example_en": "Public spaces encourage social interaction."},
            ],
        }
        validated = self.validator.validate(payload, Level.CET4, QuestionType.TRANSLATION, None)
        self.assertEqual(validated["questions"], [])
        self.assertGreaterEqual(validated["word_count"], 140)


if __name__ == "__main__":
    unittest.main()
