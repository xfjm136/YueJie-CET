from __future__ import annotations

from app.domain.enums import Level, QuestionType
from app.domain.schemas import (
    AnalysisReport,
    ItemExplanation,
    Passage,
    Question,
    QuestionSet,
    VocabularyItem,
    make_id,
)


class MockQuestionFactory:
    @classmethod
    def create(
        cls,
        level: Level,
        question_type: QuestionType,
        slot: int | None,
        model_name: str,
    ) -> QuestionSet:
        if question_type is QuestionType.BANKED_CLOZE:
            return cls._banked_cloze(level, model_name)
        if question_type is QuestionType.LONG_READING:
            return cls._long_reading(level, model_name)
        return cls._careful_reading(level, slot or 1, model_name)

    @staticmethod
    def _banked_cloze(level: Level, model_name: str) -> QuestionSet:
        shared_options = [
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
        ]
        questions = [
            Question(id=f"q{i}", prompt=f"第 {i} 空", skill_tag="vocabulary")
            for i in range(1, 11)
        ]
        answer_key = ["O", "E", "D", "L", "F", "I", "J", "G", "A", "N"]
        explanations = [
            ItemExplanation(
                question_id=f"q{i}",
                correct_answer=answer_key[i - 1],
                explanation="结合上下文语义和词性判断，这是最自然的搭配。",
                skill_tag="vocabulary",
            )
            for i in range(1, 11)
        ]
        vocabulary = [
            VocabularyItem("flexible", "flexible", level.value, "灵活的"),
            VocabularyItem("routine", "routine", level.value, "日常惯例"),
            VocabularyItem("strategy", "strategy", level.value, "策略"),
        ]
        passage = Passage(
            title="Building a Better Study Routine",
            paragraphs=[
                (
                    "Online learning became a more [1] part of student life in recent years. "
                    "Many learners had to [2] with new tools quickly, but the shift also created a [3]: "
                    "without clear planning, students often felt lost."
                ),
                (
                    "A useful [4] is to divide study time into short blocks. This makes work more [5] and "
                    "reduces the [6] that comes from trying to finish everything at once. A regular [7] also "
                    "helps students stay focused."
                ),
                (
                    "Still, a good plan should remain [8]. Different tasks require different levels of [9], "
                    "and students need to remember the long-term [10] of steady reading rather than last-minute effort."
                ),
            ],
        )
        return QuestionSet(
            id=make_id("qs"),
            level=level,
            question_type=QuestionType.BANKED_CLOZE,
            slot=None,
            title="学习节奏与在线学习",
            topic="study habits",
            passage=passage,
            questions=questions,
            answer_key=answer_key,
            analysis=AnalysisReport(
                overall_strategy="先看上下文逻辑，再看词性和固定搭配。",
                overall_summary="这篇选词填空主要考查学习场景中的高频词汇和搭配。",
                item_explanations=explanations,
                test_tips=[
                    "先做最明显的空，再回填较难位置。",
                    "警惕近义词干扰，优先判断词性和逻辑关系。",
                ],
            ),
            vocabulary=vocabulary,
            shared_options=shared_options,
            word_count=225 if level is Level.CET4 else 275,
            generator_model=model_name,
            source_type="mock",
        )

    @staticmethod
    def _long_reading(level: Level, model_name: str) -> QuestionSet:
        paragraphs = [
            "A. Universities increasingly treat reading as a skill that can be trained rather than a habit students either have or do not have.",
            "B. Some teachers begin with very short articles so that students can focus on structure before handling difficult vocabulary.",
            "C. Digital annotation tools make it easier to highlight key points, but they can also tempt students to mark too much.",
            "D. Good readers know that not every line deserves equal attention; they slow down only when a paragraph carries the main argument.",
            "E. Research suggests that students often remember information better when they summarize it in their own words.",
            "F. Time pressure is one reason many learners panic in long reading sections and start scanning randomly.",
            "G. A simple method is to label each paragraph with a tiny note such as cause, example, or conclusion.",
            "H. Once students see how ideas move from paragraph to paragraph, matching questions become easier.",
            "I. Teachers also remind students that one paragraph may answer more than one item, while another may answer none.",
            "J. Regular timed practice is still necessary because strategy alone cannot replace speed.",
            "K. In the end, confidence grows when learners know what they are looking for before they return to the passage.",
        ]
        questions = [
            Question(id=f"q{i}", prompt=prompt, skill_tag="matching")
            for i, prompt in enumerate(
                [
                    "Students should learn to notice how ideas are organized inside a passage.",
                    "Some tools help readers, but they may create a new problem at the same time.",
                    "A paragraph may match several items, and some paragraphs may match none.",
                    "Students under time pressure may lose control of their reading process.",
                    "Summarizing information can improve memory.",
                    "Practice needs both method and speed training.",
                    "Not every sentence should be read at the same speed.",
                    "Teachers may reduce article difficulty at the beginning of training.",
                    "Tiny labels can help readers track paragraph function.",
                    "A sense of purpose can make readers calmer and more accurate.",
                ],
                start=1,
            )
        ]
        answer_key = ["H", "C", "I", "F", "E", "J", "D", "B", "G", "K"]
        explanations = [
            ItemExplanation(
                question_id=f"q{i}",
                correct_answer=answer_key[i - 1],
                explanation="题干信息与对应段的中心意思一致，属于典型同义改写定位。",
                skill_tag="matching",
            )
            for i in range(1, 11)
        ]
        return QuestionSet(
            id=make_id("qs"),
            level=level,
            question_type=QuestionType.LONG_READING,
            slot=None,
            title="How to Train Academic Reading",
            topic="reading strategy",
            passage=Passage(title="How to Train Academic Reading", paragraphs=paragraphs),
            questions=questions,
            answer_key=answer_key,
            analysis=AnalysisReport(
                overall_strategy="先看题干关键词，再给每段打功能标签，最后快速回文定位。",
                overall_summary="这套长篇阅读重点考查略读、查读和段落功能识别。",
                item_explanations=explanations,
                test_tips=[
                    "先做信息特征明显的题。",
                    "遇到抽象句子时优先找同义表达，而不是逐字对照。",
                ],
            ),
            vocabulary=[
                VocabularyItem("annotation", "annotation", level.value, "标注"),
                VocabularyItem("summarize", "summarize", level.value, "概括，总结"),
                VocabularyItem("argument", "argument", level.value, "论点"),
            ],
            shared_options=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"],
            word_count=1000 if level is Level.CET4 else 1200,
            generator_model=model_name,
            source_type="mock",
        )

    @staticmethod
    def _careful_reading(level: Level, slot: int, model_name: str) -> QuestionSet:
        questions = [
            Question(
                id=f"q{i}",
                prompt=prompt,
                options=options,
                skill_tag=skill,
            )
            for i, (prompt, options, skill) in enumerate(
                [
                    (
                        "What is the main purpose of the passage?",
                        [
                            "A. To compare several mobile apps.",
                            "B. To explain why deep reading still matters.",
                            "C. To criticize all forms of digital reading.",
                            "D. To introduce a new college major.",
                        ],
                        "main_idea",
                    ),
                    (
                        "According to the passage, why do many students reread lines?",
                        [
                            "A. They forget paragraph labels.",
                            "B. They dislike long articles.",
                            "C. They do not read with a clear goal.",
                            "D. They refuse to take notes.",
                        ],
                        "detail",
                    ),
                    (
                        "What can be inferred about timed practice?",
                        [
                            "A. It is only useful for advanced learners.",
                            "B. It should be combined with reflection.",
                            "C. It usually damages comprehension.",
                            "D. It works without any strategy training.",
                        ],
                        "inference",
                    ),
                    (
                        "The word 'deliberate' is closest in meaning to ______.",
                        [
                            "A. careless",
                            "B. slow but purposeful",
                            "C. public",
                            "D. repeated by others",
                        ],
                        "vocabulary",
                    ),
                    (
                        "What is the author's attitude toward reading strategy?",
                        [
                            "A. Doubtful.",
                            "B. Strongly negative.",
                            "C. Supportive but practical.",
                            "D. Completely uninterested.",
                        ],
                        "attitude",
                    ),
                ],
                start=1,
            )
        ]
        answer_key = ["B", "C", "B", "B", "C"]
        explanations = [
            ItemExplanation(
                question_id="q1",
                correct_answer="B",
                explanation="全文围绕“深度阅读为何仍然重要以及如何训练”展开。",
                skill_tag="main_idea",
            ),
            ItemExplanation(
                question_id="q2",
                correct_answer="C",
                explanation="文中指出没有明确任务目标时，学生容易反复重读却仍抓不住重点。",
                skill_tag="detail",
            ),
            ItemExplanation(
                question_id="q3",
                correct_answer="B",
                explanation="作者强调计时练习有效，但必须和复盘、策略一起使用。",
                skill_tag="inference",
            ),
            ItemExplanation(
                question_id="q4",
                correct_answer="B",
                explanation="deliberate 在文中表示有意识、有目的地慢下来处理关键信息。",
                skill_tag="vocabulary",
            ),
            ItemExplanation(
                question_id="q5",
                correct_answer="C",
                explanation="作者认可策略训练，但也反复提醒要结合实际速度和训练量。",
                skill_tag="attitude",
            ),
        ]
        passage = Passage(
            title="Why Deep Reading Still Matters",
            paragraphs=[
                (
                    "Students today read on screens more often than previous generations did, yet many teachers notice "
                    "that reading more does not always mean reading well. Learners may move quickly through pages without "
                    "deciding what information deserves attention."
                ),
                (
                    "As a result, they often reread lines that they already saw because they never formed a clear goal in "
                    "the first place. Effective readers, by contrast, adjust their speed. They skim for structure and then "
                    "slow down deliberately when a paragraph contains an argument, an example, or a turning point."
                ),
                (
                    "Timed practice remains useful, but only when it is followed by reflection. Students need to ask why an "
                    "option was correct, what clues they missed, and how they might locate the same information faster next time."
                ),
            ],
        )
        return QuestionSet(
            id=make_id("qs"),
            level=level,
            question_type=QuestionType.CAREFUL_READING,
            slot=slot,
            title=f"仔细阅读 {slot}",
            topic="reading strategy",
            passage=passage,
            questions=questions,
            answer_key=answer_key,
            analysis=AnalysisReport(
                overall_strategy="先看题干考点，再回文找定位句，不要整篇逐句硬读。",
                overall_summary="这篇仔细阅读覆盖主旨、细节、推断、词义和态度五类高频题。",
                item_explanations=explanations,
                test_tips=[
                    "主旨题先看首段和末段。",
                    "词义题优先看上下文逻辑而不是只盯住单词本身。",
                ],
            ),
            vocabulary=[
                VocabularyItem("deliberate", "deliberate", level.value, "有意的，深思熟虑的"),
                VocabularyItem("skim", "skim", level.value, "略读"),
                VocabularyItem("reflection", "reflection", level.value, "复盘，反思"),
            ],
            shared_options=[],
            word_count=320 if level is Level.CET4 else 420,
            generator_model=model_name,
            source_type="mock",
        )

