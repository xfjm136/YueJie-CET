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
        if question_type is QuestionType.WRITING:
            return cls._writing(level, model_name)
        if question_type is QuestionType.TRANSLATION:
            return cls._translation(level, model_name)
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
        if slot == 1:
            questions = [
                Question(
                    id="q1",
                    prompt="Why did the researchers include two sleep groups in the experiment?",
                    options=[
                        "A. To compare different memory outcomes.",
                        "B. To reduce the cost of the project.",
                        "C. To shorten the testing schedule.",
                        "D. To replace the written survey.",
                    ],
                    skill_tag="detail",
                ),
                Question(
                    id="q2",
                    prompt="What did the study find about students who slept less?",
                    options=[
                        "A. They finished faster but remembered more.",
                        "B. They showed weaker recall the next day.",
                        "C. They refused to join the second test.",
                        "D. They improved after drinking coffee.",
                    ],
                    skill_tag="detail",
                ),
                Question(
                    id="q3",
                    prompt="The phrase \"memory consolidation\" most nearly means ______.",
                    options=[
                        "A. organizing and strengthening memories",
                        "B. discussing personal experiences",
                        "C. delaying a medical treatment",
                        "D. changing experimental equipment",
                    ],
                    skill_tag="vocabulary_in_context",
                ),
                Question(
                    id="q4",
                    prompt="What was the main purpose of the experiment?",
                    options=[
                        "A. To prove that all students need exactly eight hours of sleep.",
                        "B. To see how sleep length influenced memory performance.",
                        "C. To compare online and offline learning habits.",
                        "D. To test whether students liked laboratory settings.",
                    ],
                    skill_tag="main_idea",
                ),
                Question(
                    id="q5",
                    prompt="What can be inferred about sleep and learning from the passage?",
                    options=[
                        "A. Short sleep affects recall more than students expect.",
                        "B. Sleep only matters in science classes.",
                        "C. Memory declines can be solved by working longer hours.",
                        "D. Students perform best when tested immediately.",
                    ],
                    skill_tag="inference",
                ),
            ]
            answer_key = ["A", "B", "A", "B", "A"]
            explanations = [
                ItemExplanation("q1", "A", "分组是为了比较睡眠长短对记忆表现的影响。", "detail"),
                ItemExplanation("q2", "B", "实验结果指出睡眠更少的学生次日回忆更弱。", "detail"),
                ItemExplanation("q3", "A", "memory consolidation 指记忆被加工和巩固。", "vocabulary_in_context"),
                ItemExplanation("q4", "B", "全文核心是考察睡眠时长如何影响记忆表现。", "main_idea"),
                ItemExplanation("q5", "A", "作者暗示睡眠不足对学习效果的伤害常被低估。", "inference"),
            ]
            passage = Passage(
                title="How Sleep Shapes Memory",
                paragraphs=[
                    "A university research team recently examined how different sleep schedules affected students' memory. The researchers divided volunteers into two groups. One group slept for about eight hours, while the other slept for only five hours before completing the same recall tasks the next morning.",
                    "The goal was not simply to measure tiredness. Instead, the team wanted to know whether memory consolidation would suffer when students had less time for deep sleep. Both groups studied a list of unfamiliar facts in the evening and then returned for testing the next day.",
                    "The results showed a clear difference. Students who slept less could still recognize some of the information, but they performed worse on tasks that required them to recall details without help. The researchers therefore argued that sleep should be treated as part of the learning process rather than as time lost from study.",
                ],
            )
            topic = "sleep research and memory"
        else:
            questions = [
                Question(
                    id="q1",
                    prompt="What can be inferred about young shoppers from Paragraph 2?",
                    options=[
                        "A. They reject all forms of online shopping.",
                        "B. They often connect second-hand shopping with personal values.",
                        "C. They care more about speed than price.",
                        "D. They prefer large chain stores in every situation.",
                    ],
                    skill_tag="inference",
                ),
                Question(
                    id="q2",
                    prompt="Why does the author mention social media videos in the passage?",
                    options=[
                        "A. To show how quickly shopping advice can spread.",
                        "B. To prove that fashion is becoming less expensive.",
                        "C. To explain why physical stores are disappearing.",
                        "D. To criticize students for copying one another.",
                    ],
                    skill_tag="inference",
                ),
                Question(
                    id="q3",
                    prompt="What is the main idea of the passage?",
                    options=[
                        "A. Young consumers are abandoning fashion trends.",
                        "B. Traditional retailers should stop using digital marketing.",
                        "C. The rise of second-hand shopping reflects both practical and social motives.",
                        "D. Buying used goods is only a temporary campus fashion.",
                    ],
                    skill_tag="main_idea",
                ),
                Question(
                    id="q4",
                    prompt="What is the author's attitude toward second-hand shopping among young people?",
                    options=[
                        "A. Entirely doubtful.",
                        "B. Mildly supportive.",
                        "C. Openly annoyed.",
                        "D. Completely indifferent.",
                    ],
                    skill_tag="attitude",
                ),
                Question(
                    id="q5",
                    prompt="What contributes to the popularity of second-hand stores among students?",
                    options=[
                        "A. They offer a mix of savings, uniqueness, and social meaning.",
                        "B. They are required by most universities.",
                        "C. They provide better online delivery than major brands.",
                        "D. They only sell products made for young consumers.",
                    ],
                    skill_tag="detail",
                ),
            ]
            answer_key = ["B", "A", "C", "B", "A"]
            explanations = [
                ItemExplanation("q1", "B", "文中暗示许多年轻人把二手购物与环保、自我表达联系起来。", "inference"),
                ItemExplanation("q2", "A", "举社交媒体视频的例子是为了说明趋势如何被放大和传播。", "inference"),
                ItemExplanation("q3", "C", "全文围绕二手购物走红背后的务实和社会性动机展开。", "main_idea"),
                ItemExplanation("q4", "B", "作者总体持认可态度，但并未把现象说得过于理想化。", "attitude"),
                ItemExplanation("q5", "A", "细节题，对应价格、个性和社会意义三方面原因。", "detail"),
            ]
            passage = Passage(
                title="Why Used Stores Appeal to the Young",
                paragraphs=[
                    "Second-hand stores used to be seen mainly as places for people with limited budgets. Today, however, many young shoppers view them differently. They still appreciate lower prices, but they also enjoy the possibility of finding items that feel more personal than those sold in large chain stores.",
                    "Part of the change comes from social media. Short videos often turn a single lucky purchase into a story about creativity, sustainability, or identity. As a result, shopping itself begins to look less like a simple exchange of money and more like a way to express taste and values.",
                    "This does not mean every buyer has the same motive. Some are mainly trying to save money, while others want to reduce waste or avoid mass-produced products. Yet the trend has grown because second-hand shopping allows these practical and symbolic motives to meet in one place.",
                ],
            )
            topic = "second-hand shopping trend"
        return QuestionSet(
            id=make_id("qs"),
            level=level,
            question_type=QuestionType.CAREFUL_READING,
            slot=slot,
            title=f"仔细阅读 {slot}",
            topic=topic,
            passage=passage,
            questions=questions,
            answer_key=answer_key,
            analysis=AnalysisReport(
                overall_strategy="先看题干考点，再回文找定位句，不要整篇逐句硬读。",
                overall_summary=(
                    "这篇仔细阅读更偏事实检索、结果理解和语境释义。"
                    if slot == 1
                    else "这篇仔细阅读更偏观点推断、例证作用和作者态度判断。"
                ),
                item_explanations=explanations,
                test_tips=[
                    "事实驱动文先抓实验/报告目的、过程和结果。",
                    "观点驱动文先抓转折、例证意图和作者立场。",
                    "不要被原词重现迷惑，要看是否真的对应原文逻辑。",
                ],
            ),
            vocabulary=[
                VocabularyItem(
                    "consolidation" if slot == 1 else "trend",
                    "consolidation" if slot == 1 else "trend",
                    level.value,
                    "巩固，强化" if slot == 1 else "趋势，潮流",
                ),
                VocabularyItem(
                    "participant" if slot == 1 else "consumer",
                    "participant" if slot == 1 else "consumer",
                    level.value,
                    "参与者" if slot == 1 else "消费者",
                ),
                VocabularyItem(
                    "finding" if slot == 1 else "identity",
                    "finding" if slot == 1 else "identity",
                    level.value,
                    "研究发现" if slot == 1 else "身份认同",
                ),
            ],
            shared_options=[],
            word_count=320 if level is Level.CET4 else 420,
            generator_model=model_name,
            source_type="mock",
        )

    @staticmethod
    def _writing(level: Level, model_name: str) -> QuestionSet:
        min_words = 120 if level is Level.CET4 else 150
        prompt_lines = (
            [
                "Suppose your university is organizing a campus reading campaign.",
                "Write an essay to explain why regular reading still matters in the digital age.",
                "You may include personal observation, examples, and practical suggestions.",
            ]
            if level is Level.CET4
            else [
                "Suppose your university newspaper is collecting essays on technology and human learning.",
                "Write an essay discussing how AI tools can support but not replace independent thinking.",
                "You may refer to study habits, classroom practice, and long-term development.",
            ]
        )
        reference = (
            "Artificial intelligence has become part of daily learning, but students still need independent thinking to judge information, organize ideas, and make responsible decisions. AI tools may save time, offer examples, and reduce routine pressure, yet they cannot replace the habit of asking clear questions or the effort of building one’s own argument. In real study situations, students improve most when they use technology as support rather than as a substitute. For example, an AI summary may help a student review an article quickly, but deep understanding still depends on careful reading, note-taking, and personal reflection. If learners accept every generated answer without checking logic or evidence, they may become faster but weaker. Universities should therefore encourage balanced use of technology. Students need guidance on when AI can improve efficiency and when they must slow down to think for themselves. In this way, technology becomes a helpful partner, while independent thinking remains the true core of learning."
        )
        return QuestionSet(
            id=make_id("qs"),
            level=level,
            question_type=QuestionType.WRITING,
            title="Writing Task",
            topic="writing practice",
            passage=Passage(title="Task Notes", paragraphs=prompt_lines),
            questions=[],
            answer_key=[],
            analysis=AnalysisReport(
                overall_strategy="先确认立意，再组织段落结构，最后重点检查句法和连接。",
                overall_summary="写作任务强调切题、结构、语言准确性和表达完整度。",
                item_explanations=[],
                test_tips=[
                    "先列出两到三个主点，再展开例子。",
                    "注意段落之间的连接与逻辑推进。",
                    "预留时间检查拼写、时态和主谓一致。",
                ],
            ),
            vocabulary=[
                VocabularyItem("independent", "independent", level.value, "独立的"),
                VocabularyItem("reflection", "reflection", level.value, "反思"),
                VocabularyItem("substitute", "substitute", level.value, "替代品"),
            ],
            task_prompt=(
                f"For this part, you are allowed 30 minutes to write an essay of no less than {min_words} words "
                "according to the task given below."
            ),
            reference_answer=reference,
            rubric_focus=["content_relevance", "coherence", "grammar", "lexical_accuracy"],
            min_response_words=min_words,
            max_response_words=180 if level is Level.CET4 else 200,
            word_count=len(reference.split()),
            generator_model=model_name,
            source_type="mock",
        )

    @staticmethod
    def _translation(level: Level, model_name: str) -> QuestionSet:
        source_text = (
            "中国许多城市近年积极推动公共图书馆建设。除了提供借阅服务，这些图书馆还经常举办讲座、读书会和亲子活动。它们不仅丰富了市民的文化生活，也为社区交流提供了更加开放和舒适的公共空间。"
            if level is Level.CET4
            else "近年来，越来越多的中国城市开始重视历史街区的保护与更新。有关部门不仅修复老建筑，还努力改善居民生活条件，并通过文化活动吸引年轻人重新关注传统社区。这种做法有助于在现代化进程中保留城市记忆，也为地方经济带来了新的活力。"
        )
        reference = (
            "In recent years, many Chinese cities have actively promoted the development of public libraries. In addition to lending services, these libraries often organize lectures, reading clubs, and family activities. They have not only enriched the cultural life of residents but also provided a more open and comfortable public space for community interaction."
            if level is Level.CET4
            else "In recent years, more and more Chinese cities have begun to attach importance to the protection and renewal of historic neighborhoods. The relevant authorities have not only restored old buildings but also worked to improve residents’ living conditions and attract young people back to traditional communities through cultural activities. This approach helps preserve urban memory in the course of modernization and brings new vitality to local economies."
        )
        return QuestionSet(
            id=make_id("qs"),
            level=level,
            question_type=QuestionType.TRANSLATION,
            title="Translation Task",
            topic="translation practice",
            passage=Passage(title="汉译英原文", paragraphs=[source_text]),
            questions=[],
            answer_key=[],
            analysis=AnalysisReport(
                overall_strategy="先划分信息单位，再确定主干结构，最后优化英语表达的自然度。",
                overall_summary="翻译任务强调信息准确、表达通顺、语法稳定和词汇得体。",
                item_explanations=[],
                test_tips=[
                    "先保证事实和逻辑关系不丢失。",
                    "避免逐字硬译，优先还原自然英语句式。",
                    "检查时态、冠词、单复数和搭配。",
                ],
            ),
            vocabulary=[
                VocabularyItem("community", "community", level.value, "社区"),
                VocabularyItem("cultural", "cultural", level.value, "文化的"),
                VocabularyItem("vitality", "vitality", level.value, "活力"),
            ],
            task_prompt="For this part, you are allowed 30 minutes to translate the following Chinese passage into English.",
            reference_answer=reference,
            rubric_focus=["translation_accuracy", "translation_fluency", "grammar", "lexical_accuracy"],
            min_response_words=90 if level is Level.CET4 else 120,
            max_response_words=180 if level is Level.CET4 else 220,
            word_count=len(source_text),
            generator_model=model_name,
            source_type="mock",
        )
