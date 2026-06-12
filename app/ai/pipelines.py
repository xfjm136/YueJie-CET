from __future__ import annotations

import json
import re
from dataclasses import dataclass
from secrets import choice as secrets_choice
from typing import Any, Callable

from app.ai.client import DeepSeekClient
from app.ai.validators import CETQuestionValidator, QuestionSetValidationError
from app.constants import LEVEL_LABELS, QUESTION_TYPE_LABELS
from app.domain.enums import Level, QuestionType
from app.domain.schemas import (
    AnalysisReport,
    AttemptResult,
    ItemExplanation,
    Passage,
    Question,
    QuestionSet,
    ScoreDimension,
    SentenceRewrite,
    SentenceAnnotation,
    SubjectiveEvaluation,
    VocabularyItem,
    WordCorrection,
    AttemptQuestionResult,
    make_id,
)


@dataclass
class SubjectiveEvaluationError(Exception):
    errors: list[str]

    def __str__(self) -> str:
        return "；".join(self.errors)


class QuestionGenerationPipeline:
    def __init__(
        self,
        client: DeepSeekClient | None,
        default_model: str,
    ) -> None:
        self.client = client
        self.default_model = default_model
        self.validator = CETQuestionValidator()

    def generate(
        self,
        level: Level,
        question_type: QuestionType,
        slot: int | None = None,
        generation_context: str | dict[str, Any] | None = None,
        progress_callback: Callable[[str, str], None] | None = None,
    ) -> QuestionSet:
        if self.client is None:
            raise RuntimeError(
                "未配置 DEEPSEEK_API_KEY，无法进行真实 AI 出题。请在 /data/YueJie-CET/.env 中配置后重试。"
            )

        weakness_summary = self._extract_weakness_summary(generation_context)
        recent_topics = self._extract_recent_topics(generation_context)

        self._report(
            progress_callback,
            "blueprint",
            "已锁定四六级题型规范，正在整理题面蓝图与难度控制。",
        )
        blueprint = self._build_blueprint(
            level,
            question_type,
            slot,
            weakness_summary,
            recent_topics,
        )
        retry_errors: list[str] | None = None
        last_error: Exception | None = None
        max_generation_rounds = 4
        for generation_round in range(1, max_generation_rounds + 1):
            if generation_round > 1:
                self._report(
                    progress_callback,
                    "retry_generation",
                    f"上一轮仍未完全满足规范，正在重新生成第 {generation_round} 版题目。",
                )
            try:
                return self._generate_once(
                    level,
                    question_type,
                    slot,
                    blueprint,
                    progress_callback=progress_callback,
                    retry_errors=retry_errors,
                )
            except QuestionSetValidationError as exc:
                last_error = exc
                retry_errors = exc.errors
                if generation_round == max_generation_rounds:
                    raise
            except Exception as exc:
                last_error = exc
                retry_errors = [str(exc)]
                if generation_round == max_generation_rounds:
                    raise RuntimeError(f"AI 出题多轮重试后仍失败：{exc}") from exc
        if last_error is not None:
            raise last_error
        raise RuntimeError("AI 出题未能返回可验证的结果。")

    def _generate_once(
        self,
        level: Level,
        question_type: QuestionType,
        slot: int | None,
        blueprint: dict[str, Any],
        *,
        progress_callback: Callable[[str, str], None] | None = None,
        retry_errors: list[str] | None = None,
    ) -> QuestionSet:
        self._report(
            progress_callback,
            "generate_request",
            "正在向 DeepSeek 发送正式出题请求。",
        )
        payload = self.client.create_json_with_tool_schema(
            system_prompt=self._generation_system_prompt(),
            user_prompt=self._generation_user_prompt(
                level,
                question_type,
                slot,
                blueprint,
                retry_errors=retry_errors,
            ),
            tool_name="deliver_question_set",
            tool_description="Return one complete CET practice set as structured JSON.",
            parameters_schema=self._tool_parameters_schema(level, question_type),
            temperature=self._generation_temperature(question_type),
            max_tokens=self._max_tokens(question_type),
            request_timeout=self._request_timeout(question_type, repair=False),
        )
        payload = self._postprocess_payload(payload, level, question_type, slot)
        self._report(
            progress_callback,
            "validate",
            "首轮结果已返回，正在做结构校验与四六级要求校验。",
        )

        try:
            validated = self.validator.validate(payload, level, question_type, slot)
            self._report(
                progress_callback,
                "validated",
                "结构校验通过，正在整理题目数据。",
            )
            return self._question_set_from_payload(validated, level, question_type, slot, "ai")
        except QuestionSetValidationError as exc:
            self._report(
                progress_callback,
                "repair",
                f"检测到 {len(exc.errors)} 项结构偏差，正在按规则修复。",
            )
            return self._repair_until_valid(
                level,
                question_type,
                slot,
                blueprint,
                payload,
                exc.errors,
                progress_callback=progress_callback,
            )

    def _build_blueprint(
        self,
        level: Level,
        question_type: QuestionType,
        slot: int | None,
        weakness_summary: str | None,
        recent_topics: list[str] | None = None,
    ) -> dict[str, Any]:
        anti_repeat_topics = self._anti_repeat_topics(recent_topics)
        avoid_theme_keywords = self._avoid_theme_keywords(recent_topics)
        raw_topic_pool = self._topic_pool(level, question_type, slot)
        topic_pool = self._fresh_topic_pool(
            raw_topic_pool,
            anti_repeat_topics,
            avoid_theme_keywords,
        ) or raw_topic_pool
        return {
            "topic": secrets_choice(topic_pool),
            "genre": self._genre_for(level, question_type, slot),
            "register": self._register_for(level, question_type, slot),
            "source_material_hint": self._source_material_hint(level, question_type, slot),
            "exam_profile": self._exam_profile(level, question_type, slot),
            "passage_flow": self._passage_flow(level, question_type, slot),
            "prompt_style_anchors": self._prompt_style_anchors(level, question_type, slot),
            "authenticity_guardrails": self._authenticity_guardrails(level, question_type, slot),
            "writing_prompt_examples": self._writing_prompt_examples(level, question_type),
            "target_word_count": self._target_word_count(level, question_type),
            "word_count_guidance": self._strict_word_count_guidance(level, question_type),
            "skill_focus": self._skill_focus(level, question_type, slot),
            "difficulty_controls": self._difficulty_controls(level, question_type, slot),
            "structure_plan": self._structure_plan(level, question_type, slot),
            "option_strategy": self._option_strategy(level, question_type, slot),
            "writing_mode": self._writing_mode(level, question_type),
            "translation_domain": self._translation_domain(level, question_type),
            "weakness_focus": weakness_summary or "No weakness summary yet. Keep the skill mix balanced.",
            "question_id_pattern": self._question_id_pattern(question_type),
            "vocabulary_target_count": self._vocabulary_target_count(question_type),
            "analysis_style": "Chinese only, concise, evidence-based, and useful for CET review.",
            "anti_repeat_topics": anti_repeat_topics,
            "avoid_theme_keywords": avoid_theme_keywords,
            "novelty_requirement": self._novelty_requirement(
                recent_topics,
                avoid_theme_keywords,
                question_type,
                slot,
            ),
        }

    @staticmethod
    def _extract_weakness_summary(generation_context: str | dict[str, Any] | None) -> str | None:
        if generation_context is None:
            return None
        if isinstance(generation_context, str):
            return generation_context
        if isinstance(generation_context, dict):
            summary = generation_context.get("weakness_summary")
            if summary is None:
                return None
            return str(summary).strip() or None
        return str(generation_context).strip() or None

    @staticmethod
    def _extract_recent_topics(generation_context: str | dict[str, Any] | None) -> list[str]:
        if not isinstance(generation_context, dict):
            return []
        recent = generation_context.get("recent_topics", [])
        if not isinstance(recent, list):
            return []
        return [str(item).strip().lower() for item in recent if str(item).strip()]

    @staticmethod
    def _anti_repeat_topics(recent_topics: list[str] | None) -> list[str]:
        if not recent_topics:
            return []
        return [item.strip().lower() for item in recent_topics if item.strip()]

    @classmethod
    def _fresh_topic_pool(
        cls,
        topic_pool: list[str],
        anti_repeat_topics: list[str],
        avoid_theme_keywords: list[str],
    ) -> list[str]:
        blocked_topics = set(anti_repeat_topics)
        blocked_keywords = set(avoid_theme_keywords)
        fresh_topics = []
        for topic in topic_pool:
            normalized = topic.strip().lower()
            if normalized in blocked_topics:
                continue
            if blocked_keywords and len(cls._topic_keywords(normalized) & blocked_keywords) >= 2:
                continue
            fresh_topics.append(topic)
        return fresh_topics

    @staticmethod
    def _topic_keywords(text: str) -> set[str]:
        stopwords = {
            "about",
            "among",
            "around",
            "daily",
            "from",
            "into",
            "life",
            "people",
            "public",
            "social",
            "student",
            "students",
            "their",
            "through",
            "with",
            "young",
        }
        return {
            word
            for word in re.findall(r"[a-z]+", text.lower())
            if len(word) >= 4 and word not in stopwords
        }

    @classmethod
    def _avoid_theme_keywords(cls, recent_topics: list[str] | None) -> list[str]:
        if not recent_topics:
            return []
        keyword_counts: dict[str, int] = {}
        for topic in recent_topics:
            for keyword in cls._topic_keywords(topic):
                keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        ordered = sorted(keyword_counts.items(), key=lambda item: (-item[1], item[0]))
        return [keyword for keyword, _count in ordered[:12]]

    @staticmethod
    def _novelty_requirement(
        recent_topics: list[str] | None,
        avoid_theme_keywords: list[str],
        question_type: QuestionType,
        slot: int | None,
    ) -> str:
        if not recent_topics:
            return (
                "No recent-topic lock is active, but still avoid textbook-like recycling or lightly renamed repeats."
            )
        recent_preview = ", ".join(recent_topics[:4])
        keyword_preview = ", ".join(avoid_theme_keywords[:6]) or "none"
        type_hint = (
            "the other careful-reading passage in the same paper"
            if question_type is QuestionType.CAREFUL_READING and slot in {1, 2}
            else "other recently generated sets"
        )
        return (
            f"Avoid recycling recent themes from {type_hint}: {recent_preview}. "
            f"If related keywords appear again ({keyword_preview}), shift to a clearly different scene, source angle, and paragraph logic rather than merely renaming the same topic."
        )

    @staticmethod
    def _question_spec(level: Level, question_type: QuestionType, slot: int | None) -> str:
        if question_type is QuestionType.BANKED_CLOZE:
            word_spec = "CET4 200-250 words; CET6 250-300 words."
            return (
                f"{word_spec} Use 10 blanks in the passage marked [1]...[10]. "
                "Follow the real CET Section A format closely: provide exactly 15 shared options labeled A. to O. in a word bank following the passage. "
                "Each shared option must be one English word only, not a phrase or clause. "
                "Read-through coherence matters: the passage should read like one adapted English article rather than a coaching handout or list of advice. "
                "Targets and distractors should be CET-like in part of speech and collocation difficulty. "
                "Each of the 10 answers must use a different letter; no option may be reused."
            )
        if question_type is QuestionType.LONG_READING:
            word_spec = "CET4 850-1150 words; CET6 1050-1350 words."
            return (
                f"{word_spec} Provide exactly 10 statements for paragraph matching. "
                "Follow the real CET Section B format closely: each statement contains information given in one of the paragraphs, and the same paragraph may be chosen more than once. "
                "Paragraphs must be labeled A., B., C. ... with 10-14 total paragraphs. "
                "Each matching item must be an English statement, not a question, and must not include paragraph labels. "
                "The long passage should have clear article logic and paragraph functions rather than reading like disconnected mini-paragraphs."
            )
        if question_type is QuestionType.WRITING:
            min_words = 120 if level is Level.CET4 else 150
            max_words = 180 if level is Level.CET4 else 200
            style_hint = (
                "Favor a practical campus or social topic, usually in a CET4 exam style such as 'Suppose ... You are now to write ...' or another short official-looking instruction."
                if level is Level.CET4
                else "Favor a sentence-led CET6 style such as 'write an essay that begins with the sentence ...', or another short official exam instruction built around a statement or saying."
            )
            return (
                f"CET writing task. Write an English essay of at least {min_words} words but no more than {max_words} words. "
                f"{style_hint} "
                "Provide an exam-style task prompt in English, one brief content prompt paragraph, a high-scoring sample essay, and rubric focus tags. "
                "For CET4, strongly prefer prompts that look like short official exam instructions built around a campus or social situation, often beginning with 'Suppose ...' and continuing with 'You are now to write ...'. "
                "For CET4, prefer the compact official pattern of one situational sentence, one 'You are now to write ...' sentence, one time sentence, and one word-count sentence, all merged into one paragraph. "
                "Avoid adding extra debate lead-ins such as 'Some argue ... while others ...' unless the prompt absolutely requires that setup. "
                "For CET6, strongly prefer prompts that look like official sentence-led tasks such as 'write an essay that begins with the sentence ...', followed by brief official guidance. "
                "For CET6, prefer the official sequence: allowed 30 minutes; begins with the sentence '...'; make comments/cite examples/use personal experiences; word-count line; copy-the-sentence line. "
                "Keep the visible content prompt brief, like real CET tasks, rather than expanding it into a long teacher-style explanation. "
                "The visible content prompt should usually be one short paragraph rather than two separate blocks. "
                "Do not use explicit numbered outlines such as 1., 2., 3. in the prompt lines. "
                "Do not turn the prompt into a detailed scaffold with sub-questions; keep it compact like real CET writing tasks. "
                "No objective questions, options, or answer_key should be included."
            )
        if question_type is QuestionType.TRANSLATION:
            char_spec = "140-160 Chinese characters" if level is Level.CET4 else "180-200 Chinese characters"
            return (
                f"CET translation task. Provide one Chinese-to-English translation source passage of about {char_spec}. "
                "Also provide a polished reference translation in English, rubric focus tags, and response length guidance. "
                "No objective questions, options, or answer_key should be included."
            )
        word_spec = "CET4 300-350 words; CET6 400-450 words."
        slot_hint = f"This is careful reading slot {slot}. " if slot else ""
        if level is Level.CET4 and slot == 1:
            slot_style = (
                "Model it as CET4 Section C Passage One from an adapted original English source. "
                "Real papers show broad themes such as health, science, education, work, food, public life, or social observation. "
                "Keep it a little more fact-led and explanatory than Passage Two, but do not force it into a formal lab-report mold. "
                "Use common real-paper stem families such as 'What do we learn from the passage about ...?', 'What does the author say about ...?', 'Why do ...?', or 'What does the author suggest near the end of the passage?'. "
            )
        elif level is Level.CET4 and slot == 2:
            slot_style = (
                "Model it as CET4 Section C Passage Two from an adapted original English source. "
                "Real papers often use themes such as work, culture, psychology, consumption, art, education, technology, or everyday social issues. "
                "Keep it slightly more interpretive or opinion-colored than Passage One, but still text-based and non-literary. "
                "Use common real-paper stem families such as 'What often happens when ...?', 'What is ...?', 'What can be inferred ...?', or 'What does the passage suggest ...?'. "
            )
        elif level is Level.CET6 and slot == 1:
            slot_style = (
                "Model it as CET6 Section C Passage One from an adapted original English source. "
                "Real papers show broad themes such as society, work, public policy, psychology, education, health, economics, or personal development rather than only business topics. "
                "Keep it denser and more analytical than CET4, but not locked to a single domain formula. "
                "Use common real-paper stem families such as 'What often happens when ...?', 'What does the author say about ...?', 'What can we infer ...?', or 'What does ... indicate?'. "
            )
        else:
            slot_style = (
                "Model it as CET6 Section C Passage Two from an adapted original English source. "
                "Real papers often use themes such as social issues, ethics, psychology, technology, economy, history, or culture, usually with stronger reflection or argument than Passage One. "
                "Keep it close to a real magazine, newspaper, or commentary passage instead of an abstract AI-generated lecture. "
                "Use common real-paper stem families such as 'What can be inferred ...?', 'Why does the author mention ...?', 'What does the passage imply ...?', or 'What is the author's attitude ...?'. "
            )
        return (
            f"{slot_hint}{word_spec} Provide exactly 5 four-option multiple-choice questions. "
            f"{slot_style}"
            "Question stems may be either direct questions or unfinished statements. "
            "Test realistic CET skills such as main idea, important detail, inference, vocabulary in context, and author attitude when natural, but do not force exactly one of each. "
            "Question order should broadly follow passage logic. "
            "Question stems must be English-only; avoid trick options like All of the above or None of the above."
        )

    def _generation_system_prompt(self) -> str:
        return (
            "You are a CET reading item generator. "
            "Return strict json only. No markdown fences. "
            "The final output must be directly machine-readable and fully self-contained. "
            "Passage, titles, questions, and answer options must stay in English. "
            "Analysis and vocabulary explanations must be in Chinese. "
            "Keep the output exam-like, concise, structurally exact, and close to adapted CET source materials rather than blogs, motivational essays, or fictional stories."
        )

    def _generation_user_prompt(
        self,
        level: Level,
        question_type: QuestionType,
        slot: int | None,
        blueprint: dict[str, Any],
        retry_errors: list[str] | None = None,
    ) -> str:
        level_label = LEVEL_LABELS[level]
        type_label = QUESTION_TYPE_LABELS[question_type]
        retry_note = ""
        if retry_errors:
            retry_note = (
                "Retry guidance from the previous failed attempt:\n"
                f"{json.dumps(retry_errors, ensure_ascii=False)}\n"
                "Be stricter about the failed constraints this time while keeping the set natural and exam-like.\n"
            )
        return (
            f"Generate one {level_label} {type_label} practice set from this blueprint:\n"
            f"{json.dumps(blueprint, ensure_ascii=False)}\n\n"
            f"{retry_note}"
            f"Mandatory spec:\n{self._question_spec(level, question_type, slot)}\n"
            "Output rules:\n"
            "- Treat blueprint.exam_profile, blueprint.passage_flow, blueprint.prompt_style_anchors, and blueprint.authenticity_guardrails as hard style targets rather than optional inspiration.\n"
            "- Keep the passage natural and exam-like.\n"
            "- Keep the style close to CET source materials such as news features, opinion pieces, science explanations, educational commentary, lecture-derived exposition, or general academic reading.\n"
            "- Objective passages should feel like adapted original English materials, not like a student's essay, chat response, diary entry, marketing copy, or story opening.\n"
            "- Background knowledge must be common or recoverable from textual clues.\n"
            "- Ensure every answer is uniquely supported by the passage.\n"
            "- Avoid giving away answers through obvious grammatical mismatch.\n"
            "- The validator will reject the set if passage word count, question count, answer count, or paragraph labeling is off.\n"
            f"- Word-count guidance: {self._strict_word_count_guidance(level, question_type)}\n"
            "- Keep the title short and neutral.\n"
            "- Use question ids exactly as q1, q2, q3 ... in order.\n"
            "- Keep analysis concise: overall_summary in 2-4 Chinese sentences, test_tips exactly 3 items, item_explanations focused on evidence and method.\n"
            "- Provide exactly the requested number of vocabulary items, not more.\n"
            "- Only analysis.* and vocabulary.meaning_zh may be in Chinese; the rest must be in English.\n"
            "- Do not place Chinese characters in title, topic, passage, question prompts, shared options, answer options, or example_en.\n"
            "- Avoid direct dialogue, heavy quotation, dramatic scene-setting, fantasy, or overtly inspirational tone in objective passages.\n"
            "- For banked cloze, the 15 options must be shared options only, not repeated per blank.\n"
            "- For banked cloze, every shared option must be exactly one English word, not a phrase.\n"
            "- For banked cloze, each answer letter may be used only once.\n"
            "- For banked cloze, the passage should read like one adapted article with local cohesive clues, not like ten disconnected vocabulary exercise sentences.\n"
            "- For long reading, statements must be paraphrases, not direct copies of paragraph openings.\n"
            "- For long reading, every item prompt must be a statement rather than a question and must not include A./B./C. labels.\n"
            "- For long reading, most statements should be compact information-bearing claims rather than 'The passage says ...' summaries.\n"
            "- For careful reading, distribute correct options naturally and make distractors plausible.\n"
            "- For careful reading, keep four options parallel in grammar and length, and never use all/none of the above.\n"
            "- For careful reading, follow the real-paper convention that both passages belong to Section C; treat slot 1 and slot 2 as Passage One and Passage Two inside the same section rather than inventing a new section label.\n"
            "- For careful reading, stems should sound like real CET questions or unfinished statements, not like topic-specific template slogans.\n"
            "- For writing, produce a CET-style prompt format such as a short situational instruction, a survey/opinion task, a quoted statement, or a sentence-led argumentative task as appropriate to the level.\n"
            "- For CET4 writing, the prompt should look closer to official exam wording such as 'Suppose ... You are now to write ...' than to abstract philosophical debate.\n"
            "- For CET6 writing, the prompt should often look like an official sentence-led task built around a quoted sentence, followed by very brief guidance.\n"
            "- The visible content prompt should normally be one short paragraph only.\n"
            "- Do not use explicit numbered outlines like 1., 2., 3. or detailed multi-point Chinese-style writing hints in the prompt lines.\n"
            "- Keep writing prompts compact. Avoid multi-step scaffolds or teacher-style classroom instructions.\n"
            "- For translation, provide a Chinese source passage on Chinese culture, history, society, development, education, ecology, or technology as appropriate to the level.\n"
            "- Do not write in a dramatic, fictional, or conversational blog style.\n"
            "- Novelty rule: do not lightly rename a recent topic; if a nearby topic reappears, change the source angle, scenario, and paragraph logic enough to feel like a different authentic passage.\n"
            "- Use concise Chinese explanations that point back to textual evidence or reasoning path.\n"
            f"Question-type details:\n{self._question_type_details(question_type, level, slot)}\n"
            f"Structured output contract:\n{self._response_contract(question_type)}\n"
            "Return through the provided tool schema only. Do not add markdown fences or extra prose."
        )

    def _model_name(self) -> str:
        return self.client.model if self.client is not None else self.default_model

    def _repair_system_prompt(self) -> str:
        return (
            "You repair CET item JSON. "
            "Return strict json only. "
            "Fix the structure and content so it satisfies the validation errors exactly."
        )

    def _repair_user_prompt(
        self,
        level: Level,
        question_type: QuestionType,
        slot: int | None,
        blueprint: dict[str, Any],
        payload: dict[str, Any],
        errors: list[str],
        repair_round: int = 1,
    ) -> str:
        return (
            f"Repair round {repair_round}.\n"
            f"Repair this {LEVEL_LABELS[level]} {QUESTION_TYPE_LABELS[question_type]} JSON.\n"
            f"Spec:\n{self._question_spec(level, question_type, slot)}\n"
            f"Blueprint:\n{json.dumps(blueprint, ensure_ascii=False)}\n"
            f"Validation errors:\n{json.dumps(errors, ensure_ascii=False)}\n"
            f"Targeted repair guidance:\n{self._targeted_repair_guidance(level, question_type, errors)}\n"
            f"Structured output contract:\n{self._response_contract(question_type)}\n"
            "Original JSON:\n"
            f"{json.dumps(payload, ensure_ascii=False)}\n"
            "Return a corrected full JSON object only. Preserve valid content and change only what is needed. "
            "If a word-count error is listed, expand or trim the passage itself until it is inside the required range while keeping the passage coherent. "
            "During repair, keep the final set aligned with blueprint.exam_profile, blueprint.passage_flow, blueprint.prompt_style_anchors, and blueprint.authenticity_guardrails."
        )

    def _repair_until_valid(
        self,
        level: Level,
        question_type: QuestionType,
        slot: int | None,
        blueprint: dict[str, Any],
        payload: dict[str, Any],
        errors: list[str],
        progress_callback: Callable[[str, str], None] | None = None,
    ) -> QuestionSet:
        current_payload = payload
        current_errors = errors
        last_exception: QuestionSetValidationError | None = None
        for repair_round in range(1, 4):
            self._report(
                progress_callback,
                "repair",
                f"正在执行第 {repair_round} 轮结构修复与复检。",
            )
            repaired_payload = self.client.create_json_with_tool_schema(
                system_prompt=self._repair_system_prompt(),
                user_prompt=self._repair_user_prompt(
                    level,
                    question_type,
                    slot,
                    blueprint,
                    current_payload,
                    current_errors,
                    repair_round=repair_round,
                ),
                tool_name="deliver_question_set",
                tool_description="Return one corrected CET practice set as structured JSON.",
                parameters_schema=self._tool_parameters_schema(level, question_type),
                temperature=0.1,
                max_tokens=self._max_tokens(question_type),
                request_timeout=self._request_timeout(question_type, repair=True),
            )
            repaired_payload = self._postprocess_payload(
                repaired_payload,
                level,
                question_type,
                slot,
            )
            try:
                self._report(
                    progress_callback,
                    "validate",
                    f"第 {repair_round} 轮修复已返回，正在重新校验。",
                )
                validated = self.validator.validate(repaired_payload, level, question_type, slot)
                self._report(
                    progress_callback,
                    "validated",
                    f"第 {repair_round} 轮修复后已通过校验。",
                )
                return self._question_set_from_payload(
                    validated,
                    level,
                    question_type,
                    slot,
                    "ai_repaired",
                )
            except QuestionSetValidationError as exc:
                current_payload = repaired_payload
                current_errors = exc.errors
                last_exception = exc
        if last_exception is not None:
            raise last_exception
        raise RuntimeError("AI 修复失败，未获得可验证的题目结构。")

    @staticmethod
    def _report(
        progress_callback: Callable[[str, str], None] | None,
        phase: str,
        message: str,
    ) -> None:
        if progress_callback is not None:
            progress_callback(phase, message)

    def _postprocess_payload(
        self,
        payload: dict[str, Any],
        level: Level,
        question_type: QuestionType,
        slot: int | None,
    ) -> dict[str, Any]:
        normalized = dict(payload)
        normalized["title"] = str(normalized.get("title", "")).strip()
        normalized["topic"] = str(normalized.get("topic", "")).strip()
        normalized["task_prompt"] = str(normalized.get("task_prompt", "")).strip()
        normalized["reference_answer"] = str(normalized.get("reference_answer", "")).strip()
        normalized["rubric_focus"] = [
            str(item).strip() for item in normalized.get("rubric_focus", []) if str(item).strip()
        ]
        normalized["min_response_words"] = int(normalized.get("min_response_words", 0) or 0)
        normalized["max_response_words"] = int(normalized.get("max_response_words", 0) or 0)
        if question_type in {
            QuestionType.BANKED_CLOZE,
            QuestionType.LONG_READING,
            QuestionType.CAREFUL_READING,
        }:
            normalized["task_prompt"] = ""
            normalized["reference_answer"] = ""
            normalized["rubric_focus"] = []
            normalized["min_response_words"] = 0
            normalized["max_response_words"] = 0
        if question_type is QuestionType.WRITING:
            normalized["task_prompt"] = self._normalize_writing_task_prompt(
                normalized.get("task_prompt", ""),
                normalized.get("title", ""),
                level,
            )

        passage = dict(normalized.get("passage", {}))
        paragraphs = [str(item).strip() for item in passage.get("paragraphs", []) if str(item).strip()]
        if question_type is QuestionType.LONG_READING:
            paragraphs = self._normalize_long_reading_paragraphs(paragraphs)
        if question_type is QuestionType.WRITING:
            paragraphs = self._normalize_writing_prompt_lines(
                paragraphs,
                normalized.get("title", ""),
                level,
            )
        passage["paragraphs"] = paragraphs
        passage["title"] = str(passage.get("title", normalized.get("title", ""))).strip()
        normalized["passage"] = passage

        questions = []
        for index, item in enumerate(normalized.get("questions", []), start=1):
            question = dict(item) if isinstance(item, dict) else {}
            question["id"] = f"q{index}"
            question["prompt"] = self._normalized_prompt(question, question_type, index)
            question["options"] = self._normalize_question_options(question, question_type)
            question["skill_tag"] = self._normalized_skill_tag(
                question,
                level,
                question_type,
                index,
                slot,
            )
            questions.append(question)
        normalized["questions"] = questions

        normalized["answer_key"] = self._normalize_answer_key(
            normalized.get("answer_key", []),
            question_type,
        )
        normalized["shared_options"] = self._normalize_shared_options(
            normalized.get("shared_options", []),
            question_type,
        )

        analysis = dict(normalized.get("analysis", {}))
        explanations = []
        for index, item in enumerate(analysis.get("item_explanations", []), start=1):
            explanation = dict(item) if isinstance(item, dict) else {}
            explanation["question_id"] = f"q{index}"
            explanation["correct_answer"] = self._normalize_single_answer_token(
                explanation.get("correct_answer", "")
            )
            explanation["skill_tag"] = self._normalized_skill_tag(
                {"prompt": questions[index - 1]["prompt"], "skill_tag": explanation.get("skill_tag", "")},
                level,
                question_type,
                index,
                slot,
            ) if index - 1 < len(questions) else "general"
            explanations.append(explanation)
        analysis["item_explanations"] = explanations
        test_tips = [str(item).strip() for item in analysis.get("test_tips", []) if str(item).strip()]
        if len(test_tips) > 3:
            test_tips = test_tips[:3]
        elif 0 < len(test_tips) < 3:
            for tip in self._default_test_tips(question_type):
                if len(test_tips) >= 3:
                    break
                if tip not in test_tips:
                    test_tips.append(tip)
        analysis["test_tips"] = test_tips
        normalized["analysis"] = analysis

        vocab_limit = self._vocabulary_target_count(question_type)
        normalized["vocabulary"] = list(normalized.get("vocabulary", []))[:vocab_limit]
        return normalized

    @staticmethod
    def _normalized_prompt(question: dict[str, Any], question_type: QuestionType, index: int) -> str:
        if question_type is QuestionType.BANKED_CLOZE:
            return f"Blank {index}"
        return str(question.get("prompt", "")).strip()

    @staticmethod
    def _normalize_answer_key(answer_key: list[Any], question_type: QuestionType) -> list[str]:
        normalized = [
            QuestionGenerationPipeline._normalize_single_answer_token(item)
            for item in answer_key
        ]
        if question_type is QuestionType.CAREFUL_READING:
            return [item[:1] for item in normalized]
        return [item[:1] if item else "" for item in normalized]

    @staticmethod
    def _normalize_single_answer_token(raw: Any) -> str:
        text = str(raw).strip().upper().replace(".", "").replace(")", "")
        match = re.search(r"[A-O]", text)
        return match.group(0) if match else text

    @staticmethod
    def _normalize_shared_options(shared_options: list[Any], question_type: QuestionType) -> list[str]:
        if question_type is not QuestionType.BANKED_CLOZE:
            return []
        letters = [chr(ord("A") + index) for index in range(15)]
        normalized: list[str] = []
        for index, item in enumerate(shared_options[:15]):
            raw = str(item).strip()
            match = re.match(r"^[A-Oa-o][\.\)]\s*(.+)$", raw)
            content = match.group(1).strip() if match else raw
            normalized.append(f"{letters[index]}. {content}")
        return normalized

    def _normalize_question_options(
        self,
        question: dict[str, Any],
        question_type: QuestionType,
    ) -> list[str]:
        if question_type is not QuestionType.CAREFUL_READING:
            return []
        labels = ["A", "B", "C", "D"]
        options = []
        for index, raw in enumerate(question.get("options", [])[:4]):
            text = str(raw).strip()
            match = re.match(r"^[A-Da-d][\.\)]\s*(.+)$", text)
            content = match.group(1).strip() if match else text
            options.append(f"{labels[index]}. {content}")
        return options

    def _normalized_skill_tag(
        self,
        question: dict[str, Any],
        level: Level,
        question_type: QuestionType,
        index: int,
        slot: int | None,
    ) -> str:
        raw = str(question.get("skill_tag", "")).strip().lower().replace(" ", "_")
        mapping = {
            "main_idea": "main_idea",
            "mainidea": "main_idea",
            "detail": "detail",
            "detail_question": "detail",
            "matching": "matching",
            "scan": "matching",
            "inference": "inference",
            "logic": "logic",
            "collocation": "collocation",
            "context_clue": "context clue",
            "context": "context clue",
            "vocabulary": "vocabulary_in_context" if question_type is QuestionType.CAREFUL_READING else "vocabulary",
            "vocabulary_in_context": "vocabulary_in_context",
            "vocabulary_context": "vocabulary_in_context",
            "attitude": "attitude",
            "attitude_tone": "attitude",
            "tone": "attitude",
        }
        if question_type is QuestionType.LONG_READING:
            return "matching"
        if question_type is QuestionType.BANKED_CLOZE:
            return mapping.get(raw, "logic" if index in {4, 7, 10} else "vocabulary")
        normalized = mapping.get(raw)
        if normalized in {"main_idea", "detail", "inference", "vocabulary_in_context", "attitude"}:
            return normalized
        prompt = str(question.get("prompt", "")).lower()
        if "main idea" in prompt or "mainly about" in prompt:
            return "main_idea"
        if "attitude" in prompt or "tone" in prompt:
            return "attitude"
        if "infer" in prompt or "imply" in prompt:
            return "inference"
        if "word" in prompt or "phrase" in prompt or "most likely means" in prompt:
            return "vocabulary_in_context"
        if "according to" in prompt or "why does" in prompt or "which of the following" in prompt:
            return "detail"
        ordered = ["main_idea", "detail", "inference", "vocabulary_in_context", "attitude"]
        if question_type is QuestionType.CAREFUL_READING and slot == 1 and level is Level.CET4:
            ordered = ["detail", "detail", "main_idea", "detail", "inference"]
        elif question_type is QuestionType.CAREFUL_READING and slot == 2 and level is Level.CET4:
            ordered = ["detail", "detail", "inference", "attitude", "detail"]
        elif question_type is QuestionType.CAREFUL_READING and slot == 1 and level is Level.CET6:
            ordered = ["detail", "inference", "detail", "main_idea", "inference"]
        elif question_type is QuestionType.CAREFUL_READING and slot == 2 and level is Level.CET6:
            ordered = ["inference", "detail", "inference", "attitude", "vocabulary_in_context"]
        return ordered[min(index - 1, len(ordered) - 1)]

    @staticmethod
    def _normalize_long_reading_paragraphs(paragraphs: list[str]) -> list[str]:
        normalized: list[str] = []
        expected_letters = [chr(ord("A") + index) for index in range(len(paragraphs))]
        for index, paragraph in enumerate(paragraphs):
            raw = paragraph.strip()
            match = re.match(r"^([A-Z])[\.\):]\s*(.+)$", raw)
            content = match.group(2).strip() if match else raw
            normalized.append(f"{expected_letters[index]}. {content}")
        return normalized

    @staticmethod
    def _normalize_writing_prompt_lines(
        paragraphs: list[str],
        title: str,
        level: Level,
    ) -> list[str]:
        cleaned = []
        title_normalized = title.strip().lower()
        for paragraph in paragraphs:
            line = re.sub(r"^\s*\d+\s*[\.\):\-]\s*", "", paragraph).strip()
            if line and line.strip().lower() != title_normalized:
                cleaned.append(line)
        min_words = 120 if level is Level.CET4 else 150
        max_words = 180 if level is Level.CET4 else 200
        if not cleaned:
            topic = title.strip() or "the following topic"
            return [
                f"Write an essay on {topic}. You should write at least {min_words} words but no more than {max_words} words."
            ]
        content = " ".join(cleaned).strip()
        if level is Level.CET4 and "suppose" in content.lower() and "you are now to write" in content.lower():
            content = re.sub(
                r"\bSome argue .*?(?:\.|;)\s*",
                "",
                content,
                flags=re.IGNORECASE,
            ).strip()
        content = re.sub(
            r"\bWrite about \d+ words but no more than \d+ words\.?",
            "",
            content,
            flags=re.IGNORECASE,
        ).strip()
        content = re.sub(
            r"\bWrite about \d+ words\.?",
            "",
            content,
            flags=re.IGNORECASE,
        ).strip()
        content = re.sub(
            r"\bWrite an essay of about \d+\s*[–-]\s*\d+ words\.?",
            "",
            content,
            flags=re.IGNORECASE,
        ).strip()
        if "you should write" not in content.lower():
            content = (
                f"{content} You should write at least {min_words} words but no more than {max_words} words."
            ).strip()
        if level is Level.CET4 and "you will have 30 minutes to write the essay" not in content.lower():
            content = (
                content.replace(
                    "You should write at least",
                    "You will have 30 minutes to write the essay. You should write at least",
                    1,
                )
                if "you should write at least" in content.lower()
                else f"{content} You will have 30 minutes to write the essay."
            ).strip()
        if level is Level.CET4:
            content = re.sub(
                r"\bYou are allowed 30 minutes(?: for this task)?\.",
                "You will have 30 minutes to write the essay.",
                content,
                flags=re.IGNORECASE,
            ).strip()
        content = re.sub(
            r"(You will have 30 minutes(?: for this task)?\.)\s*(You will have 30 minutes to write the essay\.)",
            r"\2",
            content,
            flags=re.IGNORECASE,
        ).strip()
        content = re.sub(
            r"(You will have 30 minutes to write the essay\.)\s*(You will have 30 minutes to write the essay\.)",
            r"\1",
            content,
            flags=re.IGNORECASE,
        ).strip()
        if (
            level is Level.CET6
            and "begins with the sentence" in content.lower()
            and "you can make comments, cite examples or use your personal experiences to develop your essay" not in content.lower()
        ):
            content = (
                f"{content} You can make comments, cite examples or use your personal experiences to develop your essay."
            ).strip()
        if (
            level is Level.CET6
            and "begins with the sentence" in content.lower()
            and "you should copy the sentence given in quotes" not in content.lower()
        ):
            content = (
                f"{content} You should copy the sentence given in quotes at the beginning of your essay."
            ).strip()
        return [content]

    @staticmethod
    def _normalize_writing_task_prompt(raw: Any, title: str, level: Level) -> str:
        return "Directions:"

    @staticmethod
    def _default_test_tips(question_type: QuestionType) -> list[str]:
        if question_type is QuestionType.BANKED_CLOZE:
            return [
                "先判断词性，再结合固定搭配筛选选项。",
                "优先处理确定度高的空位，减少干扰项影响。",
                "注意上下文逻辑和指代关系，不要孤立看单句。",
            ]
        if question_type is QuestionType.LONG_READING:
            return [
                "先扫题干关键词，再回原文找同义改写。",
                "优先看段首句和转折句，快速判断段落主旨。",
                "若两段都像答案，重点比较信息范围和语义重心。",
            ]
        if question_type is QuestionType.WRITING:
            return [
                "先确定立意和段落结构，再展开主要论点。",
                "优先保证句子准确和衔接自然，不必为了复杂而复杂。",
                "写完后重点检查拼写、时态、冠词和主谓一致。",
            ]
        if question_type is QuestionType.TRANSLATION:
            return [
                "先划清信息单位，避免漏译或错译关键关系。",
                "优先保证表达自然完整，不要逐字硬译。",
                "检查时态、冠词、单复数和固定搭配。",
            ]
        return [
            "主旨题先抓首段和尾段，细节题一定回文定位。",
            "推断题关注转折、让步和作者评价性措辞。",
            "词义题优先结合上下文逻辑，不要只看单词本身。",
        ]

    def _tool_parameters_schema(
        self,
        level: Level,
        question_type: QuestionType,
    ) -> dict[str, Any]:
        skill_enums = self._skill_enums(question_type)
        is_objective = question_type in {
            QuestionType.BANKED_CLOZE,
            QuestionType.LONG_READING,
            QuestionType.CAREFUL_READING,
        }
        if question_type is QuestionType.BANKED_CLOZE:
            question_schema = {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "prompt": {"type": "string"},
                    "skill_tag": {"type": "string", "enum": skill_enums},
                },
                "required": ["id", "prompt", "skill_tag"],
                "additionalProperties": False,
            }
        elif question_type is QuestionType.LONG_READING:
            question_schema = {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "prompt": {"type": "string"},
                    "skill_tag": {"type": "string", "enum": skill_enums},
                },
                "required": ["id", "prompt", "skill_tag"],
                "additionalProperties": False,
            }
        else:
            question_schema = {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "prompt": {"type": "string"},
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "skill_tag": {"type": "string", "enum": skill_enums},
                },
                "required": ["id", "prompt", "options", "skill_tag"],
                "additionalProperties": False,
            }
        explanation_schema = {
            "type": "object",
            "properties": {
                "question_id": {"type": "string"},
                "correct_answer": {"type": "string"},
                "explanation": {"type": "string"},
            },
            "required": ["question_id", "correct_answer", "explanation"],
            "additionalProperties": False,
        }
        vocabulary_schema = {
            "type": "object",
            "properties": {
                "lemma": {"type": "string"},
                "surface_form": {"type": "string"},
                "level_hint": {"type": "string", "enum": [level.value]},
                "meaning_zh": {"type": "string"},
                "example_en": {"type": "string"},
            },
            "required": [
                "lemma",
                "surface_form",
                "level_hint",
                "meaning_zh",
                "example_en",
            ],
            "additionalProperties": False,
        }
        shared_options_schema = {
            "type": "array",
            "items": {"type": "string"},
        }
        if question_type is QuestionType.BANKED_CLOZE:
            shared_options_schema["minItems"] = 15
            shared_options_schema["maxItems"] = 15
        else:
            shared_options_schema["maxItems"] = 0
        if is_objective:
            task_prompt_schema: dict[str, Any] = {"type": "string", "enum": [""]}
            reference_answer_schema: dict[str, Any] = {"type": "string", "enum": [""]}
            rubric_focus_schema: dict[str, Any] = {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 0,
                "maxItems": 0,
            }
            min_response_words_schema: dict[str, Any] = {"type": "integer", "enum": [0]}
            max_response_words_schema: dict[str, Any] = {"type": "integer", "enum": [0]}
        else:
            task_prompt_schema = {"type": "string"}
            reference_answer_schema = {"type": "string"}
            rubric_focus_schema = {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 4,
                "maxItems": 6,
            }
            min_response_words_schema = {"type": "integer"}
            max_response_words_schema = {"type": "integer"}
        answer_item_enum = (
            ["A", "B", "C", "D"] if question_type is QuestionType.CAREFUL_READING else None
        )
        answer_items: dict[str, Any] = {"type": "string"}
        if answer_item_enum is not None:
            answer_items["enum"] = answer_item_enum
        if question_type in {QuestionType.WRITING, QuestionType.TRANSLATION}:
            question_count = 0
        elif question_type is QuestionType.CAREFUL_READING:
            question_count = 5
        else:
            question_count = 10
        vocab_count = self._vocabulary_target_count(question_type)
        return {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "topic": {"type": "string"},
                "task_prompt": task_prompt_schema,
                "reference_answer": reference_answer_schema,
                "rubric_focus": rubric_focus_schema,
                "min_response_words": min_response_words_schema,
                "max_response_words": max_response_words_schema,
                "shared_options": shared_options_schema,
                "passage": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "paragraphs": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["title", "paragraphs"],
                    "additionalProperties": False,
                },
                "questions": {
                    "type": "array",
                    "items": question_schema,
                    "minItems": question_count,
                    "maxItems": question_count,
                },
                "answer_key": {
                    "type": "array",
                    "items": answer_items,
                    "minItems": question_count,
                    "maxItems": question_count,
                },
                "analysis": {
                    "type": "object",
                    "properties": {
                        "overall_strategy": {"type": "string"},
                        "overall_summary": {"type": "string"},
                        "item_explanations": {
                            "type": "array",
                            "items": explanation_schema,
                            "minItems": question_count,
                            "maxItems": question_count,
                        },
                        "test_tips": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 3,
                            "maxItems": 3,
                        },
                    },
                    "required": [
                        "overall_strategy",
                        "overall_summary",
                        "item_explanations",
                        "test_tips",
                    ],
                    "additionalProperties": False,
                },
                "vocabulary": {
                    "type": "array",
                    "items": vocabulary_schema,
                    "minItems": max(3, vocab_count),
                    "maxItems": vocab_count,
                },
            },
            "required": [
                "title",
                "topic",
                "task_prompt",
                "reference_answer",
                "rubric_focus",
                "min_response_words",
                "max_response_words",
                "shared_options",
                "passage",
                "questions",
                "answer_key",
                "analysis",
                "vocabulary",
            ],
            "additionalProperties": False,
        }

    @staticmethod
    def _response_contract(question_type: QuestionType) -> str:
        common = (
            "- top-level keys: title, topic, task_prompt, reference_answer, rubric_focus, "
            "min_response_words, max_response_words, shared_options, passage, questions, answer_key, analysis, vocabulary.\n"
            "- passage must contain title and paragraphs.\n"
            "- analysis must contain overall_strategy, overall_summary, item_explanations, and exactly 3 test_tips.\n"
            "- each item_explanation must contain question_id, correct_answer, explanation, and skill_tag.\n"
        )
        if question_type in {
            QuestionType.BANKED_CLOZE,
            QuestionType.LONG_READING,
            QuestionType.CAREFUL_READING,
        }:
            return (
                common
                + "- for objective sets, task_prompt and reference_answer must be empty strings.\n"
                + "- for objective sets, rubric_focus must be an empty array and min_response_words/max_response_words must be 0."
            )
        if question_type is QuestionType.WRITING:
            return (
                common
                + "- writing must keep shared_options empty, questions empty, and answer_key empty.\n"
                + "- writing must provide a CET-style prompt paragraph and a high-scoring sample essay."
            )
        return (
            common
            + "- translation must keep shared_options empty, questions empty, and answer_key empty.\n"
            + "- translation must provide one Chinese source paragraph and one natural English reference translation."
        )

    @staticmethod
    def _skill_enums(question_type: QuestionType) -> list[str]:
        if question_type is QuestionType.BANKED_CLOZE:
            return ["vocabulary", "logic", "collocation", "context clue"]
        if question_type is QuestionType.LONG_READING:
            return ["matching"]
        if question_type is QuestionType.WRITING:
            return ["grammar", "lexical_accuracy", "coherence", "content_relevance"]
        if question_type is QuestionType.TRANSLATION:
            return ["translation_accuracy", "translation_fluency", "grammar", "lexical_accuracy"]
        return [
            "main_idea",
            "detail",
            "inference",
            "vocabulary_in_context",
            "attitude",
        ]

    @staticmethod
    def _max_tokens(question_type: QuestionType) -> int:
        if question_type is QuestionType.LONG_READING:
            return 5600
        if question_type is QuestionType.WRITING:
            return 2600
        if question_type is QuestionType.TRANSLATION:
            return 2400
        if question_type is QuestionType.CAREFUL_READING:
            return 3200
        return 2600

    @staticmethod
    def _generation_temperature(question_type: QuestionType) -> float:
        if question_type is QuestionType.BANKED_CLOZE:
            return 0.32
        if question_type is QuestionType.WRITING:
            return 0.42
        if question_type is QuestionType.TRANSLATION:
            return 0.34
        if question_type is QuestionType.CAREFUL_READING:
            return 0.36
        return 0.4

    @staticmethod
    def _request_timeout(question_type: QuestionType, *, repair: bool) -> float:
        if question_type is QuestionType.LONG_READING:
            return 190.0 if repair else 240.0
        if question_type is QuestionType.CAREFUL_READING:
            return 150.0 if repair else 200.0
        if question_type is QuestionType.BANKED_CLOZE:
            return 140.0 if repair else 190.0
        if question_type is QuestionType.WRITING:
            return 140.0 if repair else 180.0
        return 140.0 if repair else 180.0

    @staticmethod
    def _target_word_count(level: Level, question_type: QuestionType) -> int:
        if question_type is QuestionType.BANKED_CLOZE:
            return 230 if level is Level.CET4 else 280
        if question_type is QuestionType.LONG_READING:
            return 1000 if level is Level.CET4 else 1200
        if question_type is QuestionType.WRITING:
            return 150 if level is Level.CET4 else 180
        if question_type is QuestionType.TRANSLATION:
            return 150 if level is Level.CET4 else 190
        return 325 if level is Level.CET4 else 425

    @staticmethod
    def _word_count_bounds(level: Level, question_type: QuestionType) -> tuple[int, int]:
        if question_type is QuestionType.BANKED_CLOZE:
            return (200, 250) if level is Level.CET4 else (250, 300)
        if question_type is QuestionType.LONG_READING:
            return (850, 1150) if level is Level.CET4 else (1050, 1350)
        if question_type is QuestionType.WRITING:
            return (120, 180) if level is Level.CET4 else (150, 200)
        if question_type is QuestionType.TRANSLATION:
            return (140, 160) if level is Level.CET4 else (180, 200)
        return (300, 350) if level is Level.CET4 else (400, 450)

    def _strict_word_count_guidance(self, level: Level, question_type: QuestionType) -> str:
        lower, upper = self._word_count_bounds(level, question_type)
        target = self._target_word_count(level, question_type)
        if question_type is QuestionType.LONG_READING:
            return (
                f"Keep the passage around {target} words overall, "
                f"normally within {lower}-{upper}, and prefer 11-12 paragraphs with roughly even length."
            )
        if question_type is QuestionType.WRITING:
            return (
                f"Keep the sample essay within about {lower}-{upper} English words, aiming around {target}. "
                "Set min_response_words to the official lower bound only."
            )
        if question_type is QuestionType.TRANSLATION:
            return (
                f"Keep the Chinese source passage within about {lower}-{upper} Chinese characters, "
                f"aiming around {target}."
            )
        return f"Keep the passage paragraphs between {lower} and {upper} words, aiming around {target}."

    @staticmethod
    def _genre_for(level: Level, question_type: QuestionType, slot: int | None) -> str:
        if question_type is QuestionType.BANKED_CLOZE:
            return (
                "adapted expository or practical feature on study, work, health, or public life"
                if level is Level.CET4
                else "adapted newspaper or magazine commentary with mildly academic tone"
            )
        if question_type is QuestionType.LONG_READING:
            return (
                "multi-paragraph explanatory or service-style feature"
                if level is Level.CET4
                else "information-dense multi-paragraph feature or social-science explainer"
            )
        if question_type is QuestionType.WRITING:
            return (
                "campus-institution CET4 writing prompt or official sentence-led CET6 essay prompt"
                if level is Level.CET4
                else "quoted-sentence CET6 essay prompt on education, society, career, values, or technology"
            )
        if question_type is QuestionType.TRANSLATION:
            return (
                "Chinese expository passage on daily life, public values, or Chinese culture"
                if level is Level.CET4
                else "Chinese expository passage on culture, development, technology, or social thought"
            )
        if slot == 2:
            return "denser explanatory or analytical article"
        return "clear argumentative or expository commentary"

    @staticmethod
    def _register_for(level: Level, question_type: QuestionType, slot: int | None) -> str:
        if question_type is QuestionType.BANKED_CLOZE:
            return (
                "clear expository standard written English with strong local contextual clues"
                if level is Level.CET4
                else "moderately dense standard written English with stronger collocation and abstract-noun competition"
            )
        if question_type is QuestionType.LONG_READING:
            return (
                "readable information-heavy standard written English"
                if level is Level.CET4
                else "information-dense but trackable standard written English with stronger paraphrase distance"
            )
        if question_type is QuestionType.TRANSLATION:
            return "concise Chinese source text with clear logical relations"
        if question_type is QuestionType.WRITING:
            return "exam-style writing instruction in standard written English"
        if question_type is QuestionType.CAREFUL_READING and slot == 2:
            if level is Level.CET4:
                return "clear expository or lightly interpretive standard written English from education, culture, society, technology, or daily-life topics"
            return "mildly academic analytical or reflective standard written English with denser logic"
        if question_type is QuestionType.CAREFUL_READING:
            if level is Level.CET4:
                return "clear explanatory standard written English from science, health, education, work, or public-life topics"
            return "clear analytical standard written English from society, work, public affairs, psychology, economics, or education topics"
        return "clear exam-style standard written English"

    @staticmethod
    def _skill_focus(level: Level, question_type: QuestionType, slot: int | None) -> list[str]:
        if question_type is QuestionType.BANKED_CLOZE:
            return ["vocabulary", "logic", "collocation", "context clue"]
        if question_type is QuestionType.LONG_READING:
            return (
                ["matching", "scanning", "topic sentence recognition", "detail filtering"]
                if level is Level.CET4
                else ["matching", "scanning", "paraphrase recognition", "claim-evidence tracking"]
            )
        if question_type is QuestionType.WRITING:
            return ["content relevance", "coherence", "grammar", "lexical accuracy"]
        if question_type is QuestionType.TRANSLATION:
            return ["translation accuracy", "translation fluency", "grammar", "lexical accuracy"]
        if slot == 1 and level is Level.CET4:
            return [
                "what we learn from the passage about a concrete issue",
                "detail retrieval grounded in evidence, example, or explanation",
                "reason, result, trend, or finding interpretation when natural",
                "text-supported inference",
                "term in context only when the passage naturally supports it",
            ]
        if slot == 2 and level is Level.CET4:
            return [
                "what the passage says or suggests about a social, cultural, or practical issue",
                "conclusion, expectation, or implication from the passage",
                "author view, evaluation, or recommendation when natural",
                "reason or example purpose",
                "inference supported by the passage rather than free speculation",
            ]
        if slot == 1 and level is Level.CET6:
            return [
                "what happens, what is found, or what is said in a denser analytical passage",
                "problem, finding, consequence, or policy implication",
                "advice, response, or practical implication when natural",
                "comparison or evidence-backed interpretation",
                "term in context only when the passage naturally supports it",
            ]
        if slot == 2:
            return [
                "what the author thinks about a social, ethical, or technological issue",
                "example, metaphor, or citation purpose",
                "implied judgment, response, or broader implication",
                "problem framing and reasoning under denser exposition",
                "detail used as support for the author's line of argument",
            ]
        return [
            "main idea or author purpose",
            "important detail retrieval",
            "text-supported inference",
            "overall organization or viewpoint",
            "contextual meaning only when natural",
        ]

    @staticmethod
    def _difficulty_controls(
        level: Level,
        question_type: QuestionType,
        slot: int | None,
    ) -> list[str]:
        controls = [
            "background knowledge must remain common or recoverable from the text",
            "distractors must be plausible but text-checkable",
            "avoid creative-writing tone and avoid trivia-dependent content",
        ]
        if level is Level.CET4:
            controls.extend(
                [
                    "use more familiar scenarios and shorter reasoning chains",
                    "keep sentence structure varied but not overly nested",
                ]
            )
        else:
            controls.extend(
                [
                    "allow mildly academic vocabulary and denser logical transitions",
                    "increase paraphrase distance without obscuring textual support",
                ]
            )
        if question_type is QuestionType.CAREFUL_READING and slot == 2:
            if level is Level.CET4:
                controls.extend(
                    [
                        "make causes, reactions, and suggestions distinct rather than interchangeable",
                        "make at least one item test what the author implies rather than what the passage states directly",
                        "use social-behavior wording that remains familiar to CET4 candidates",
                    ]
                )
            else:
                controls.extend(
                    [
                        "make inference distractors slightly closer in plausibility",
                        "allow denser logical transitions and one harder local-paraphrase item",
                        "use at least one example-purpose or citation-purpose question naturally",
                        "make attitude choices subtle rather than openly emotional",
                    ]
                )
        if question_type is QuestionType.CAREFUL_READING and slot == 1:
            if level is Level.CET4:
                controls.extend(
                    [
                        "prefer concrete textual evidence and explicit local clues over overly abstract inference chains",
                        "let detail, explanation, effect, and suggestion questions dominate more than attitude questions",
                        "allow one or two broad official stems such as 'What do we learn...' or 'What does the author say...' instead of forcing study jargon",
                    ]
                )
            else:
                controls.extend(
                    [
                        "keep the analysis dense but text-checkable, even when the topic is not business-specific",
                        "prefer concrete claims, evidence, effects, or implications before abstract attitude judgment",
                        "avoid forcing all questions into motive/strategy/comparison wording if the passage does not naturally support it",
                    ]
                )
        if question_type is QuestionType.WRITING:
            controls.append("keep the essay prompt practical, neutral, and familiar to CET candidates")
            if level is Level.CET6:
                controls.append("allow a slightly more abstract claim, but keep the argument grounded and non-literary")
        if question_type is QuestionType.TRANSLATION:
            controls.append("use clear Chinese source sentences without literary ornament or obscure historical allusions")
        if question_type is QuestionType.BANKED_CLOZE and level is Level.CET6:
            controls.append("allow slightly denser nominal style and closer synonym distractors than CET4")
        if question_type is QuestionType.LONG_READING and level is Level.CET6:
            controls.append("increase paraphrase distance, but keep each match recoverable through claim-evidence logic")
        return controls

    @staticmethod
    def _structure_plan(level: Level, question_type: QuestionType, slot: int | None) -> list[str]:
        if question_type is QuestionType.BANKED_CLOZE:
            return [
                "2-3 coherent expository paragraphs with 10 inline blanks",
                "15 shared options labeled A. to O.",
                "blank positions should be spread across the passage",
            ]
        if question_type is QuestionType.LONG_READING:
            return [
                "11-12 labeled paragraphs from A. onward with roughly even paragraph length",
                "10 statement items for paragraph matching",
                "at least one paragraph is redundant or reused",
            ]
        if question_type is QuestionType.WRITING:
            return [
                "one short exam instruction block",
                "one compact content prompt paragraph in English with clear CET writing-task shape",
                "one high-scoring sample essay and rubric focus tags",
            ]
        if question_type is QuestionType.TRANSLATION:
            return [
                "one Chinese source paragraph",
                "one polished reference translation",
                "rubric focus tags and response length guidance",
            ]
        if slot == 2:
            if level is Level.CET4:
                return [
                    "4-6 coherent expository or commentary paragraphs in a social-life, campus, or consumer-trend passage",
                    "5 four-option questions or unfinished statements",
                    "question order should roughly follow passage logic and emphasize inference, causes, reactions, and viewpoint",
                ]
            return [
                "4-6 coherent expository or analytical paragraphs in a reflective, ethical, or technology-critique passage",
                "5 four-option questions or unfinished statements",
                "question order should roughly follow passage logic and emphasize inference, citation purpose, and nuanced attitude",
            ]
        if level is Level.CET4:
            return [
                "4-6 coherent expository paragraphs in a research, health, or science-popularization passage",
                "5 four-option questions or unfinished statements",
                "question order should roughly follow passage logic with stronger cause/effect, result, and term-in-context focus",
            ]
        return [
            "4-6 coherent expository or report-style paragraphs in a business, workplace, or economics passage",
            "5 four-option questions or unfinished statements",
            "question order should roughly follow passage logic with stronger consequence, finding, and comparison focus",
        ]

    @staticmethod
    def _option_strategy(level: Level, question_type: QuestionType, slot: int | None) -> list[str]:
        if question_type is QuestionType.BANKED_CLOZE:
            return [
                "mix part-of-speech competition and collocation traps",
                "include a few near-synonyms but keep only one grammatically and logically best answer",
                "make every blank solvable through both syntax and meaning rather than world knowledge alone",
            ]
        if question_type is QuestionType.LONG_READING:
            return [
                "write statement prompts as paraphrases rather than copied lines",
                "use clues from claim, evidence, contrast, or scope rather than simple keyword matching alone",
                "let one or two statements require cross-sentence understanding instead of single-sentence lookup",
            ]
        if question_type is QuestionType.WRITING:
            return [
                "the sample essay should be coherent, natural, and score in the upper CET band",
                "the prompt should favor practical social, campus, or technology themes",
                "the prompt format itself should resemble CET writing instructions rather than a generic composition title",
                "avoid numbered outlines or overly explicit three-point writing scaffolds",
            ]
        if question_type is QuestionType.TRANSLATION:
            return [
                "the reference translation should sound natural instead of word-for-word",
                "the Chinese source should include information points that reward accurate restructuring",
            ]
        if question_type is QuestionType.CAREFUL_READING and slot == 2:
            if level is Level.CET4:
                return [
                    "distractors should reflect misread causes, group reactions, or practical suggestions",
                    "one option may echo the trend but miss the real motive, implied view, or consequence",
                ]
            return [
                "distractors should reflect close paraphrase confusions, local inference errors, or scope mistakes",
                "one option may look text-related but fail on logical relation, citation purpose, or implied stance",
            ]
        if question_type is QuestionType.CAREFUL_READING and slot == 1 and level is Level.CET4:
            return [
                "distractors should reflect confusion among details, causes, effects, suggestions, or local wording in the passage",
                "one option may echo the passage closely but distort the scope, emphasis, or actual conclusion",
            ]
        if question_type is QuestionType.CAREFUL_READING and slot == 1 and level is Level.CET6:
            return [
                "distractors should reflect close paraphrase confusion, incomplete evidence reading, or overstated interpretation",
                "one option may sound plausible and sophisticated but go beyond what the passage actually supports",
            ]
        return [
            "distractors should reflect common misreadings or overgeneralizations",
            "avoid giveaway wording and keep answer distribution natural",
        ]

    @staticmethod
    def _question_id_pattern(question_type: QuestionType) -> str:
        if question_type is QuestionType.CAREFUL_READING:
            return "q1 to q5"
        if question_type in {QuestionType.WRITING, QuestionType.TRANSLATION}:
            return "no objective question ids"
        return "q1 to q10"

    @staticmethod
    def _vocabulary_target_count(question_type: QuestionType) -> int:
        if question_type is QuestionType.LONG_READING:
            return 6
        if question_type in {QuestionType.WRITING, QuestionType.TRANSLATION}:
            return 6
        return 5

    @staticmethod
    def _source_material_hint(level: Level, question_type: QuestionType, slot: int | None) -> str:
        if question_type is QuestionType.BANKED_CLOZE:
            return (
                "adapted magazine or newspaper-style expository passage with stable formal tone"
                if level is Level.CET4
                else "adapted commentary or feature passage from magazines, newspapers, or general academic sources"
            )
        if question_type is QuestionType.LONG_READING:
            return (
                "adapted long-form feature, public information article, or educational explainer"
                if level is Level.CET4
                else "adapted long-form social-science or public-interest feature with clear section logic"
            )
        if question_type is QuestionType.WRITING:
            return (
                "CET writing prompt rather than free composition topic"
            )
        if question_type is QuestionType.TRANSLATION:
            return "Chinese source passage suitable for CET paragraph translation"
        if slot == 1:
            return (
                "adapted expository or analytical passage from original English materials, often slightly more fact-led than the companion passage"
            )
        return (
            "adapted article, commentary, or reflective passage from original English materials, often slightly more interpretive than the companion passage"
        )

    @staticmethod
    def _writing_mode(level: Level, question_type: QuestionType) -> str:
        if question_type is not QuestionType.WRITING:
            return ""
        return (
            "mainly short official exam instructions in a single-paragraph CET4 real-paper pattern: 'Suppose ... You are now to write ... You will have 30 minutes to write the essay ...'"
            if level is Level.CET4
            else "mainly official exam instructions in a sentence-led CET6 real-paper pattern: 'For this part ... write an essay that begins with the sentence ... You can make comments, cite examples ... You should copy the sentence ...'"
        )

    @staticmethod
    def _writing_prompt_examples(level: Level, question_type: QuestionType) -> list[str]:
        if question_type is not QuestionType.WRITING:
            return []
        if level is Level.CET4:
            return [
                "Directions: Suppose the student union of your university is collecting opinions on improving its work for the coming year. You are now to write a response by suggesting how it can better enrich student life. You will have 30 minutes to write the essay. You should write at least 120 words but no more than 180 words.",
                "Directions: Suppose your university is organizing a forum on how students can make the best use of on-campus resources for academic development. You are now to write an essay to express your view. You will have 30 minutes to write the essay. You should write at least 120 words but no more than 180 words.",
                "Directions: Suppose your university is conducting a survey to collect students' opinions on the appropriate use of AI technology in assisting learning. You are now to write an essay to express your view. You will have 30 minutes to write the essay. You should write at least 120 words but no more than 180 words.",
            ]
        return [
            "Directions: For this part, you are allowed 30 minutes to write an essay that begins with the sentence 'While striving for the Chinese Dream, young people enjoy more opportunities to realize their self-worth.' You can make comments, cite examples or use your personal experiences to develop your essay. You should write at least 150 words but no more than 200 words. You should copy the sentence given in quotes at the beginning of your essay.",
            "Directions: For this part, you are allowed 30 minutes to write an essay that begins with the sentence 'It is believed that teachers can exert a profound influence on their students' academic pursuit and personal development.' You can make comments, cite examples or use your personal experiences to develop your essay. You should write at least 150 words but no more than 200 words. You should copy the sentence given in quotes at the beginning of your essay.",
            "Directions: For this part, you are allowed 30 minutes to write an essay that begins with the sentence 'With the increasing application of AI technology, there is a growing concern that it may negatively impact human creativity.' You can make comments, cite examples or use your personal experiences to develop your essay. You should write at least 150 words but no more than 200 words. You should copy the sentence given in quotes at the beginning of your essay.",
        ]

    @staticmethod
    def _translation_domain(level: Level, question_type: QuestionType) -> str:
        if question_type is not QuestionType.TRANSLATION:
            return ""
        return (
            "Chinese culture, history, public life, education, transport, ecology, or everyday social development"
            if level is Level.CET4
            else "Chinese culture, history, economic development, innovation, governance, ecology, or social transformation"
        )

    @staticmethod
    def _exam_profile(level: Level, question_type: QuestionType, slot: int | None) -> str:
        if question_type is QuestionType.BANKED_CLOZE:
            return (
                "Section A word-bank passage adapted from a familiar CET4-style explainer, campus-public-life feature, or accessible social commentary."
                if level is Level.CET4
                else "Section A word-bank passage adapted from a denser CET6-style commentary, public-interest feature, or mildly academic explainer."
            )
        if question_type is QuestionType.LONG_READING:
            return (
                "Section B paragraph-matching article with clear paragraph roles, readable information density, and statement-level paraphrase."
                if level is Level.CET4
                else "Section B paragraph-matching article with denser claim-evidence logic, wider paraphrase distance, and stronger section-level organization."
            )
        if question_type is QuestionType.WRITING:
            return (
                "CET4 writing prompt in official instruction style."
                if level is Level.CET4
                else "CET6 writing prompt in official sentence-led or statement-led instruction style."
            )
        if question_type is QuestionType.TRANSLATION:
            return "CET translation prompt with one Chinese source paragraph."
        if level is Level.CET4 and slot == 1:
            return (
                "CET4 Section C Passage One: an adapted expository passage from real-paper-style sources on health, science, education, work, food, or public life, usually slightly more fact-led than Passage Two."
            )
        if level is Level.CET4 and slot == 2:
            return (
                "CET4 Section C Passage Two: an adapted article on work, psychology, culture, education, consumption, technology, or everyday social issues, usually slightly more interpretive than Passage One."
            )
        if level is Level.CET6 and slot == 1:
            return (
                "CET6 Section C Passage One: an adapted analytical passage on society, work, public affairs, psychology, education, health, economics, or personal development."
            )
        return (
            "CET6 Section C Passage Two: an adapted analytical or reflective passage on social issues, ethics, psychology, technology, economy, history, or culture."
        )

    @staticmethod
    def _passage_flow(level: Level, question_type: QuestionType, slot: int | None) -> list[str]:
        if question_type is QuestionType.BANKED_CLOZE:
            return (
                [
                    "open with a concrete issue or observation",
                    "develop the explanation with examples, causes, or consequences",
                    "close with implication, contrast, or practical takeaway without sounding like a tips list",
                ]
                if level is Level.CET4
                else [
                    "open with a claim, problem, or social observation",
                    "develop it through explanation, comparison, or institutional logic",
                    "close with implication or qualification rather than a motivational summary",
                ]
            )
        if question_type is QuestionType.LONG_READING:
            return (
                [
                    "lead paragraph introduces the issue or service context",
                    "middle paragraphs split into clear functions such as example, problem, response, comparison, or recommendation",
                    "later paragraphs synthesize lessons or practical implications",
                ]
                if level is Level.CET4
                else [
                    "lead paragraph frames a public-interest or institutional issue",
                    "middle paragraphs separate claims, cases, data points, objections, or policy responses",
                    "later paragraphs draw out implication, limitation, or broader consequence",
                ]
            )
        if question_type is QuestionType.CAREFUL_READING and slot == 1:
            return (
                [
                    "introduce a concrete issue, finding, situation, or explanatory claim",
                    "develop details, reasons, examples, data, or evidence",
                    "end with implication, suggestion, or broader takeaway",
                ]
                if level is Level.CET4
                else [
                    "introduce a concrete problem, claim, or line of analysis",
                    "develop it with evidence, comparison, examples, or reasoning",
                    "end with implication, warning, judgment, or recommendation",
                ]
            )
        if question_type is QuestionType.CAREFUL_READING and slot == 2:
            return (
                [
                    "introduce a problem, social pattern, belief, or phenomenon",
                    "analyze it through reasons, examples, contrast, or consequences",
                    "surface the author's suggestion, judgment, or implied view",
                ]
                if level is Level.CET4
                else [
                    "introduce a common belief, modern problem, or cultural tension",
                    "critique it through reasoning plus one or more examples or citations",
                    "end with implied stance, warning, or broader reflection",
                ]
            )
        return []

    @staticmethod
    def _prompt_style_anchors(level: Level, question_type: QuestionType, slot: int | None) -> list[str]:
        if question_type is QuestionType.BANKED_CLOZE:
            return [
                "blanks should rely on sentence grammar plus discourse logic, not isolated synonym guessing",
                "shared options should create real part-of-speech and collocation competition",
                "the passage should still read like an adapted article, not like ten separate sentence exercises",
            ]
        if question_type is QuestionType.LONG_READING:
            return (
                [
                    "statements may target an example, warning, practical suggestion, paragraph function, or contrast",
                    "compact statements such as 'One common mistake is to ...' or 'Some beginners benefit from ...' are more natural than 'The passage says ...'",
                    "items should feel like official matching statements rather than wh-questions rewritten as fragments",
                ]
                if level is Level.CET4
                else [
                    "statements may target a claim, evidence chain, institutional response, contrast, limitation, or policy implication",
                    "compact statements such as 'Local firms suffer when ...' or 'One reported advantage lies in ...' are more natural than 'The article tells us ...'",
                    "items should read like dense information prompts rather than generic summary lines",
                ]
            )
        if question_type is QuestionType.CAREFUL_READING and level is Level.CET4 and slot == 1:
            return [
                "What do we learn from the passage about ...?",
                "What does the author say about ...?",
                "What is one of the reasons for ...?",
                "What is the general trend in ...?",
                "What can we conclude from the passage or the new findings?",
            ]
        if question_type is QuestionType.CAREFUL_READING and level is Level.CET4 and slot == 2:
            return [
                "What does the passage say about ...?",
                "What does the author think of ...?",
                "What can we expect of ...?",
                "What can we conclude about ...?",
                "Why does the author mention ...?",
            ]
        if question_type is QuestionType.CAREFUL_READING and level is Level.CET6 and slot == 1:
            return [
                "What often happens when ...?",
                "What does the author say about ...?",
                "What is the finding of one study about ...?",
                "What is the problem with ...?",
                "How can ... according to the passage?",
            ]
        if question_type is QuestionType.CAREFUL_READING and level is Level.CET6 and slot == 2:
            return [
                "What does the author think about ...?",
                "What do we learn from the passage about ...?",
                "Why does the author use or mention ...?",
                "What is important to ...?",
                "What can be inferred from the passage?",
            ]
        return []

    @staticmethod
    def _authenticity_guardrails(level: Level, question_type: QuestionType, slot: int | None) -> list[str]:
        guardrails = [
            "do not sound like a coaching handout, textbook moral lesson, or motivational essay",
            "do not make every question stem begin with the same formula",
            "keep the passage close to adapted original English materials rather than AI-style explanation",
        ]
        if question_type is QuestionType.BANKED_CLOZE:
            guardrails.extend(
                [
                    "avoid obvious tips-list language such as first, second, finally or here are some suggestions",
                    "avoid turning the passage into classroom advice or a personal reflection",
                ]
            )
        if question_type is QuestionType.LONG_READING:
            guardrails.extend(
                [
                    "avoid generic statements beginning with The passage says, The author believes, or The article tells us in most items",
                    "avoid making all ten matches look like summary slogans instead of information-bearing paraphrases",
                ]
            )
        if question_type is QuestionType.CAREFUL_READING and slot == 1:
            guardrails.extend(
                [
                    "avoid overusing broad attitude questions when the passage is fact-driven",
                    "avoid drifting into a vague inspirational conclusion unrelated to the evidence",
                    "avoid forcing every fact-led set into experiment jargon or business-report jargon if the real-paper topic does not call for it",
                ]
            )
        if question_type is QuestionType.CAREFUL_READING and slot == 2:
            guardrails.extend(
                [
                    "avoid pure fact lookup dominating the set when the passage is viewpoint-driven",
                    "avoid turning attitude or inference items into easy slogan recognition",
                    "avoid forcing an attitude question or vocabulary question if the passage would naturally support stronger conclusion, implication, or example-purpose items instead",
                ]
            )
        if question_type is QuestionType.CAREFUL_READING and level is Level.CET4:
            guardrails.append("keep the reasoning accessible enough for CET4 instead of writing a mini-CET6 passage")
        if question_type is QuestionType.CAREFUL_READING and level is Level.CET6:
            guardrails.append("allow denser logic than CET4, but keep every answer text-supported and exam-like")
        return guardrails

    @staticmethod
    def _topic_pool(level: Level, question_type: QuestionType, slot: int | None) -> list[str]:
        common_cet4 = [
            "sleep habits and learning efficiency",
            "volunteering and campus community life",
            "urban transportation and daily decision making",
            "healthy routines in digital life",
            "consumer choices and sustainability",
            "public communication and trust",
            "library use and independent learning",
            "exercise and student well-being",
            "public health awareness in daily life",
            "green commuting and city convenience",
            "reading habits and information overload",
            "community service and civic participation",
            "museum learning and youth participation",
            "shared public spaces and student routines",
            "food delivery habits and campus time use",
            "community fitness and neighborhood life",
        ]
        common_cet6 = [
            "behavioral economics in everyday decisions",
            "AI tools and workplace reskilling",
            "science communication and public understanding",
            "city planning and climate adaptation",
            "lifelong learning in an aging society",
            "media literacy and information credibility",
            "platform governance and digital responsibility",
            "innovation policy and talent development",
            "workplace flexibility and performance culture",
            "public trust in institutions and expertise",
            "attention economy and deep thinking",
            "technology ethics and social choice",
            "institutional incentives and human behavior",
            "risk communication and public judgment",
            "urban renewal and social memory",
            "organizational trust and performance measurement",
        ]
        if level is Level.CET4 and question_type is QuestionType.LONG_READING:
            return common_cet4 + [
                "student internships and career preparation",
                "community services and civic habits",
                "public libraries and neighborhood learning",
                "healthy campus dining and student choices",
                "museum visits and informal education",
                "urban parks and community well-being",
            ]
        if level is Level.CET6 and question_type is QuestionType.LONG_READING:
            return common_cet6 + [
                "innovation policy and university research",
                "workplace flexibility and productivity",
                "urban renewal and historic preservation",
                "public health systems and risk communication",
                "research culture and interdisciplinary cooperation",
                "transport infrastructure and regional integration",
            ]
        if question_type is QuestionType.CAREFUL_READING and slot == 1:
            return (
                [
                    "circadian rhythm and medication timing",
                    "retirement delay and economic pressure",
                    "pandas and evolutionary comfort",
                    "talent and sustained effort",
                    "junk food advertising and health habits",
                    "genetic testing and personal decision making",
                    "plant-based meat and food choice",
                    "sleep and infant dreaming",
                    "screen time and student attention",
                    "health routines and learning efficiency",
                    "work patterns and life planning",
                ]
                if level is Level.CET4
                else [
                    "friendship and radical individualism",
                    "interdependence and unwanted advice",
                    "mindfulness and health benefits",
                    "public libraries and community life",
                    "career anxiety and self-worth",
                    "simulation training and real-world judgment",
                    "GDP and social well-being",
                    "social media use and personal development",
                    "job application pressure and preparation",
                    "AI and human communication concerns",
                    "education, public life, and social responsibility",
                ]
            )
        if question_type is QuestionType.CAREFUL_READING and slot == 2:
            return (
                [
                    "physical beauty and social pressure",
                    "opera, classical music, and popular taste",
                    "vegan restaurants and changing food culture",
                    "cross-cultural communication and student growth",
                    "college Chinese and curriculum debate",
                    "AI use in learning and student judgment",
                    "academic writing and university requirements",
                    "student organizations and campus life",
                    "campus resources and academic development",
                    "public opinion, taste, and cultural change",
                    "society, values, and personal choice",
                ]
                if level is Level.CET4
                else [
                    "truth, equity, and public dialogue",
                    "solitude and mental focus",
                    "technology and user frustration",
                    "women's pay gap and labor-market change",
                    "social media and responsible use",
                    "Chinese Dream and self-worth",
                    "teachers and student development",
                    "competition and personal challenges",
                    "AI technology and human communication",
                    "social issues, identity, and reflection",
                    "public values and historical understanding",
                ]
            )
        if question_type is QuestionType.BANKED_CLOZE:
            return (
                [
                    "study habits and time management",
                    "healthy routines and daily discipline",
                    "online learning and classroom adaptation",
                    "volunteering and campus community",
                    "public transportation and daily choices",
                    "consumer decisions and sustainability",
                    "public libraries and reading habits",
                    "sleep and student well-being",
                    "community health and public awareness",
                    "food waste reduction on campus",
                    "exercise routines and self-discipline",
                    "shared study spaces and learning habits",
                    "public service convenience in city life",
                    "digital payments and daily efficiency",
                ]
                if level is Level.CET4
                else [
                    "workplace learning and reskilling",
                    "science communication and public trust",
                    "technology use and independent judgment",
                    "urban planning and social efficiency",
                    "behavioral economics in everyday choices",
                    "environmental responsibility and public action",
                    "public policy and civic cooperation",
                    "research literacy and misinformation",
                    "productivity and digital overload",
                    "organizational learning and adaptation",
                    "public infrastructure and social efficiency",
                    "urban sustainability and citizen behavior",
                    "innovation culture and practical judgment",
                    "data use and public confidence",
                ]
            )
        if question_type is QuestionType.WRITING:
            return (
                [
                    "improving the work of the student union",
                    "making the best use of on-campus resources",
                    "students' cross-cultural communication abilities",
                    "the necessity of making College Chinese a compulsory course",
                    "the appropriate use of AI technology in assisting learning",
                    "academic writing as a required university skill",
                    "improving campus volunteer or student-support services",
                    "better use of university library and learning resources",
                    "ways to enrich student life through campus organizations",
                    "how universities can better support students' academic development",
                ]
                if level is Level.CET4
                else [
                    "young people, the Chinese Dream, and self-worth",
                    "teachers' influence on academic pursuit and personal development",
                    "preparing for challenges in a competitive world",
                    "using social media properly and responsibly",
                    "preparing for increasingly demanding job applications",
                    "AI technology and its impact on human creativity or communication",
                    "independent thinking in a changing technological world",
                    "competition, preparation, and personal development",
                    "education, opportunity, and self-development",
                    "responsibility, judgment, and modern public life",
                ]
            )
        if question_type is QuestionType.TRANSLATION:
            return (
                [
                    "public libraries and community life",
                    "traditional festivals and cultural memory",
                    "volunteering and social responsibility",
                    "green travel and urban life",
                    "public services and civic convenience",
                    "family education and good habits",
                    "mobile payment and daily convenience",
                    "traditional crafts and modern life",
                    "sports and public health awareness",
                    "tea culture and daily hospitality",
                    "public transportation and green living",
                    "community festivals and neighborhood life",
                    "local tourism and cultural memory",
                    "voluntary service and social warmth",
                ]
                if level is Level.CET4
                else [
                    "historic neighborhoods and urban renewal",
                    "Chinese culture and modern communication",
                    "technological development and social progress",
                    "ecological protection and long-term governance",
                    "traditional craftsmanship and innovation",
                    "rural revitalization and balanced development",
                    "high-speed rail and regional development",
                    "digital economy and public life",
                    "scientific innovation and national growth",
                    "cultural heritage protection and modern cities",
                    "green development and ecological civilization",
                    "traditional philosophy and contemporary values",
                    "major engineering projects and public benefit",
                    "education reform and talent cultivation",
                ]
            )
        if level is Level.CET4:
            return common_cet4
        return common_cet6

    def _schema_example(self, level: Level, question_type: QuestionType) -> str:
        if question_type is QuestionType.BANKED_CLOZE:
            return (
                "{"
                '"title": "string", '
                '"topic": "string", '
                '"shared_options": ["A. example", "B. example"], '
                '"passage": {"title": "string", "paragraphs": ["string with [1]...[10] blanks"]}, '
                '"questions": [{"id": "q1", "prompt": "Blank 1", "skill_tag": "logic"}], '
                '"answer_key": ["A"], '
                '"analysis": {"overall_strategy": "中文", "overall_summary": "中文", "item_explanations": [{"question_id": "q1", "correct_answer": "A", "explanation": "中文", "skill_tag": "logic"}], "test_tips": ["中文", "中文", "中文"]}, '
                f'"vocabulary": [{self._vocabulary_item_example(level)}]'
                "}"
            )
        if question_type is QuestionType.LONG_READING:
            return (
                "{"
                '"title": "string", '
                '"topic": "string", '
                '"shared_options": [], '
                '"passage": {"title": "string", "paragraphs": ["A. paragraph text", "B. paragraph text"]}, '
                '"questions": [{"id": "q1", "prompt": "English statement to match", "skill_tag": "matching"}], '
                '"answer_key": ["A"], '
                '"analysis": {"overall_strategy": "中文", "overall_summary": "中文", "item_explanations": [{"question_id": "q1", "correct_answer": "A", "explanation": "中文", "skill_tag": "matching"}], "test_tips": ["中文", "中文", "中文"]}, '
                f'"vocabulary": [{self._vocabulary_item_example(level)}]'
                "}"
            )
        if question_type is QuestionType.WRITING:
            return (
                "{"
                '"title": "string", '
                '"topic": "string", '
                '"task_prompt": "English instruction line", '
                '"reference_answer": "high-scoring English essay", '
                '"rubric_focus": ["content_relevance", "coherence", "grammar", "lexical_accuracy"], '
                '"min_response_words": 120, '
                f'"max_response_words": {180 if level is Level.CET4 else 200}, '
                '"shared_options": [], '
                '"passage": {"title": "string", "paragraphs": ["English prompt line 1", "English prompt line 2"]}, '
                '"questions": [], '
                '"answer_key": [], '
                '"analysis": {"overall_strategy": "中文", "overall_summary": "中文", "item_explanations": [], "test_tips": ["中文", "中文", "中文"]}, '
                f'"vocabulary": [{self._vocabulary_item_example(level)}]'
                "}"
            )
        if question_type is QuestionType.TRANSLATION:
            return (
                "{"
                '"title": "string", '
                '"topic": "string", '
                '"task_prompt": "English instruction line", '
                '"reference_answer": "natural English translation", '
                '"rubric_focus": ["translation_accuracy", "translation_fluency", "grammar", "lexical_accuracy"], '
                '"min_response_words": 120, '
                '"max_response_words": 220, '
                '"shared_options": [], '
                '"passage": {"title": "string", "paragraphs": ["中文原文段落"]}, '
                '"questions": [], '
                '"answer_key": [], '
                '"analysis": {"overall_strategy": "中文", "overall_summary": "中文", "item_explanations": [], "test_tips": ["中文", "中文", "中文"]}, '
                f'"vocabulary": [{self._vocabulary_item_example(level)}]'
                "}"
            )
        return (
            "{"
            '"title": "string", '
            '"topic": "string", '
            '"shared_options": [], '
            '"passage": {"title": "string", "paragraphs": ["string"]}, '
            '"questions": [{"id": "q1", "prompt": "English question prompt", "options": ["A. ...", "B. ...", "C. ...", "D. ..."], "skill_tag": "main_idea"}], '
            '"answer_key": ["A"], '
            '"analysis": {"overall_strategy": "中文", "overall_summary": "中文", "item_explanations": [{"question_id": "q1", "correct_answer": "A", "explanation": "中文", "skill_tag": "main_idea"}], "test_tips": ["中文", "中文", "中文"]}, '
            f'"vocabulary": [{self._vocabulary_item_example(level)}]'
            "}"
        )

    def _question_type_details(
        self,
        question_type: QuestionType,
        level: Level,
        slot: int | None,
    ) -> str:
        if question_type is QuestionType.BANKED_CLOZE:
            return (
                "- questions must contain 10 items, one per blank, with prompts like Blank 1, Blank 2, ...\n"
                "- shared_options must contain exactly 15 entries labeled A. to O.\n"
                "- each shared option must be a single English word only.\n"
                "- answer_key must contain exactly 10 letters, each letter must exist in shared_options, and no letter may repeat.\n"
                "- follow the real Section A pattern of a word bank placed after the passage rather than per-question options.\n"
                + (
                    "- CET4 passages should usually stay close to study, health, campus, or public-life topics with explicit local clues.\n"
                    "- Prefer article-like explanation over direct self-help advice or numbered tips.\n"
                    if level is Level.CET4
                    else "- CET6 passages may be denser and more analytical, but should still be solvable through contextual clues rather than specialist knowledge.\n"
                    "- Prefer commentary, social explanation, or public-interest analysis over loose motivational writing.\n"
                )
            )
        if question_type is QuestionType.LONG_READING:
            return (
                "- passage.paragraphs must contain labeled paragraphs beginning with A., B., C. ...\n"
                "- questions must contain 10 English statements for matching, not questions.\n"
                "- answer_key must contain paragraph letters only.\n"
                "- the prompt flow should reflect the official rule that the same paragraph may answer more than one item.\n"
                + (
                    "- CET4 long reading should usually feel like a readable explanatory feature or service-style article.\n"
                    "- Statement prompts should often target an example, warning, recommendation, paragraph role, or practical detail.\n"
                    if level is Level.CET4
                    else "- CET6 long reading may be more information-dense and paraphrase-heavy, but statement matches must remain text-supported.\n"
                    "- Statement prompts should often target a claim, evidence chain, contrast, limitation, or policy implication.\n"
                )
            )
        if question_type is QuestionType.WRITING:
            max_words = 180 if level is Level.CET4 else 200
            style_note = (
                "- CET4 prompts should usually stay practical, concrete, and close to campus or university issues.\n"
                "- CET4 prompts should strongly prefer the real-paper pattern 'Suppose ... You are now to write ... You will have 30 minutes to write the essay ... You should write at least 120 words but no more than 180 words.'\n"
                if level is Level.CET4
                else "- CET6 prompts should usually be sentence-led and somewhat more argumentative or abstract than CET4.\n"
                "- CET6 prompts should strongly prefer the real-paper pattern 'For this part ... write an essay that begins with the sentence ... You can make comments, cite examples or use your personal experiences ... You should copy the sentence ...'\n"
            )
            return (
                "- task_prompt must be an English exam instruction.\n"
                "- passage.paragraphs must contain 1 compact English prompt paragraph in official CET style.\n"
                "- questions and answer_key must both be empty.\n"
                f"- reference_answer must be a high-scoring English essay within about the official {self.validator.WRITING_MIN_WORDS[level]}-{max_words} word band.\n"
                f"{style_note}"
            )
        if question_type is QuestionType.TRANSLATION:
            style_note = (
                "- CET4 source passages should usually stay closer to daily life, public values, or accessible cultural themes.\n"
                if level is Level.CET4
                else "- CET6 source passages may be denser in logic and abstraction, but should remain expository rather than literary.\n"
            )
            return (
                "- task_prompt must be an English exam instruction.\n"
                "- passage.paragraphs must contain the Chinese source passage.\n"
                "- questions and answer_key must both be empty.\n"
                "- reference_answer must be a natural English translation.\n"
                f"{style_note}"
            )
        if slot == 1:
            style_note = (
                "- This set should emulate one passage inside Section C, specifically Passage One.\n"
                "- CET4 Passage One in real papers is often slightly more fact-led and explanatory, but the theme may range across health, science, work, food, education, or public life.\n"
                if level is Level.CET4
                else "- This set should emulate one passage inside Section C, specifically Passage One.\n"
                "- CET6 Passage One in real papers is often analytical and text-dense, but the theme may range across society, work, psychology, public issues, economics, health, or education.\n"
            )
            return (
                "- questions must contain exactly 5 items.\n"
                "- each question must contain exactly 4 options labeled A. to D.\n"
                "- question stems may be direct questions or unfinished statements.\n"
                f"{style_note}"
                "- prioritize official CET-style direct questions or unfinished statements such as what we learn, what the author says, why something happens, what is suggested, and term-in-context when natural.\n"
                "- keep the question order close to the passage flow.\n"
                "- do not force all five skill types to appear exactly once; detail questions may appear more than once.\n"
                "- answer_key must contain exactly 5 letters from A to D."
            )
        if slot == 2:
            style_note = (
                "- This set should emulate one passage inside Section C, specifically Passage Two.\n"
                "- CET4 Passage Two in real papers is often slightly more interpretive or opinion-colored than Passage One, but the theme may still range across culture, psychology, work, technology, food, education, or society.\n"
                if level is Level.CET4
                else "- This set should emulate one passage inside Section C, specifically Passage Two.\n"
                "- CET6 Passage Two in real papers often carries stronger reflection, argument, or value judgment than Passage One, but should still feel like an adapted original article rather than a classroom essay.\n"
            )
            return (
                "- questions must contain exactly 5 items.\n"
                "- each question must contain exactly 4 options labeled A. to D.\n"
                "- question stems may be direct questions or unfinished statements.\n"
                f"{style_note}"
                "- prioritize official CET-style questions such as what can be inferred, why the author mentions something, what the passage implies, what is suggested, and what the author's attitude is when natural.\n"
                "- keep the question order close to the passage flow.\n"
                "- do not force all five skill types to appear exactly once; inference and detail may carry more weight than attitude or vocabulary.\n"
                "- answer_key must contain exactly 5 letters from A to D."
            )
        return (
            "- questions must contain exactly 5 items.\n"
            "- each question must contain exactly 4 options labeled A. to D.\n"
            "- question stems may be direct questions or unfinished statements.\n"
            "- do not use All of the above or None of the above.\n"
            "- answer_key must contain exactly 5 letters from A to D."
        )

    def _vocabulary_item_example(self, level: Level) -> str:
        return (
            "{"
            '"lemma": "string", '
            '"surface_form": "string", '
            f'"level_hint": "{level.value}", '
            '"meaning_zh": "中文", '
            '"example_en": "string"'
            "}"
        )

    def _targeted_repair_guidance(
        self,
        level: Level,
        question_type: QuestionType,
        errors: list[str],
    ) -> str:
        lower, upper = self._word_count_bounds(level, question_type)
        joined = " ".join(errors)
        match = re.search(r"当前为\s*(\d+)", joined)
        current = int(match.group(1)) if match else None
        if current is not None and current < lower:
            expand_by = max(lower - current + 30, 40)
            return (
                f"The current passage is too short. Expand the English passage only by about {expand_by} words "
                f"so the final passage safely lands within {lower}-{upper} words. Keep question ids, answer_key, and analysis alignment intact."
            )
        if current is not None and current > upper:
            trim_by = max(current - upper + 20, 30)
            return (
                f"The current passage is too long. Trim the English passage only by about {trim_by} words "
                f"so the final passage safely lands within {lower}-{upper} words. Keep question ids, answer_key, and analysis alignment intact."
            )
        if "中文" in joined or "Chinese" in joined:
            return (
                "Remove Chinese characters from all English-only fields such as title, topic, passage, prompts, options, and example_en. "
                "Keep Chinese only inside analysis.* and vocabulary.meaning_zh."
            )
        if "单个英文词" in joined or "single English word" in joined:
            return (
                "Rewrite every banked-cloze shared option as one English word only. "
                "Do not use phrases, clauses, or multi-word expressions."
            )
        if "教学提示或建议清单" in joined:
            return (
                "Rewrite only the English passage so it sounds like an adapted CET article rather than a tips list or coaching handout. "
                "Keep the topic, blank positions, answer_key, and vocabulary load aligned."
            )
        if "item_explanations.correct_answer 不一致" in joined or "解析文本显式指向" in joined:
            return (
                "Do not rewrite the passage, question prompts, or answer options. "
                "Re-check every question against the existing passage and options, then correct only answer_key and analysis.item_explanations so that both the answer label and the Chinese explanation point to the same supported answer."
            )
        if "长篇阅读匹配句不应大量以 The passage says" in joined or "长篇阅读匹配句的信息类型过于单一" in joined:
            return (
                "Keep the passage and paragraph labels stable. Rewrite only the matching statements so they become compact exam-style paraphrase statements instead of meta summaries."
            )
        if "仔细阅读 1（CET4）" in joined or "仔细阅读 2（CET4）" in joined or "仔细阅读 1（CET6）" in joined or "仔细阅读 2（CET6）" in joined:
            return (
                "Preserve the overall topic when possible, but revise the careful-reading passage framing and question stems so they match the designated CET slot: fact-driven sets should emphasize purpose/process/findings, while viewpoint-driven sets should emphasize reason/inference/purpose/attitude."
            )
        if question_type is QuestionType.LONG_READING:
            return (
                "If any paragraph-label or length issue remains, keep 11-12 labeled paragraphs and fix only the problematic parts without changing the test style."
            )
        return "Fix only the listed validation issues and keep all already-valid content unchanged."

    def _question_set_from_payload(
        self,
        payload: dict[str, Any],
        level: Level,
        question_type: QuestionType,
        slot: int | None,
        source_type: str,
    ) -> QuestionSet:
        passage = Passage.from_dict(payload["passage"])
        questions = [Question.from_dict(item) for item in payload["questions"]]
        answer_key = [str(item).strip().upper() for item in payload["answer_key"]]
        analysis = AnalysisReport(
            overall_strategy=payload["analysis"]["overall_strategy"],
            overall_summary=payload["analysis"]["overall_summary"],
            item_explanations=[
                ItemExplanation.from_dict(item)
                for item in payload["analysis"].get("item_explanations", [])
            ],
            test_tips=list(payload["analysis"].get("test_tips", [])),
        )
        vocabulary = [VocabularyItem.from_dict(item) for item in payload.get("vocabulary", [])]
        return QuestionSet(
            id=make_id("qs"),
            level=level,
            question_type=question_type,
            slot=slot,
            title=payload["title"],
            topic=payload.get("topic", ""),
            passage=passage,
            questions=questions,
            answer_key=answer_key,
            analysis=analysis,
            vocabulary=vocabulary,
            shared_options=list(payload.get("shared_options", [])),
            task_prompt=payload.get("task_prompt", ""),
            reference_answer=payload.get("reference_answer", ""),
            rubric_focus=list(payload.get("rubric_focus", [])),
            min_response_words=int(payload.get("min_response_words", 0)),
            max_response_words=int(payload.get("max_response_words", 0)),
            word_count=int(payload.get("word_count", 0)),
            generator_model=self._model_name(),
            source_type=source_type,
        )


class SubjectiveEvaluationPipeline:
    def __init__(self, client: DeepSeekClient | None, default_model: str) -> None:
        self.client = client
        self.default_model = default_model

    def evaluate(
        self,
        question_set: QuestionSet,
        response_text: str,
        duration_seconds: int,
    ) -> SubjectiveEvaluation:
        if self.client is None:
            raise RuntimeError(
                "未配置 DEEPSEEK_API_KEY，无法进行写作/翻译 AI 评阅。请在 /data/YueJie-CET/.env 中配置后重试。"
            )
        last_error: SubjectiveEvaluationError | None = None
        for attempt in range(1, 4):
            payload = self.client.create_json_with_tool_schema(
                system_prompt=self._system_prompt(),
                user_prompt=self._user_prompt(
                    question_set,
                    response_text,
                    duration_seconds,
                    retry_attempt=attempt,
                    previous_errors=last_error.errors if last_error else None,
                ),
                tool_name="deliver_subjective_evaluation",
                tool_description="Return one CET subjective-task evaluation as structured JSON.",
                parameters_schema=self._tool_schema(question_set),
                temperature=0.18,
                max_tokens=2600,
                request_timeout=self._request_timeout(question_set, response_text),
            )
            try:
                return self._normalize_evaluation_payload(payload, question_set)
            except SubjectiveEvaluationError as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise RuntimeError("AI 评阅未返回有效结果。")

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are a CET writing and translation evaluator. "
            "Return strict json only. "
            "Score according to CET-style standards, provide concise Chinese feedback, and keep all corrected English natural."
        )

    def _user_prompt(
        self,
        question_set: QuestionSet,
        response_text: str,
        duration_seconds: int,
        retry_attempt: int = 1,
        previous_errors: list[str] | None = None,
    ) -> str:
        task_type = "writing" if question_set.question_type is QuestionType.WRITING else "translation"
        retry_note = ""
        if retry_attempt > 1:
            retry_note = (
                f"\nRetry attempt: {retry_attempt}.\n"
                f"Previous structural issues: {json.dumps(previous_errors or [], ensure_ascii=False)}\n"
                "You must fix those issues and return a fully populated evaluation this time.\n"
            )
        return (
            f"Evaluate this CET {task_type} response.\n"
            f"Level: {question_set.level.value}\n"
            f"Task prompt: {question_set.task_prompt}\n"
            f"Passage/task lines: {json.dumps(question_set.passage.paragraphs, ensure_ascii=False)}\n"
            f"Reference answer: {question_set.reference_answer}\n"
            f"Rubric focus: {json.dumps(question_set.rubric_focus, ensure_ascii=False)}\n"
            f"Suggested time used: {duration_seconds} seconds.\n"
            f"Candidate response:\n{response_text}\n\n"
            f"{retry_note}"
            "Evaluation requirements:\n"
            "- score on a 15-point CET-style scale.\n"
            "- provide 4 score dimensions tied to the rubric.\n"
            "- identify misspelled, misused, or weak words and give corrected English words plus short Chinese reason and Chinese meaning.\n"
            "- identify ungrammatical or awkward sentences and rewrite them in high-quality English.\n"
            "- split the response into sentence-level annotations; for each sentence, comment briefly on strengths and weaknesses in Chinese and provide a better English revision when needed.\n"
            "- rewrite the full response into a high-scoring version while preserving the topic.\n"
            "- weakness_tags should highlight grammar, lexical_accuracy, coherence, content_relevance, translation_accuracy, or translation_fluency as appropriate.\n"
            "- keep all feedback in Chinese except corrected English words, rewritten English sentences, and the high-score version.\n"
            "- if the response is too short, reflect that in score and feedback."
        )

    def _tool_schema(self, question_set: QuestionSet) -> dict[str, Any]:
        dimension_names = (
            ["content_relevance", "coherence", "grammar", "lexical_accuracy"]
            if question_set.question_type is QuestionType.WRITING
            else ["translation_accuracy", "translation_fluency", "grammar", "lexical_accuracy"]
        )
        return {
            "type": "object",
            "properties": {
                "score_15": {"type": "number"},
                "estimated_reported_score": {"type": "number"},
                "grade_band": {"type": "string"},
                "overall_feedback_zh": {"type": "string"},
                "score_dimensions": {
                    "type": "array",
                    "minItems": 4,
                    "maxItems": 4,
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "enum": dimension_names},
                            "score": {"type": "number"},
                            "max_score": {"type": "number"},
                            "feedback_zh": {"type": "string"},
                        },
                        "required": ["name", "score", "max_score", "feedback_zh"],
                        "additionalProperties": False,
                    },
                },
                "wrong_words": {
                    "type": "array",
                    "maxItems": 8,
                    "items": {
                        "type": "object",
                        "properties": {
                            "original": {"type": "string"},
                            "corrected": {"type": "string"},
                            "meaning_zh": {"type": "string"},
                            "reason_zh": {"type": "string"},
                            "skill_tag": {"type": "string", "enum": ["lexical_accuracy", "grammar"]},
                        },
                        "required": ["original", "corrected", "meaning_zh", "reason_zh", "skill_tag"],
                        "additionalProperties": False,
                    },
                },
                "sentence_rewrites": {
                    "type": "array",
                    "maxItems": 6,
                    "items": {
                        "type": "object",
                        "properties": {
                            "original_sentence": {"type": "string"},
                            "revised_sentence": {"type": "string"},
                            "reason_zh": {"type": "string"},
                            "skill_tag": {"type": "string", "enum": ["grammar", "coherence", "translation_fluency"]},
                        },
                        "required": ["original_sentence", "revised_sentence", "reason_zh", "skill_tag"],
                        "additionalProperties": False,
                    },
                },
                "sentence_annotations": {
                    "type": "array",
                    "maxItems": 12,
                    "items": {
                        "type": "object",
                        "properties": {
                            "original_sentence": {"type": "string"},
                            "strengths_zh": {"type": "string"},
                            "issues_zh": {"type": "string"},
                            "revised_sentence": {"type": "string"},
                            "skill_tag": {
                                "type": "string",
                                "enum": [
                                    "grammar",
                                    "lexical_accuracy",
                                    "coherence",
                                    "content_relevance",
                                    "translation_accuracy",
                                    "translation_fluency",
                                    "general",
                                ],
                            },
                        },
                        "required": [
                            "original_sentence",
                            "strengths_zh",
                            "issues_zh",
                            "revised_sentence",
                            "skill_tag",
                        ],
                        "additionalProperties": False,
                    },
                },
                "high_score_version": {"type": "string"},
                "weakness_tags": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [
                            "grammar",
                            "lexical_accuracy",
                            "coherence",
                            "content_relevance",
                            "translation_accuracy",
                            "translation_fluency",
                        ],
                    },
                },
            },
            "required": [
                "score_15",
                "estimated_reported_score",
                "grade_band",
                "overall_feedback_zh",
                "score_dimensions",
                "wrong_words",
                "sentence_rewrites",
                "sentence_annotations",
                "high_score_version",
                "weakness_tags",
            ],
            "additionalProperties": False,
        }

    @staticmethod
    def _request_timeout(question_set: QuestionSet, response_text: str) -> float:
        response_words = len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", response_text))
        if question_set.question_type is QuestionType.TRANSLATION:
            return 180.0 if response_words < 120 else 210.0
        return 170.0 if response_words < 160 else 210.0

    def _normalize_evaluation_payload(
        self,
        payload: dict[str, Any],
        question_set: QuestionSet,
    ) -> SubjectiveEvaluation:
        errors = self._validate_evaluation_payload(payload, question_set)
        if errors:
            raise SubjectiveEvaluationError(errors)
        score_15 = max(0.0, min(15.0, float(payload.get("score_15", 0.0))))
        estimated_reported = max(0.0, min(106.5, float(payload.get("estimated_reported_score", score_15 / 15 * 106.5))))
        dimensions = [
            ScoreDimension.from_dict(item)
            for item in payload.get("score_dimensions", [])
        ]
        wrong_words = [
            WordCorrection.from_dict(item)
            for item in payload.get("wrong_words", [])
        ]
        sentence_rewrites = [
            SentenceRewrite.from_dict(item)
            for item in payload.get("sentence_rewrites", [])
        ]
        sentence_annotations = [
            SentenceAnnotation.from_dict(item)
            for item in payload.get("sentence_annotations", [])
        ]
        weakness_tags = [str(item).strip() for item in payload.get("weakness_tags", []) if str(item).strip()]
        high_score_version = str(payload.get("high_score_version", "")).strip()
        overall_feedback = str(payload.get("overall_feedback_zh", "")).strip()
        if not weakness_tags:
            weakness_tags = [item.name for item in dimensions if item.score < item.max_score * 0.75]
        return SubjectiveEvaluation(
            score_15=round(score_15, 1),
            estimated_reported_score=round(estimated_reported, 1),
            grade_band=str(payload.get("grade_band", "")).strip(),
            overall_feedback_zh=overall_feedback,
            score_dimensions=dimensions,
            wrong_words=wrong_words[:8],
            sentence_rewrites=sentence_rewrites[:6],
            sentence_annotations=sentence_annotations[:12],
            high_score_version=high_score_version,
            weakness_tags=weakness_tags[:6],
        )

    def _validate_evaluation_payload(
        self,
        payload: dict[str, Any],
        question_set: QuestionSet,
    ) -> list[str]:
        errors: list[str] = []
        try:
            score_15 = float(payload.get("score_15", -1))
        except (TypeError, ValueError):
            score_15 = -1
        if not 0.0 <= score_15 <= 15.0:
            errors.append("score_15 缺失或不在 0-15 范围内")

        overall_feedback = str(payload.get("overall_feedback_zh", "")).strip()
        if len(overall_feedback) < 12:
            errors.append("overall_feedback_zh 过短或为空")

        dimensions = payload.get("score_dimensions", [])
        if not isinstance(dimensions, list) or len(dimensions) != 4:
            errors.append("score_dimensions 必须正好提供 4 项")
        else:
            required = (
                {"content_relevance", "coherence", "grammar", "lexical_accuracy"}
                if question_set.question_type is QuestionType.WRITING
                else {
                    "translation_accuracy",
                    "translation_fluency",
                    "grammar",
                    "lexical_accuracy",
                }
            )
            names = {str(item.get("name", "")).strip() for item in dimensions if isinstance(item, dict)}
            if names != required:
                errors.append("score_dimensions.name 与题型要求不匹配")
            for item in dimensions:
                if not isinstance(item, dict):
                    errors.append("score_dimensions 存在非法项")
                    continue
                if len(str(item.get("feedback_zh", "")).strip()) < 6:
                    errors.append("score_dimensions.feedback_zh 不能为空")
                    break

        wrong_words = payload.get("wrong_words", [])
        sentence_rewrites = payload.get("sentence_rewrites", [])
        sentence_annotations = payload.get("sentence_annotations", [])
        high_score_version = str(payload.get("high_score_version", "")).strip()
        weakness_tags = payload.get("weakness_tags", [])

        if not isinstance(wrong_words, list):
            errors.append("wrong_words 必须为数组")
        if not isinstance(sentence_rewrites, list):
            errors.append("sentence_rewrites 必须为数组")
        if not isinstance(sentence_annotations, list):
            errors.append("sentence_annotations 必须为数组")
        if not high_score_version:
            errors.append("high_score_version 不能为空")
        if not isinstance(weakness_tags, list) or not weakness_tags:
            errors.append("weakness_tags 不能为空")

        if sentence_annotations == [] and sentence_rewrites == [] and wrong_words == []:
            errors.append("批注结果为空，至少应返回逐句批注、改写或错词中的一类")

        for item in sentence_annotations[:3]:
            if not isinstance(item, dict):
                errors.append("sentence_annotations 存在非法项")
                break
            if not str(item.get("original_sentence", "")).strip():
                errors.append("sentence_annotations.original_sentence 不能为空")
                break
            if not (
                str(item.get("strengths_zh", "")).strip()
                or str(item.get("issues_zh", "")).strip()
                or str(item.get("revised_sentence", "")).strip()
            ):
                errors.append("sentence_annotations 至少应包含亮点、问题或改写之一")
                break

        return errors
