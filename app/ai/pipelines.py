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
        max_generation_rounds = 3
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
        topic_pool = [
            item
            for item in self._topic_pool(level, question_type, slot)
            if item not in anti_repeat_topics
        ] or self._topic_pool(level, question_type, slot)
        return {
            "topic": secrets_choice(topic_pool),
            "genre": self._genre_for(level, question_type, slot),
            "register": self._register_for(level, question_type, slot),
            "source_material_hint": self._source_material_hint(level, question_type, slot),
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

    @staticmethod
    def _question_spec(level: Level, question_type: QuestionType, slot: int | None) -> str:
        if question_type is QuestionType.BANKED_CLOZE:
            word_spec = "CET4 200-250 words; CET6 250-300 words."
            return (
                f"{word_spec} Use 10 blanks in the passage marked [1]...[10]. "
                "Provide exactly 15 shared options labeled A. to O. "
                "Each shared option must be one English word only, not a phrase or clause. "
                "Targets and distractors should be CET-like in part of speech and collocation difficulty. "
                "Each of the 10 answers must use a different letter; no option may be reused."
            )
        if question_type is QuestionType.LONG_READING:
            word_spec = "CET4 850-1150 words; CET6 1050-1350 words."
            return (
                f"{word_spec} Provide exactly 10 statements for paragraph matching. "
                "Paragraphs must be labeled A., B., C. ... with 10-14 total paragraphs, and at least one paragraph should be redundant or one paragraph may answer more than one item. "
                "Each matching item must be an English statement, not a question, and must not include paragraph labels."
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
                "For CET6, strongly prefer prompts that look like official sentence-led tasks such as 'write an essay that begins with the sentence ...', followed by brief official guidance. "
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
                "Model it as a fact-driven CET4 passage: usually research, experiment, medicine, health, or science-popularization material. "
                "Question stems should naturally use words like experiment, study, researcher, participant, result, or finding when appropriate. "
                "Most items should focus on cause-and-effect detail, study findings, result interpretation, and definition in context. "
            )
        elif level is Level.CET4 and slot == 2:
            slot_style = (
                "Model it as a viewpoint-driven CET4 passage: usually social-life, campus-life, consumer, or public-psychology material. "
                "Question stems should naturally use words like trend, consumer, reason, reaction, suggestion, or attitude when appropriate. "
                "Most items should focus on implicit inference, reason analysis, group reaction, example purpose, and author suggestion or viewpoint. "
            )
        elif level is Level.CET6 and slot == 1:
            slot_style = (
                "Model it as a fact-driven CET6 passage: usually business, economics, workplace strategy, or market-report material. "
                "Question stems should naturally use words like motive, consequence, strategy, expansion, or comparison when appropriate. "
                "Most items should focus on cause-and-effect detail, market consequence, result interpretation, comparative evidence, and term-in-context understanding. "
            )
        else:
            slot_style = (
                "Model it as a viewpoint-driven CET6 passage: usually psychology, ethics, technology-reflection, or social-critique material with denser reasoning. "
                "Question stems should naturally use words like infer, cite, imply, stance, skeptical, or critical when appropriate. "
                "Most items should focus on implicit inference, citation/example purpose, implied logic, and nuanced attitude or main idea. "
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
        schema = self._schema_example(level, question_type)
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
            "- For long reading, statements must be paraphrases, not direct copies of paragraph openings.\n"
            "- For long reading, every item prompt must be a statement rather than a question and must not include A./B./C. labels.\n"
            "- For careful reading, distribute correct options naturally and make distractors plausible.\n"
            "- For careful reading, keep four options parallel in grammar and length, and never use all/none of the above.\n"
            "- For writing, produce a CET-style prompt format such as a short situational instruction, a survey/opinion task, a quoted statement, or a sentence-led argumentative task as appropriate to the level.\n"
            "- For CET4 writing, the prompt should look closer to official exam wording such as 'Suppose ... You are now to write ...' than to abstract philosophical debate.\n"
            "- For CET6 writing, the prompt should often look like an official sentence-led task built around a quoted sentence, followed by very brief guidance.\n"
            "- The visible content prompt should normally be one short paragraph only.\n"
            "- Do not use explicit numbered outlines like 1., 2., 3. or detailed multi-point Chinese-style writing hints in the prompt lines.\n"
            "- Keep writing prompts compact. Avoid multi-step scaffolds or teacher-style classroom instructions.\n"
            "- For translation, provide a Chinese source passage on Chinese culture, history, society, development, education, ecology, or technology as appropriate to the level.\n"
            "- Do not write in a dramatic, fictional, or conversational blog style.\n"
            "- Use concise Chinese explanations that point back to textual evidence or reasoning path.\n"
            f"Question-type details:\n{self._question_type_details(question_type, level, slot)}\n"
            "Return one JSON object using exactly this schema shape and no extra top-level keys:\n"
            f"{schema}"
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
            f"Required schema:\n{self._schema_example(level, question_type)}\n"
            "Original JSON:\n"
            f"{json.dumps(payload, ensure_ascii=False)}\n"
            "Return a corrected full JSON object only. Preserve valid content and change only what is needed. "
            "If a word-count error is listed, expand or trim the passage itself until it is inside the required range while keeping the passage coherent."
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
        for repair_round in range(1, 3):
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
                "task_prompt": {"type": "string"},
                "reference_answer": {"type": "string"},
                "rubric_focus": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 4,
                    "maxItems": 6,
                },
                "min_response_words": {"type": "integer"},
                "max_response_words": {"type": "integer"},
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
                "scenario-based campus or social essay prompt"
                if level is Level.CET4
                else "sentence-led or view-comment essay prompt"
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
                return "clear social or consumer commentary in standard written English"
            return "mildly academic reflective or critical standard written English with denser logic"
        if question_type is QuestionType.CAREFUL_READING:
            if level is Level.CET4:
                return "clear research-style or science-popularization standard written English"
            return "clear business or economics commentary in standard written English"
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
                "cause-and-effect detail",
                "study findings or result interpretation",
                "participant / variable / procedure detail",
                "text-supported inference",
                "definition or term in context when natural",
            ]
        if slot == 2 and level is Level.CET4:
            return [
                "implicit inference",
                "reason behind a trend or behavior",
                "group reaction or suggestion",
                "example purpose",
                "author viewpoint or attitude when natural",
            ]
        if slot == 1 and level is Level.CET6:
            return [
                "cause-and-effect detail in business or market logic",
                "market finding or consequence interpretation",
                "comparison backed by evidence",
                "strategy detail or report conclusion",
                "term in context when natural",
            ]
        if slot == 2:
            return [
                "implicit inference under denser reflective logic",
                "example purpose or citation purpose",
                "implied stance or nuanced attitude",
                "main idea under abstract exposition",
                "detail only as support for the author's line of reasoning",
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
                        "keep the experimental setup, groups, or findings explicitly stated",
                        "prefer cause/effect and finding questions over broad attitude questions",
                        "prefer purpose, process, and result questions before abstract interpretation",
                    ]
                )
            else:
                controls.extend(
                    [
                        "keep the market logic and strategic consequences explicit enough to text-check",
                        "prefer consequence, report finding, and comparison questions over broad attitude questions",
                        "prefer concrete business evidence before abstract reflection",
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
                "distractors should reflect confusion among study purpose, procedure, groups, and findings",
                "one option may sound scientific but misstate the actual result, variable, or causal link",
            ]
        if question_type is QuestionType.CAREFUL_READING and slot == 1 and level is Level.CET6:
            return [
                "distractors should reflect confusion among business motive, expansion consequence, and comparative advantage",
                "one option may sound commercially sensible but overstate what the report or market evidence actually proves",
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
                "adapted research report, science explainer, or business report from original English materials"
            )
        return (
            "adapted opinion piece, social commentary, or reflective magazine article from original English materials"
        )

    @staticmethod
    def _writing_mode(level: Level, question_type: QuestionType) -> str:
        if question_type is not QuestionType.WRITING:
            return ""
        return (
            "mainly short official exam instructions, especially CET4 forms like 'Suppose ... You are now to write ...'"
            if level is Level.CET4
            else "mainly official sentence-led CET6 writing, especially forms like 'write an essay that begins with the sentence ...'"
        )

    @staticmethod
    def _writing_prompt_examples(level: Level, question_type: QuestionType) -> list[str]:
        if question_type is not QuestionType.WRITING:
            return []
        if level is Level.CET4:
            return [
                "A short campus or social situation beginning with 'Suppose ...' and followed by 'You are now to write ...'.",
                "A compact official-looking instruction about a practical campus or social issue.",
                "A short CET4 task paragraph without numbered sub-points.",
            ]
        return [
            "A sentence-led official task such as 'For this part, you are allowed 30 minutes to write an essay that begins with the sentence ...'.",
            "A quoted saying or statement followed by brief official guidance on commenting or citing examples.",
            "A compact CET6 prompt paragraph without numbered sub-points or classroom-style scaffolding.",
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
                    "sleep deprivation and memory performance",
                    "screen time and student attention",
                    "exercise habits and learning efficiency",
                    "nutrition research and daily energy levels",
                    "stress experiments and college decision making",
                    "medical or behavioral study findings in student life",
                    "breakfast habits and classroom performance",
                    "walking routines and concentration in students",
                    "noise exposure and study efficiency",
                    "reading format and memory retention",
                    "study breaks and sustained attention",
                ]
                if level is Level.CET4
                else [
                    "pricing strategy and market competition",
                    "platform expansion and local business pressure",
                    "workplace incentives and productivity trade-offs",
                    "consumer credit and retail strategy",
                    "business models and competitive advantage",
                    "corporate expansion and market consequence",
                    "subscription models and customer retention",
                    "automation and workplace restructuring",
                    "supply chain pressure and pricing decisions",
                    "brand strategy and consumer loyalty",
                    "labor-market incentives and firm behavior",
                ]
            )
        if question_type is QuestionType.CAREFUL_READING and slot == 2:
            return (
                [
                    "second-hand shopping among young consumers",
                    "campus social habits and student identity",
                    "young people's digital behavior and attention",
                    "public trends and peer influence",
                    "changing consumer preferences among students",
                    "social-media habits and everyday decision making",
                    "shared bicycles and urban student mobility",
                    "volunteering and social belonging among students",
                    "digital reading and fragmented attention",
                    "minimalist consumption among young adults",
                    "peer influence on campus lifestyle choices",
                ]
                if level is Level.CET4
                else [
                    "algorithmic influence on deep thinking",
                    "privacy promises and tech-company credibility",
                    "echo chambers and public reasoning",
                    "digital platforms and moral responsibility",
                    "technology criticism and human autonomy",
                    "social psychology of online conformity",
                    "misinformation and civic judgment",
                    "AI-generated content and trust",
                    "remote work and identity formation",
                    "surveillance convenience and personal freedom",
                    "platform design and democratic discourse",
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
                    "reading habits and independent learning",
                    "exercise and healthy university life",
                    "volunteering and personal growth",
                    "time management and study efficiency",
                    "balanced technology use on campus",
                    "environmental responsibility in daily life",
                    "library use and independent learning",
                    "campus service and student responsibility",
                    "how to handle stress in college life",
                    "the importance of teamwork in college",
                ]
                if level is Level.CET4
                else [
                    "AI tools and independent thinking",
                    "lifelong learning in a changing world",
                    "efficiency and reflection in modern study",
                    "public trust and responsible communication",
                    "innovation and human judgment",
                    "technology and the quality of learning",
                    "competition and cooperation in modern society",
                    "individual choice and social responsibility",
                    "the value of patience in a fast-moving world",
                    "efficiency and depth in the information age",
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
                + (
                    "- CET4 passages should usually stay close to study, health, campus, or public-life topics with explicit local clues.\n"
                    if level is Level.CET4
                    else "- CET6 passages may be denser and more analytical, but should still be solvable through contextual clues rather than specialist knowledge.\n"
                )
            )
        if question_type is QuestionType.LONG_READING:
            return (
                "- passage.paragraphs must contain labeled paragraphs beginning with A., B., C. ...\n"
                "- questions must contain 10 English statements for matching, not questions.\n"
                "- answer_key must contain paragraph letters only.\n"
                + (
                    "- CET4 long reading should usually feel like a readable explanatory feature or service-style article.\n"
                    if level is Level.CET4
                    else "- CET6 long reading may be more information-dense and paraphrase-heavy, but statement matches must remain text-supported.\n"
                )
            )
        if question_type is QuestionType.WRITING:
            max_words = 180 if level is Level.CET4 else 200
            style_note = (
                "- CET4 prompts should usually stay practical, concrete, and close to campus or daily social issues.\n"
                if level is Level.CET4
                else "- CET6 prompts should usually be sentence-led or opinion-led and somewhat more argumentative or abstract than CET4.\n"
            )
            return (
                "- task_prompt must be an English exam instruction.\n"
                "- passage.paragraphs must contain 2-3 English prompt lines.\n"
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
                "- CET4 Passage One should usually be fact-driven research, medicine, health, or science-popularization material, with question wording around experiment/study/result/participants.\n"
                if level is Level.CET4
                else "- CET6 Passage One should usually be fact-driven business, market, workplace, or economics material, with question wording around motive/consequence/strategy/comparison.\n"
            )
            return (
                "- questions must contain exactly 5 items.\n"
                "- each question must contain exactly 4 options labeled A. to D.\n"
                "- question stems may be direct questions or unfinished statements.\n"
                f"{style_note}"
                "- prioritize cause/effect, findings, comparisons, and term-in-context over broad attitude questions.\n"
                "- do not force all five skill types to appear exactly once; detail questions may appear more than once.\n"
                "- answer_key must contain exactly 5 letters from A to D."
            )
        if slot == 2:
            style_note = (
                "- CET4 Passage Two should usually be viewpoint-driven social-life, campus, or consumer-trend material, with question wording around trend/reason/reaction/suggestion.\n"
                if level is Level.CET4
                else "- CET6 Passage Two should usually be viewpoint-driven psychology, ethics, or technology-reflection material, with question wording around infer/cite/imply/attitude.\n"
            )
            return (
                "- questions must contain exactly 5 items.\n"
                "- each question must contain exactly 4 options labeled A. to D.\n"
                "- question stems may be direct questions or unfinished statements.\n"
                f"{style_note}"
                "- prioritize inference, example/citation purpose, and viewpoint/attitude over pure fact lookup.\n"
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
        if "item_explanations.correct_answer 不一致" in joined or "解析文本显式指向" in joined:
            return (
                "Do not rewrite the passage, question prompts, or answer options. "
                "Re-check every question against the existing passage and options, then correct only answer_key and analysis.item_explanations so that both the answer label and the Chinese explanation point to the same supported answer."
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
