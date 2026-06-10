from __future__ import annotations

import json
import re
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
    SubjectiveEvaluation,
    VocabularyItem,
    WordCorrection,
    AttemptQuestionResult,
    make_id,
)


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
        weakness_summary: str | None = None,
        progress_callback: Callable[[str, str], None] | None = None,
    ) -> QuestionSet:
        if self.client is None:
            raise RuntimeError(
                "未配置 DEEPSEEK_API_KEY，无法进行真实 AI 出题。请在 /data/YueJie-CET/.env 中配置后重试。"
            )

        self._report(
            progress_callback,
            "blueprint",
            "已锁定四六级题型规范，正在整理题面蓝图与难度控制。",
        )
        blueprint = self._build_blueprint(level, question_type, slot, weakness_summary)
        retry_errors: list[str] | None = None
        last_validation_error: QuestionSetValidationError | None = None
        for generation_round in range(1, 3):
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
                last_validation_error = exc
                retry_errors = exc.errors
                if generation_round == 2:
                    raise
        if last_validation_error is not None:
            raise last_validation_error
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
    ) -> dict[str, Any]:
        return {
            "topic": secrets_choice(self._topic_pool(level, question_type)),
            "genre": self._genre_for(question_type),
            "register": self._register_for(level, question_type, slot),
            "target_word_count": self._target_word_count(level, question_type),
            "word_count_guidance": self._strict_word_count_guidance(level, question_type),
            "skill_focus": self._skill_focus(question_type, slot),
            "difficulty_controls": self._difficulty_controls(level, question_type, slot),
            "structure_plan": self._structure_plan(question_type, slot),
            "option_strategy": self._option_strategy(question_type),
            "weakness_focus": weakness_summary or "No weakness summary yet. Keep the skill mix balanced.",
            "question_id_pattern": self._question_id_pattern(question_type),
            "vocabulary_target_count": self._vocabulary_target_count(question_type),
            "analysis_style": "Chinese only, concise, evidence-based, and useful for CET review.",
        }

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
            return (
                f"CET writing task. Write an English essay of no less than {min_words} words. "
                "Provide an exam-style task prompt in English, 2-3 short instruction lines, a high-scoring sample essay, and rubric focus tags. "
                "No objective questions, options, or answer_key should be included."
            )
        if question_type is QuestionType.TRANSLATION:
            char_spec = "140-160 Chinese characters" if level is Level.CET4 else "180-200 Chinese characters"
            return (
                f"CET translation task. Provide one Chinese-to-English translation source passage of about {char_spec}. "
                "Also provide a polished reference translation in English, rubric focus tags, and response length guidance. "
                "No objective questions, options, or answer_key should be included."
            )
        word_spec = "CET4 290-360 words; CET6 390-470 words."
        slot_hint = f"This is careful reading slot {slot}. " if slot else ""
        return (
            f"{slot_hint}{word_spec} Provide exactly 5 four-option multiple-choice questions. "
            "Use this distribution exactly once each: main idea, detail, inference, vocabulary in context, and attitude/tone. "
            "Question stems must be English-only; avoid trick options like All of the above or None of the above."
        )

    def _generation_system_prompt(self) -> str:
        return (
            "You are a CET reading item generator. "
            "Return strict json only. No markdown fences. "
            "The final output must be directly machine-readable and fully self-contained. "
            "Passage, titles, questions, and answer options must stay in English. "
            "Analysis and vocabulary explanations must be in Chinese. "
            "Keep the output exam-like, concise, and structurally exact."
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
            "- Keep the style close to CET source materials such as news features, opinion pieces, science explanations, educational commentary, or general academic reading.\n"
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
            "- For banked cloze, the 15 options must be shared options only, not repeated per blank.\n"
            "- For banked cloze, every shared option must be exactly one English word, not a phrase.\n"
            "- For banked cloze, each answer letter may be used only once.\n"
            "- For long reading, statements must be paraphrases, not direct copies of paragraph openings.\n"
            "- For long reading, every item prompt must be a statement rather than a question and must not include A./B./C. labels.\n"
            "- For careful reading, distribute correct options naturally and make distractors plausible.\n"
            "- For careful reading, keep four options parallel in grammar and length, and never use all/none of the above.\n"
            "- For writing, produce an exam-style essay prompt, task instructions, and a high-scoring sample essay.\n"
            "- For translation, provide a Chinese source passage and a natural English reference translation.\n"
            "- Do not write in a dramatic, fictional, or conversational blog style.\n"
            "- Use concise Chinese explanations that point back to textual evidence or reasoning path.\n"
            f"Question-type details:\n{self._question_type_details(question_type)}\n"
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

        passage = dict(normalized.get("passage", {}))
        paragraphs = [str(item).strip() for item in passage.get("paragraphs", []) if str(item).strip()]
        if question_type is QuestionType.LONG_READING:
            paragraphs = self._normalize_long_reading_paragraphs(paragraphs)
        passage["paragraphs"] = paragraphs
        passage["title"] = str(passage.get("title", normalized.get("title", ""))).strip()
        normalized["passage"] = passage

        questions = []
        for index, item in enumerate(normalized.get("questions", []), start=1):
            question = dict(item) if isinstance(item, dict) else {}
            question["id"] = f"q{index}"
            question["prompt"] = self._normalized_prompt(question, question_type, index)
            question["options"] = self._normalize_question_options(question, question_type)
            question["skill_tag"] = self._normalized_skill_tag(question, question_type, index, slot)
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
            explanation["correct_answer"] = (
                str(explanation.get("correct_answer", "")).strip().upper().replace(".", "")
            )
            explanation["skill_tag"] = self._normalized_skill_tag(
                {"prompt": questions[index - 1]["prompt"], "skill_tag": explanation.get("skill_tag", "")},
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
            str(item).strip().upper().replace(".", "").replace(")", "")
            for item in answer_key
        ]
        if question_type is QuestionType.CAREFUL_READING:
            return [item[:1] for item in normalized]
        return [item[:1] if item else "" for item in normalized]

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
        if slot == 2:
            ordered = ["inference", "detail", "attitude", "vocabulary_in_context", "main_idea"]
        return ordered[(index - 1).min(len(ordered) - 1)]

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
            return 980 if level is Level.CET4 else 1180
        if question_type is QuestionType.WRITING:
            return 160 if level is Level.CET4 else 190
        if question_type is QuestionType.TRANSLATION:
            return 150 if level is Level.CET4 else 190
        return 330 if level is Level.CET4 else 430

    @staticmethod
    def _word_count_bounds(level: Level, question_type: QuestionType) -> tuple[int, int]:
        if question_type is QuestionType.BANKED_CLOZE:
            return (200, 250) if level is Level.CET4 else (250, 300)
        if question_type is QuestionType.LONG_READING:
            return (850, 1150) if level is Level.CET4 else (1050, 1350)
        if question_type is QuestionType.WRITING:
            return (120, 220) if level is Level.CET4 else (150, 260)
        if question_type is QuestionType.TRANSLATION:
            return (140, 160) if level is Level.CET4 else (180, 200)
        return (290, 360) if level is Level.CET4 else (390, 470)

    def _strict_word_count_guidance(self, level: Level, question_type: QuestionType) -> str:
        lower, upper = self._word_count_bounds(level, question_type)
        target = self._target_word_count(level, question_type)
        if question_type is QuestionType.LONG_READING:
            return (
                f"Keep the passage paragraphs between {lower} and {upper} words, "
                f"aim around {target}, and prefer 11-12 paragraphs with roughly even length."
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
    def _genre_for(question_type: QuestionType) -> str:
        if question_type is QuestionType.BANKED_CLOZE:
            return "adapted magazine or newspaper feature"
        if question_type is QuestionType.LONG_READING:
            return "multi-paragraph explanatory feature"
        if question_type is QuestionType.WRITING:
            return "short campus or social issue essay prompt"
        if question_type is QuestionType.TRANSLATION:
            return "Chinese expository or cultural passage for translation"
        return "argumentative or expository article"

    @staticmethod
    def _register_for(level: Level, question_type: QuestionType, slot: int | None) -> str:
        if question_type is QuestionType.LONG_READING:
            return "information-dense but readable standard written English"
        if question_type is QuestionType.TRANSLATION:
            return "concise Chinese source text with clear logical relations"
        if question_type is QuestionType.WRITING:
            return "exam-style writing instruction in standard written English"
        if question_type is QuestionType.CAREFUL_READING and slot == 2:
            return (
                "mildly academic and logically denser standard written English"
                if level is Level.CET6
                else "slightly denser exam-style standard written English"
            )
        return "clear exam-style standard written English"

    @staticmethod
    def _skill_focus(question_type: QuestionType, slot: int | None) -> list[str]:
        if question_type is QuestionType.BANKED_CLOZE:
            return ["vocabulary", "logic", "collocation", "context clue"]
        if question_type is QuestionType.LONG_READING:
            return ["matching", "scanning", "paraphrase recognition", "detail filtering"]
        if question_type is QuestionType.WRITING:
            return ["content relevance", "coherence", "grammar", "lexical accuracy"]
        if question_type is QuestionType.TRANSLATION:
            return ["translation accuracy", "translation fluency", "grammar", "lexical accuracy"]
        if slot == 2:
            return ["inference", "logic chain", "detail", "attitude", "vocabulary in context"]
        return ["main idea", "detail", "inference", "vocabulary in context", "attitude"]

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
            controls.append("make inference and attitude distractors slightly closer in plausibility")
        if question_type is QuestionType.WRITING:
            controls.append("keep the essay prompt practical, neutral, and familiar to CET candidates")
        if question_type is QuestionType.TRANSLATION:
            controls.append("use clear Chinese source sentences without literary ornament or obscure historical allusions")
        return controls

    @staticmethod
    def _structure_plan(question_type: QuestionType, slot: int | None) -> list[str]:
        if question_type is QuestionType.BANKED_CLOZE:
            return [
                "2-4 coherent paragraphs with 10 inline blanks",
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
                "2-3 prompt lines in English",
                "one high-scoring sample essay and rubric focus tags",
            ]
        if question_type is QuestionType.TRANSLATION:
            return [
                "one Chinese source paragraph",
                "one polished reference translation",
                "rubric focus tags and response length guidance",
            ]
        if slot == 2:
            return [
                "4-6 coherent paragraphs",
                "5 four-option questions with slightly denser reasoning",
                "question order should roughly follow passage logic",
            ]
        return [
            "4-6 coherent paragraphs",
            "5 four-option questions",
            "question order should roughly follow passage logic",
        ]

    @staticmethod
    def _option_strategy(question_type: QuestionType) -> list[str]:
        if question_type is QuestionType.BANKED_CLOZE:
            return [
                "mix part-of-speech competition and collocation traps",
                "include a few near-synonyms but keep only one grammatically and logically best answer",
            ]
        if question_type is QuestionType.LONG_READING:
            return [
                "write statement prompts as paraphrases rather than copied lines",
                "use clues from claim, evidence, contrast, or scope rather than simple keyword matching alone",
            ]
        if question_type is QuestionType.WRITING:
            return [
                "the sample essay should be coherent, natural, and score in the upper CET band",
                "the prompt should favor practical social, campus, or technology themes",
            ]
        if question_type is QuestionType.TRANSLATION:
            return [
                "the reference translation should sound natural instead of word-for-word",
                "the Chinese source should include information points that reward accurate restructuring",
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
    def _topic_pool(level: Level, question_type: QuestionType) -> list[str]:
        common_cet4 = [
            "sleep habits and learning efficiency",
            "volunteering and campus community life",
            "urban transportation and daily decision making",
            "healthy routines in digital life",
            "consumer choices and sustainability",
            "public communication and trust",
        ]
        common_cet6 = [
            "behavioral economics in everyday decisions",
            "AI tools and workplace reskilling",
            "science communication and public understanding",
            "city planning and climate adaptation",
            "lifelong learning in an aging society",
            "media literacy and information credibility",
        ]
        if level is Level.CET4 and question_type is QuestionType.LONG_READING:
            return common_cet4 + [
                "student internships and career preparation",
                "community services and civic habits",
            ]
        if level is Level.CET6 and question_type is QuestionType.LONG_READING:
            return common_cet6 + [
                "innovation policy and university research",
                "workplace flexibility and productivity",
            ]
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
                '"max_response_words": 220, '
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

    def _question_type_details(self, question_type: QuestionType) -> str:
        if question_type is QuestionType.BANKED_CLOZE:
            return (
                "- questions must contain 10 items, one per blank, with prompts like Blank 1, Blank 2, ...\n"
                "- shared_options must contain exactly 15 entries labeled A. to O.\n"
                "- each shared option must be a single English word only.\n"
                "- answer_key must contain exactly 10 letters, each letter must exist in shared_options, and no letter may repeat."
            )
        if question_type is QuestionType.LONG_READING:
            return (
                "- passage.paragraphs must contain labeled paragraphs beginning with A., B., C. ...\n"
                "- questions must contain 10 English statements for matching, not questions.\n"
                "- answer_key must contain paragraph letters only."
            )
        if question_type is QuestionType.WRITING:
            return (
                "- task_prompt must be an English exam instruction.\n"
                "- passage.paragraphs must contain 2-3 English prompt lines.\n"
                "- questions and answer_key must both be empty.\n"
                "- reference_answer must be a high-scoring English essay."
            )
        if question_type is QuestionType.TRANSLATION:
            return (
                "- task_prompt must be an English exam instruction.\n"
                "- passage.paragraphs must contain the Chinese source passage.\n"
                "- questions and answer_key must both be empty.\n"
                "- reference_answer must be a natural English translation."
            )
        return (
            "- questions must contain exactly 5 items.\n"
            "- each question must contain exactly 4 options labeled A. to D.\n"
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
        payload = self.client.create_json_with_tool_schema(
            system_prompt=self._system_prompt(),
            user_prompt=self._user_prompt(question_set, response_text, duration_seconds),
            tool_name="deliver_subjective_evaluation",
            tool_description="Return one CET subjective-task evaluation as structured JSON.",
            parameters_schema=self._tool_schema(question_set),
            temperature=0.18,
            max_tokens=2600,
        )
        return self._normalize_evaluation_payload(payload, question_set)

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
    ) -> str:
        task_type = "writing" if question_set.question_type is QuestionType.WRITING else "translation"
        return (
            f"Evaluate this CET {task_type} response.\n"
            f"Level: {question_set.level.value}\n"
            f"Task prompt: {question_set.task_prompt}\n"
            f"Passage/task lines: {json.dumps(question_set.passage.paragraphs, ensure_ascii=False)}\n"
            f"Reference answer: {question_set.reference_answer}\n"
            f"Rubric focus: {json.dumps(question_set.rubric_focus, ensure_ascii=False)}\n"
            f"Suggested time used: {duration_seconds} seconds.\n"
            f"Candidate response:\n{response_text}\n\n"
            "Evaluation requirements:\n"
            "- score on a 15-point CET-style scale.\n"
            "- provide 4 score dimensions tied to the rubric.\n"
            "- identify misspelled, misused, or weak words and give corrected English words plus short Chinese reason and Chinese meaning.\n"
            "- identify ungrammatical or awkward sentences and rewrite them in high-quality English.\n"
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
            high_score_version=high_score_version,
            weakness_tags=weakness_tags[:6],
        )
