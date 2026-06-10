from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.domain.enums import Level, QuestionType


@dataclass
class QuestionSetValidationError(Exception):
    errors: list[str]

    def __str__(self) -> str:
        return "；".join(self.errors)


class CETQuestionValidator:
    BANKED_CLOZE_RANGES = {
        Level.CET4: (200, 250),
        Level.CET6: (250, 300),
    }
    LONG_READING_RANGES = {
        Level.CET4: (850, 1150),
        Level.CET6: (1050, 1350),
    }
    CAREFUL_READING_RANGES = {
        Level.CET4: (290, 360),
        Level.CET6: (390, 470),
    }
    CAREFUL_READING_SKILLS = {
        "main_idea",
        "detail",
        "inference",
        "vocabulary_in_context",
        "attitude",
    }
    WRITING_MIN_WORDS = {
        Level.CET4: 120,
        Level.CET6: 150,
    }
    TRANSLATION_CHAR_RANGES = {
        Level.CET4: (140, 160),
        Level.CET6: (180, 200),
    }

    def validate(
        self,
        payload: dict[str, Any],
        level: Level,
        question_type: QuestionType,
        slot: int | None,
    ) -> dict[str, Any]:
        normalized = self._normalize(payload, level, question_type, slot)
        errors: list[str] = []

        for key in ("title", "topic", "passage", "questions", "answer_key", "analysis", "vocabulary"):
            if key not in normalized:
                errors.append(f"缺少字段：{key}")

        passage = normalized.get("passage", {})
        paragraphs = passage.get("paragraphs", [])
        if not isinstance(paragraphs, list) or not paragraphs:
            errors.append("passage.paragraphs 必须为非空列表")

        actual_word_count = self._word_count(paragraphs)
        normalized["word_count"] = actual_word_count

        questions = normalized.get("questions", [])
        answer_key = normalized.get("answer_key", [])
        if len({item.get("id") for item in questions if isinstance(item, dict)}) != len(questions):
            errors.append("questions.id 必须唯一")

        analysis = normalized.get("analysis", {})
        item_explanations = analysis.get("item_explanations", [])
        if len(item_explanations) != len(questions):
            errors.append("analysis.item_explanations 数量必须与 questions 一致")
        if len(analysis.get("test_tips", [])) != 3:
            errors.append("analysis.test_tips 必须正好提供 3 条")

        explanation_ids = {item.get("question_id") for item in item_explanations if isinstance(item, dict)}
        question_ids = {item.get("id") for item in questions if isinstance(item, dict)}
        if explanation_ids and question_ids and explanation_ids != question_ids:
            errors.append("analysis.item_explanations 的 question_id 必须与 questions.id 一一对应")

        if len(normalized.get("vocabulary", [])) < 3:
            errors.append("vocabulary 至少应提供 3 个重点词汇")

        self._validate_language_boundaries(normalized, question_type, errors)

        if question_type is QuestionType.BANKED_CLOZE:
            self._validate_banked_cloze(normalized, level, actual_word_count, errors)
        elif question_type is QuestionType.LONG_READING:
            self._validate_long_reading(normalized, level, actual_word_count, errors)
        elif question_type is QuestionType.CAREFUL_READING:
            self._validate_careful_reading(normalized, level, slot, actual_word_count, errors)
        elif question_type is QuestionType.WRITING:
            self._validate_writing(normalized, level, errors)
        else:
            self._validate_translation(normalized, level, errors)

        if errors:
            raise QuestionSetValidationError(errors)
        return normalized

    def _normalize(
        self,
        payload: dict[str, Any],
        level: Level,
        question_type: QuestionType,
        slot: int | None,
    ) -> dict[str, Any]:
        normalized = dict(payload)
        normalized["title"] = str(normalized.get("title", "")).strip()
        normalized["topic"] = str(normalized.get("topic", "")).strip()
        normalized["shared_options"] = list(normalized.get("shared_options", []))
        normalized["task_prompt"] = str(normalized.get("task_prompt", "")).strip()
        normalized["reference_answer"] = str(normalized.get("reference_answer", "")).strip()
        normalized["rubric_focus"] = [
            str(item).strip() for item in normalized.get("rubric_focus", [])
        ]
        normalized["min_response_words"] = int(normalized.get("min_response_words", 0) or 0)
        normalized["max_response_words"] = int(normalized.get("max_response_words", 0) or 0)
        normalized["passage"] = dict(normalized.get("passage", {}))
        normalized["passage"]["title"] = str(normalized["passage"].get("title", normalized["title"])).strip()
        normalized["passage"]["paragraphs"] = [
            str(item).strip() for item in normalized["passage"].get("paragraphs", [])
        ]

        normalized["questions"] = [
            {
                "id": str(item.get("id", "")).strip(),
                "prompt": str(item.get("prompt", "")).strip(),
                "options": [str(option).strip() for option in item.get("options", [])],
                "skill_tag": str(item.get("skill_tag", "general")).strip() or "general",
                "hint": item.get("hint"),
            }
            for item in normalized.get("questions", [])
        ]
        normalized["answer_key"] = [str(item).strip().upper() for item in normalized.get("answer_key", [])]

        analysis = dict(normalized.get("analysis", {}))
        analysis["overall_strategy"] = str(analysis.get("overall_strategy", "")).strip()
        analysis["overall_summary"] = str(analysis.get("overall_summary", "")).strip()
        analysis["test_tips"] = [str(item).strip() for item in analysis.get("test_tips", [])]
        analysis["item_explanations"] = [
            {
                "question_id": str(item.get("question_id", "")).strip(),
                "correct_answer": str(item.get("correct_answer", "")).strip().upper(),
                "explanation": str(item.get("explanation", "")).strip(),
                "skill_tag": str(item.get("skill_tag", "general")).strip() or "general",
            }
            for item in analysis.get("item_explanations", [])
        ]
        normalized["analysis"] = analysis

        normalized["vocabulary"] = [
            {
                "lemma": str(item.get("lemma", "")).strip(),
                "surface_form": str(item.get("surface_form", item.get("lemma", ""))).strip(),
                "level_hint": str(item.get("level_hint", level.value)).strip() or level.value,
                "meaning_zh": str(item.get("meaning_zh", "")).strip(),
                "example_en": str(item.get("example_en", "")).strip(),
            }
            for item in normalized.get("vocabulary", [])
        ]
        normalized["slot"] = slot
        normalized["level"] = level.value
        normalized["question_type"] = question_type.value
        return normalized

    @staticmethod
    def _word_count(paragraphs: list[str]) -> int:
        return len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", " ".join(paragraphs)))

    @staticmethod
    def _char_count(text: str) -> int:
        return len(re.findall(r"[\u4e00-\u9fff]", text))

    @staticmethod
    def _contains_cjk(text: str) -> bool:
        return bool(re.search(r"[\u4e00-\u9fff]", text))

    @staticmethod
    def _looks_like_single_english_word(text: str) -> bool:
        return bool(re.fullmatch(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", text))

    def _validate_language_boundaries(
        self,
        payload: dict[str, Any],
        question_type: QuestionType,
        errors: list[str],
    ) -> None:
        english_only_fields = [
            ("title", payload.get("title", "")),
            ("topic", payload.get("topic", "")),
            ("passage.title", payload.get("passage", {}).get("title", "")),
            ("task_prompt", payload.get("task_prompt", "")),
            ("reference_answer", payload.get("reference_answer", "")),
        ]
        for field_name, value in english_only_fields:
            text = str(value).strip()
            if not text:
                if field_name not in {"task_prompt", "reference_answer"}:
                    errors.append(f"{field_name} 不能为空")
            elif self._contains_cjk(text):
                errors.append(f"{field_name} 必须为英文，不应包含中文")

        for index, paragraph in enumerate(payload.get("passage", {}).get("paragraphs", []), start=1):
            if question_type is QuestionType.TRANSLATION:
                if not self._contains_cjk(paragraph):
                    errors.append(f"passage.paragraphs 第 {index} 段必须为中文原文")
            elif self._contains_cjk(paragraph):
                errors.append(f"passage.paragraphs 第 {index} 段必须为英文，不应包含中文")

        for index, question in enumerate(payload.get("questions", []), start=1):
            prompt = str(question.get("prompt", "")).strip()
            if not prompt:
                errors.append(f"questions 第 {index} 题 prompt 不能为空")
            elif self._contains_cjk(prompt):
                errors.append(f"questions 第 {index} 题 prompt 必须为英文，不应包含中文")
            for option_index, option in enumerate(question.get("options", []), start=1):
                if self._contains_cjk(option):
                    errors.append(
                        f"questions 第 {index} 题 option {option_index} 必须为英文，不应包含中文"
                    )

        for index, option in enumerate(payload.get("shared_options", []), start=1):
            if self._contains_cjk(option):
                errors.append(f"shared_options 第 {index} 项必须为英文，不应包含中文")

        for index, item in enumerate(payload.get("vocabulary", []), start=1):
            if self._contains_cjk(item.get("lemma", "")):
                errors.append(f"vocabulary 第 {index} 项 lemma 必须为英文")
            if self._contains_cjk(item.get("surface_form", "")):
                errors.append(f"vocabulary 第 {index} 项 surface_form 必须为英文")
            example_en = str(item.get("example_en", "")).strip()
            if example_en and self._contains_cjk(example_en):
                errors.append(f"vocabulary 第 {index} 项 example_en 必须为英文")
            if not self._contains_cjk(item.get("meaning_zh", "")):
                errors.append(f"vocabulary 第 {index} 项 meaning_zh 应提供中文释义")

    def _validate_banked_cloze(
        self,
        payload: dict[str, Any],
        level: Level,
        actual_word_count: int,
        errors: list[str],
    ) -> None:
        lower, upper = self.BANKED_CLOZE_RANGES[level]
        if not lower <= actual_word_count <= upper:
            errors.append(f"选词填空篇章词数应在 {lower}-{upper} 之间，当前为 {actual_word_count}")

        questions = payload["questions"]
        if len(questions) != 10:
            errors.append("选词填空必须提供 10 个空")
        for index, question in enumerate(questions, start=1):
            if question.get("prompt") != f"Blank {index}":
                errors.append(f"选词填空第 {index} 题 prompt 必须为 Blank {index}")
            if question.get("options"):
                errors.append(f"选词填空第 {index} 题不应包含独立 options")

        options = payload.get("shared_options", [])
        if len(options) != 15:
            errors.append("选词填空必须提供 15 个共享选项")

        option_letters = []
        normalized_options = []
        for raw in options:
            match = re.match(r"^([A-O])[\.\)]\s+(.+)$", raw)
            if not match:
                errors.append(f"共享选项格式不正确：{raw}")
                continue
            letter, content = match.groups()
            if not self._looks_like_single_english_word(content.strip()):
                errors.append(f"共享选项 {letter} 必须是单个英文词，不可为短语：{content.strip()}")
            option_letters.append(letter)
            normalized_options.append(f"{letter}. {content.strip()}")
        if len(set(option_letters)) != len(option_letters):
            errors.append("共享选项字母必须唯一")
        payload["shared_options"] = normalized_options

        if len(payload["answer_key"]) != 10:
            errors.append("选词填空 answer_key 必须有 10 项")
        if len(set(payload["answer_key"])) != len(payload["answer_key"]):
            errors.append("选词填空 answer_key 不能重复使用同一选项")
        for answer in payload["answer_key"]:
            if answer not in option_letters:
                errors.append(f"answer_key 中存在不在选项范围内的答案：{answer}")

        passage_text = " ".join(payload["passage"]["paragraphs"])
        for index in range(1, 11):
            if f"[{index}]" not in passage_text:
                errors.append("选词填空篇章需用 [1]...[10] 标记空位")
                break

    def _validate_long_reading(
        self,
        payload: dict[str, Any],
        level: Level,
        actual_word_count: int,
        errors: list[str],
    ) -> None:
        lower, upper = self.LONG_READING_RANGES[level]
        if not lower <= actual_word_count <= upper:
            errors.append(f"长篇阅读词数应在 {lower}-{upper} 之间，当前为 {actual_word_count}")

        questions = payload["questions"]
        if len(questions) != 10:
            errors.append("长篇阅读必须提供 10 个匹配题")
        for index, question in enumerate(questions, start=1):
            if question.get("options"):
                errors.append(f"长篇阅读第 {index} 题不应包含独立 options")
            prompt = str(question.get("prompt", "")).strip()
            if prompt.endswith("?"):
                errors.append(f"长篇阅读第 {index} 题必须写成陈述句，不应为问句")
            if re.match(r"^[A-Z][\.\)]\s", prompt):
                errors.append(f"长篇阅读第 {index} 题 prompt 不应包含段落标签前缀")
            if len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", prompt)) < 5:
                errors.append(f"长篇阅读第 {index} 题 prompt 过短，不像完整匹配陈述句")

        paragraphs = payload["passage"]["paragraphs"]
        if not 10 <= len(paragraphs) <= 14:
            errors.append("长篇阅读段落数建议在 10-14 段之间")

        labels = []
        for paragraph in paragraphs:
            match = re.match(r"^([A-Z])\.\s", paragraph)
            if match:
                labels.append(match.group(1))
        if len(labels) < 10:
            errors.append("长篇阅读段落需使用 A. / B. 形式标号")
        else:
            expected = [chr(ord("A") + index) for index in range(len(labels))]
            if labels != expected:
                errors.append("长篇阅读段落标签必须按 A. B. C. 顺序连续排列")

        if len(payload["answer_key"]) != 10:
            errors.append("长篇阅读 answer_key 必须有 10 项")
        for answer in payload["answer_key"]:
            if answer not in labels:
                errors.append(f"长篇阅读答案 {answer} 不在段落标签范围内")
        if len(set(payload["answer_key"])) == len(payload["answer_key"]) and len(labels) == len(
            payload["answer_key"]
        ):
            errors.append("长篇阅读需体现冗余段或复用段，不能刚好一题对应一段且无冗余")

    def _validate_careful_reading(
        self,
        payload: dict[str, Any],
        level: Level,
        slot: int | None,
        actual_word_count: int,
        errors: list[str],
    ) -> None:
        lower, upper = self.CAREFUL_READING_RANGES[level]
        if not lower <= actual_word_count <= upper:
            errors.append(f"仔细阅读词数应在 {lower}-{upper} 之间，当前为 {actual_word_count}")

        questions = payload["questions"]
        if len(questions) != 5:
            errors.append("仔细阅读必须提供 5 道单选题")
        skill_tags = [question.get("skill_tag", "") for question in questions]
        if set(skill_tags) != self.CAREFUL_READING_SKILLS or len(set(skill_tags)) != 5:
            errors.append(
                "仔细阅读 5 题 skill_tag 必须恰好覆盖 main_idea/detail/inference/vocabulary_in_context/attitude 各一次"
            )
        for index, question in enumerate(questions, start=1):
            if len(question.get("options", [])) != 4:
                errors.append(f"仔细阅读第 {index} 题必须提供 4 个选项")
            option_letters = [option[:1] for option in question.get("options", [])]
            if option_letters != ["A", "B", "C", "D"]:
                errors.append(f"仔细阅读第 {index} 题 options 必须按 A-D 标注")
            normalized_options = [option[3:].strip().lower() for option in question.get("options", []) if len(option) >= 3]
            if len(set(normalized_options)) != len(normalized_options):
                errors.append(f"仔细阅读第 {index} 题 4 个选项内容必须互不重复")
            if any(option in {"all of the above", "none of the above"} for option in normalized_options):
                errors.append(f"仔细阅读第 {index} 题不得使用 All/None of the above 类选项")
            prompt = str(question.get("prompt", "")).strip().lower()
            vocab_like = (
                "most likely means" in prompt
                or "most nearly means" in prompt
                or "closest in meaning" in prompt
                or "refers to" in prompt
            )
            if not vocab_like and not prompt.endswith("?"):
                errors.append(f"仔细阅读第 {index} 题题干应为标准英文问句")
        if len(payload["answer_key"]) != 5:
            errors.append("仔细阅读 answer_key 必须有 5 项")
        for answer in payload["answer_key"]:
            if answer not in {"A", "B", "C", "D"}:
                errors.append(f"仔细阅读答案必须是 A-D，发现：{answer}")

    def _validate_writing(
        self,
        payload: dict[str, Any],
        level: Level,
        errors: list[str],
    ) -> None:
        if payload["questions"]:
            errors.append("写作题不应包含客观题 questions")
        if payload["answer_key"]:
            errors.append("写作题不应包含 answer_key")
        if payload["shared_options"]:
            errors.append("写作题不应包含 shared_options")
        if not payload.get("task_prompt"):
            errors.append("写作题必须提供 task_prompt")
        if not payload.get("reference_answer"):
            errors.append("写作题必须提供 reference_answer")
        if len(payload["passage"]["paragraphs"]) < 2:
            errors.append("写作题题面说明至少应包含 2 行英文提示")
        min_words = self.WRITING_MIN_WORDS[level]
        if payload.get("min_response_words", 0) < min_words:
            errors.append(f"写作题 min_response_words 不应低于 {min_words}")
        reference_word_count = self._word_count([payload.get("reference_answer", "")])
        payload["word_count"] = reference_word_count
        if reference_word_count < min_words:
            errors.append(f"写作题参考范文词数不应低于 {min_words}，当前为 {reference_word_count}")
        if len(payload.get("rubric_focus", [])) < 4:
            errors.append("写作题 rubric_focus 至少应包含 4 个评分维度")

    def _validate_translation(
        self,
        payload: dict[str, Any],
        level: Level,
        errors: list[str],
    ) -> None:
        if payload["questions"]:
            errors.append("翻译题不应包含客观题 questions")
        if payload["answer_key"]:
            errors.append("翻译题不应包含 answer_key")
        if payload["shared_options"]:
            errors.append("翻译题不应包含 shared_options")
        if not payload.get("task_prompt"):
            errors.append("翻译题必须提供 task_prompt")
        if not payload.get("reference_answer"):
            errors.append("翻译题必须提供 reference_answer")
        if len(payload["passage"]["paragraphs"]) != 1:
            errors.append("翻译题应提供 1 段中文原文")
        source_text = " ".join(payload["passage"]["paragraphs"])
        if not self._contains_cjk(source_text):
            errors.append("翻译题原文必须为中文")
        lower, upper = self.TRANSLATION_CHAR_RANGES[level]
        char_count = self._char_count(source_text)
        payload["word_count"] = char_count
        if not lower <= char_count <= upper:
            errors.append(f"翻译题中文原文字数应在 {lower}-{upper} 汉字之间，当前为 {char_count}")
        if self._contains_cjk(payload.get("reference_answer", "")):
            errors.append("翻译题参考译文必须为英文，不应包含中文")
        if len(payload.get("rubric_focus", [])) < 4:
            errors.append("翻译题 rubric_focus 至少应包含 4 个评分维度")
