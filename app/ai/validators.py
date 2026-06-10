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

        if question_type is QuestionType.BANKED_CLOZE:
            self._validate_banked_cloze(normalized, level, actual_word_count, errors)
        elif question_type is QuestionType.LONG_READING:
            self._validate_long_reading(normalized, level, actual_word_count, errors)
        else:
            self._validate_careful_reading(normalized, level, slot, actual_word_count, errors)

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
        if len(payload["answer_key"]) != 5:
            errors.append("仔细阅读 answer_key 必须有 5 项")
        for answer in payload["answer_key"]:
            if answer not in {"A", "B", "C", "D"}:
                errors.append(f"仔细阅读答案必须是 A-D，发现：{answer}")
