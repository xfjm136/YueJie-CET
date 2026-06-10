from __future__ import annotations

from collections import Counter

from app.constants import SKILL_ADVICE, SKILL_LABELS
from app.domain.schemas import (
    AttemptQuestionResult,
    AttemptResult,
    QuestionSet,
    make_id,
)


def grade_attempt(
    question_set: QuestionSet,
    answers: dict[str, str],
    duration_seconds: int,
) -> AttemptResult:
    explanation_map = {
        item.question_id: item for item in question_set.analysis.item_explanations
    }
    results: list[AttemptQuestionResult] = []
    wrong_skills: Counter[str] = Counter()

    for index, question in enumerate(question_set.questions):
        user_answer = answers.get(question.id, "").strip().upper()
        correct_answer = question_set.answer_key[index].strip().upper()
        explanation = explanation_map.get(question.id)
        is_correct = user_answer == correct_answer
        if not is_correct:
            wrong_skills[question.skill_tag] += 1
        results.append(
            AttemptQuestionResult(
                question_id=question.id,
                user_answer=user_answer,
                correct_answer=correct_answer,
                is_correct=is_correct,
                explanation=explanation.explanation if explanation else "",
                skill_tag=question.skill_tag,
            )
        )

    correct_count = sum(1 for item in results if item.is_correct)
    total_count = len(results)
    accuracy = correct_count / total_count if total_count else 0.0

    if accuracy >= 0.9:
        summary = "状态很好，这套题的整体把握比较稳，可以继续提升速度。"
    elif accuracy >= 0.7:
        summary = "整体完成度不错，重点再补一补失分能力点。"
    elif accuracy >= 0.5:
        summary = "基础还能继续巩固，建议先把定位和排除法练扎实。"
    else:
        summary = "这套题暴露出较多薄弱项，建议先看解析再重做一遍。"

    recommendations: list[str] = []
    for skill, _count in wrong_skills.most_common(2):
        recommendations.append(
            f"{SKILL_LABELS.get(skill, skill)}：{SKILL_ADVICE.get(skill, SKILL_ADVICE['general'])}"
        )
    if duration_seconds > 25 * 60:
        recommendations.append(f"{SKILL_LABELS['speed']}：{SKILL_ADVICE['speed']}")
    if not recommendations:
        recommendations.append("继续保持当前节奏，下一套可以尝试更高准确率与更短时间。")

    return AttemptResult(
        id=make_id("attempt"),
        question_set_id=question_set.id,
        correct_count=correct_count,
        total_count=total_count,
        accuracy=accuracy,
        duration_seconds=duration_seconds,
        summary=summary,
        recommendations=recommendations,
        question_results=results,
    )

