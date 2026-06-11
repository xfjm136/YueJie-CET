from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from app.domain.enums import Level, QuestionType
from app.runtime import build_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-generate CET samples for manual review."
    )
    parser.add_argument("--level", choices=["cet4", "cet6"], required=True)
    parser.add_argument(
        "--question-type",
        choices=[
            "banked_cloze",
            "long_reading",
            "careful_reading",
            "writing",
            "translation",
            "all",
        ],
        default="all",
    )
    parser.add_argument("--slot", type=int, choices=[1, 2])
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument(
        "--output-dir",
        default="data/sample-review",
        help="Directory for generated review bundles.",
    )
    return parser.parse_args()


def question_type_targets(
    question_type: str,
    slot: int | None,
) -> list[tuple[QuestionType, int | None]]:
    if question_type == "all":
        return [
            (QuestionType.BANKED_CLOZE, None),
            (QuestionType.LONG_READING, None),
            (QuestionType.CAREFUL_READING, 1),
            (QuestionType.CAREFUL_READING, 2),
            (QuestionType.WRITING, None),
            (QuestionType.TRANSLATION, None),
        ]
    mapping = {
        "banked_cloze": QuestionType.BANKED_CLOZE,
        "long_reading": QuestionType.LONG_READING,
        "careful_reading": QuestionType.CAREFUL_READING,
        "writing": QuestionType.WRITING,
        "translation": QuestionType.TRANSLATION,
    }
    qtype = mapping[question_type]
    if qtype is QuestionType.CAREFUL_READING:
        if slot is not None:
            return [(qtype, slot)]
        return [(qtype, 1), (qtype, 2)]
    return [(qtype, None)]


def summarize(question_set: Any) -> dict[str, Any]:
    prompts = [item.prompt for item in question_set.questions]
    skills = Counter(item.skill_tag for item in question_set.questions)
    return {
        "id": question_set.id,
        "level": question_set.level.value,
        "question_type": question_set.question_type.value,
        "slot": question_set.slot,
        "title": question_set.title,
        "topic": question_set.topic,
        "source_type": question_set.source_type,
        "generator_model": question_set.generator_model,
        "word_count": question_set.word_count,
        "question_count": len(question_set.questions),
        "skills": dict(skills),
        "prompts_preview": prompts[:5],
        "analysis_summary": question_set.analysis.overall_summary,
        "test_tips": question_set.analysis.test_tips,
        "answer_key": question_set.answer_key,
    }


def main() -> None:
    args = parse_args()
    runtime = build_runtime()
    level = Level(args.level)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    all_summaries: list[dict[str, Any]] = []
    for question_type, slot in question_type_targets(args.question_type, args.slot):
        bucket: list[dict[str, Any]] = []
        for index in range(args.count):
            question_set = runtime.question_service.generate_question_set(
                level=level,
                question_type=question_type,
                slot=slot,
            )
            summary = summarize(question_set)
            bucket.append(summary)
            all_summaries.append(summary)
            payload_path = output_root / f"{stamp}-{level.value}-{question_type.value}-{slot or 0}-{index + 1}.json"
            payload_path.write_text(
                json.dumps(question_set.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        summary_path = output_root / f"{stamp}-{level.value}-{question_type.value}-{slot or 0}-summary.json"
        summary_path.write_text(
            json.dumps(bucket, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    review_index = output_root / f"{stamp}-{level.value}-review-index.json"
    review_index.write_text(
        json.dumps(all_summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"ok": True, "count": len(all_summaries), "output": str(review_index)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
