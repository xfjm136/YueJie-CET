from __future__ import annotations

from typing import Callable

from app.ai.pipelines import QuestionGenerationPipeline
from app.data.db import Database
from app.domain.enums import Level, QuestionType
from app.domain.schemas import QuestionSet


class QuestionService:
    def __init__(self, db: Database, pipeline: QuestionGenerationPipeline) -> None:
        self.db = db
        self.pipeline = pipeline

    def generate_question_set(
        self,
        level: Level,
        question_type: QuestionType,
        slot: int | None = None,
        weakness_summary: str | None = None,
        progress_callback: Callable[[str, str], None] | None = None,
    ) -> QuestionSet:
        if progress_callback is not None:
            progress_callback("prepare", "正在读取题型统计、词汇库与薄弱项信息。")
        final_weakness_summary = weakness_summary or self.db.latest_weakness_summary(level, question_type)
        question_set = self.pipeline.generate(
            level,
            question_type,
            slot,
            final_weakness_summary,
            progress_callback=progress_callback,
        )
        if progress_callback is not None:
            progress_callback("save", "题目已通过校验，正在保存题集与解析。")
        self.db.save_question_set(question_set)
        if progress_callback is not None:
            progress_callback("done", "题目、解析与词汇均已准备完成。")
        return question_set
