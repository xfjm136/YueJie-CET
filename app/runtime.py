from __future__ import annotations

from dataclasses import dataclass

from app.ai.client import DeepSeekClient
from app.ai.pipelines import QuestionGenerationPipeline
from app.config import Settings, get_settings
from app.data.db import Database
from app.services.attempt_service import AttemptService
from app.services.question_service import QuestionService
from app.services.stats_service import StatsService
from app.services.weakness_service import WeaknessService


@dataclass
class Runtime:
    settings: Settings
    db: Database
    question_service: QuestionService
    attempt_service: AttemptService
    stats_service: StatsService
    weakness_service: WeaknessService


def build_runtime() -> Runtime:
    settings = get_settings()
    db = Database(settings.db_path)
    db.init_schema()

    client = (
        DeepSeekClient(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            model=settings.deepseek_model,
            timeout=settings.request_timeout,
        )
        if settings.deepseek_api_key
        else None
    )
    pipeline = QuestionGenerationPipeline(client, settings.deepseek_model)
    weakness_service = WeaknessService(db)
    question_service = QuestionService(db, pipeline)
    attempt_service = AttemptService(db, weakness_service)
    stats_service = StatsService(db)
    return Runtime(
        settings=settings,
        db=db,
        question_service=question_service,
        attempt_service=attempt_service,
        stats_service=stats_service,
        weakness_service=weakness_service,
    )

