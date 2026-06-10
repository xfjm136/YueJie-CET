from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from typing import Any

from app.domain.enums import Level, QuestionType
from app.runtime import build_runtime
from app.ai.mock_data import MockQuestionFactory


def _success(**payload: Any) -> None:
    print(json.dumps({"ok": True, **payload}, ensure_ascii=False))


def _failure(message: str, *, details: Any | None = None, exit_code: int = 1) -> None:
    error_text = message
    if details not in (None, ""):
        error_text = f"{message}: {details}"
    print(json.dumps({"ok": False, "error": error_text, "details": details}, ensure_ascii=False))
    raise SystemExit(exit_code)


def _emit_event(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False), flush=True)


def _parse_level(raw: str) -> Level:
    try:
        return Level(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid level: {raw}") from exc


def _parse_question_type(raw: str) -> QuestionType:
    try:
        return QuestionType(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid question type: {raw}") from exc


def _read_json_stdin() -> dict[str, Any]:
    raw = sys.stdin.read().strip()
    if not raw:
        return {}
    return json.loads(raw)


def command_overview(_args: argparse.Namespace) -> None:
    runtime = build_runtime()
    _success(
        overview=runtime.stats_service.overview_data(),
        settings={
            "theme_mode": runtime.db.get_app_setting("theme_mode", "dark"),
            "background_mode": runtime.db.get_app_setting("background_mode", "opaque"),
            "palette_mode": runtime.db.get_app_setting("palette_mode", "fern"),
            "deepseek_model": runtime.settings.deepseek_model,
        },
    )


def command_type_stats(args: argparse.Namespace) -> None:
    runtime = build_runtime()
    _success(level=args.level.value, cards=runtime.stats_service.level_type_data(args.level))


def command_generate(args: argparse.Namespace) -> None:
    runtime = build_runtime()
    question_set = runtime.question_service.generate_question_set(
        level=args.level,
        question_type=args.question_type,
        slot=args.slot,
    )
    _success(question_set=question_set.to_dict())


def command_generate_live(args: argparse.Namespace) -> None:
    _emit_event("progress", phase="boot", message="正在加载运行时配置与本地题库。")
    runtime = build_runtime()

    def on_progress(phase: str, message: str) -> None:
        _emit_event("progress", phase=phase, message=message)

    try:
        question_set = runtime.question_service.generate_question_set(
            level=args.level,
            question_type=args.question_type,
            slot=args.slot,
            progress_callback=on_progress,
        )
    except Exception as exc:
        _emit_event("error", error=str(exc))
        raise SystemExit(1) from exc

    _emit_event("result", ok=True, question_set=question_set.to_dict())


def command_generate_mock(args: argparse.Namespace) -> None:
    runtime = build_runtime()
    question_set = MockQuestionFactory.create(
        level=args.level,
        question_type=args.question_type,
        slot=args.slot,
        model_name=runtime.settings.deepseek_model,
    )
    runtime.db.save_question_set(question_set)
    runtime.db.upsert_vocabulary_items(question_set.vocabulary)
    _success(question_set=question_set.to_dict())


def command_submit(_args: argparse.Namespace) -> None:
    runtime = build_runtime()
    payload = _read_json_stdin()
    question_set_id = payload.get("question_set_id")
    if not question_set_id:
        _failure("missing question_set_id")
    question_set = runtime.db.get_question_set(question_set_id)
    if question_set is None:
        _failure("question set not found", details=question_set_id)

    started_at_raw = payload.get("started_at")
    if not started_at_raw:
        _failure("missing started_at")
    try:
        started_at = datetime.fromisoformat(started_at_raw)
    except ValueError as exc:
        _failure("invalid started_at", details=str(exc))

    answers = payload.get("answers", {})
    if not isinstance(answers, dict):
        _failure("answers must be a JSON object")
    result = runtime.attempt_service.submit_attempt(
        question_set=question_set,
        answers={str(key): str(value) for key, value in answers.items()},
        started_at=started_at,
        is_history_retry=bool(payload.get("is_history_retry", False)),
    )
    _success(result=result.to_dict())


def command_history(args: argparse.Namespace) -> None:
    runtime = build_runtime()
    _success(history=runtime.stats_service.list_history(limit=args.limit))


def command_review(args: argparse.Namespace) -> None:
    runtime = build_runtime()
    bundle = runtime.db.get_attempt_bundle(args.attempt_id)
    if bundle is None:
        _failure("attempt not found", details=args.attempt_id)
    _success(
        attempt=bundle["attempt"],
        question_set=bundle["question_set"].to_dict(),
        result=bundle["result"].to_dict(),
        answers=bundle["answers"],
    )


def command_vocabulary(args: argparse.Namespace) -> None:
    runtime = build_runtime()
    _success(vocabulary=runtime.stats_service.list_vocabulary(limit=args.limit))


def command_weakness(args: argparse.Namespace) -> None:
    runtime = build_runtime()
    _success(weakness=runtime.stats_service.list_weakness_snapshots(limit=args.limit))


def command_settings(_args: argparse.Namespace) -> None:
    runtime = build_runtime()
    _success(
        settings={
            "theme_mode": runtime.db.get_app_setting("theme_mode", "dark"),
            "background_mode": runtime.db.get_app_setting("background_mode", "opaque"),
            "palette_mode": runtime.db.get_app_setting("palette_mode", "fern"),
            "deepseek_model": runtime.settings.deepseek_model,
            "db_path": str(runtime.settings.db_path),
            "deepseek_base_url": runtime.settings.deepseek_base_url,
            "api_key_configured": bool(runtime.settings.deepseek_api_key),
        }
    )


def command_set_setting(args: argparse.Namespace) -> None:
    runtime = build_runtime()
    runtime.db.set_app_setting(args.key, args.value)
    _success(key=args.key, value=args.value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m app.bridge")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("overview").set_defaults(func=command_overview)

    type_stats = subparsers.add_parser("type-stats")
    type_stats.add_argument("--level", type=_parse_level, required=True)
    type_stats.set_defaults(func=command_type_stats)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--level", type=_parse_level, required=True)
    generate.add_argument("--question-type", type=_parse_question_type, required=True)
    generate.add_argument("--slot", type=int)
    generate.set_defaults(func=command_generate)

    generate_live = subparsers.add_parser("generate-live")
    generate_live.add_argument("--level", type=_parse_level, required=True)
    generate_live.add_argument("--question-type", type=_parse_question_type, required=True)
    generate_live.add_argument("--slot", type=int)
    generate_live.set_defaults(func=command_generate_live)

    generate_mock = subparsers.add_parser("generate-mock")
    generate_mock.add_argument("--level", type=_parse_level, required=True)
    generate_mock.add_argument("--question-type", type=_parse_question_type, required=True)
    generate_mock.add_argument("--slot", type=int)
    generate_mock.set_defaults(func=command_generate_mock)

    submit = subparsers.add_parser("submit")
    submit.set_defaults(func=command_submit)

    history = subparsers.add_parser("history")
    history.add_argument("--limit", type=int, default=30)
    history.set_defaults(func=command_history)

    review = subparsers.add_parser("review")
    review.add_argument("--attempt-id", required=True)
    review.set_defaults(func=command_review)

    vocabulary = subparsers.add_parser("vocabulary")
    vocabulary.add_argument("--limit", type=int, default=100)
    vocabulary.set_defaults(func=command_vocabulary)

    weakness = subparsers.add_parser("weakness")
    weakness.add_argument("--limit", type=int, default=20)
    weakness.set_defaults(func=command_weakness)

    subparsers.add_parser("settings").set_defaults(func=command_settings)

    set_setting = subparsers.add_parser("set-setting")
    set_setting.add_argument("--key", required=True)
    set_setting.add_argument("--value", required=True)
    set_setting.set_defaults(func=command_set_setting)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover - CLI safety net
        _failure("bridge command failed", details=str(exc))


if __name__ == "__main__":
    main()
