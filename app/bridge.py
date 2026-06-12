from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
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


def _heartbeat_message(phase: str, elapsed_total: int, elapsed_phase: int) -> str:
    if phase in {"boot", "prepare"}:
        return f"运行时仍在准备环境与本地数据，已等待 {elapsed_total} 秒。"
    if phase == "blueprint":
        return f"仍在整理四六级题面蓝图与难度约束，已等待 {elapsed_total} 秒。"
    if phase in {"generate_request", "retry_generation"}:
        return (
            f"DeepSeek 正在生成题目正文、题干与解析，当前阶段已等待 {elapsed_phase} 秒，"
            f"累计 {elapsed_total} 秒。"
        )
    if phase == "validate":
        return f"正在校验 JSON 结构与 CET 规范细项，累计已等待 {elapsed_total} 秒。"
    if phase == "repair":
        return f"正在根据校验结果修复题目结构，当前阶段已等待 {elapsed_phase} 秒。"
    if phase in {"save", "done"}:
        return f"题目已接近完成，正在写入本地题库与词汇表，累计 {elapsed_total} 秒。"
    return f"AI 生成流程仍在继续，累计已等待 {elapsed_total} 秒。"


def _submit_heartbeat_message(phase: str, elapsed_total: int, elapsed_phase: int) -> str:
    if phase == "boot":
        return f"正在加载评分运行时与本地题库，已等待 {elapsed_total} 秒。"
    if phase == "prepare":
        return f"正在整理答案、评分维度与题目元数据，已等待 {elapsed_total} 秒。"
    if phase == "score_request":
        return (
            f"DeepSeek 正在评分并生成批注，当前阶段已等待 {elapsed_phase} 秒，"
            f"累计 {elapsed_total} 秒。"
        )
    if phase == "analysis":
        return f"正在整理错词、病句改写与高分版本，累计已等待 {elapsed_total} 秒。"
    if phase == "save":
        return f"正在保存本次记录、词汇与薄弱项，累计已等待 {elapsed_total} 秒。"
    if phase == "grade":
        return f"正在整理客观题判分结果，累计已等待 {elapsed_total} 秒。"
    return f"AI 评分流程仍在继续，累计已等待 {elapsed_total} 秒。"


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
    progress_queue: queue.Queue[tuple[str, str]] = queue.Queue()
    result_queue: queue.Queue[tuple[str, Any]] = queue.Queue()

    def on_progress(phase: str, message: str) -> None:
        progress_queue.put((phase, message))

    def worker() -> None:
        try:
            question_set = runtime.question_service.generate_question_set(
                level=args.level,
                question_type=args.question_type,
                slot=args.slot,
                progress_callback=on_progress,
            )
        except Exception as exc:
            result_queue.put(("error", exc))
            return
        result_queue.put(("result", question_set))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    current_phase = "boot"
    phase_started_at = time.monotonic()
    total_started_at = phase_started_at
    next_heartbeat_at = total_started_at + 6

    while True:
        drained_progress = False
        while True:
            try:
                phase, message = progress_queue.get_nowait()
            except queue.Empty:
                break
            drained_progress = True
            current_phase = phase
            phase_started_at = time.monotonic()
            next_heartbeat_at = phase_started_at + 6
            _emit_event("progress", phase=phase, message=message)

        try:
            status, payload = result_queue.get(timeout=0.8)
        except queue.Empty:
            if not drained_progress:
                now = time.monotonic()
                if now >= next_heartbeat_at:
                    _emit_event(
                        "progress",
                        phase=current_phase,
                        message=_heartbeat_message(
                            current_phase,
                            int(now - total_started_at),
                            int(now - phase_started_at),
                        ),
                    )
                    next_heartbeat_at = now + 6
            if thread.is_alive():
                continue
            thread.join(timeout=0.1)
            if result_queue.empty():
                _emit_event("error", error="生成线程已退出，但未返回结果。")
                raise SystemExit(1)
            continue

        while True:
            try:
                phase, message = progress_queue.get_nowait()
            except queue.Empty:
                break
            current_phase = phase
            phase_started_at = time.monotonic()
            _emit_event("progress", phase=phase, message=message)

        if status == "error":
            _emit_event("error", error=str(payload))
            raise SystemExit(1) from payload

        question_set = payload
        break

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


def command_submit_live(_args: argparse.Namespace) -> None:
    payload = _read_json_stdin()
    _emit_event("progress", phase="boot", message="正在加载评分运行时与本地题库。")
    runtime = build_runtime()

    question_set_id = payload.get("question_set_id")
    if not question_set_id:
        _emit_event("error", error="missing question_set_id")
        raise SystemExit(1)
    question_set = runtime.db.get_question_set(question_set_id)
    if question_set is None:
        _emit_event("error", error=f"question set not found: {question_set_id}")
        raise SystemExit(1)

    started_at_raw = payload.get("started_at")
    if not started_at_raw:
        _emit_event("error", error="missing started_at")
        raise SystemExit(1)
    try:
        started_at = datetime.fromisoformat(started_at_raw)
    except ValueError as exc:
        _emit_event("error", error=f"invalid started_at: {exc}")
        raise SystemExit(1) from exc

    answers = payload.get("answers", {})
    if not isinstance(answers, dict):
        _emit_event("error", error="answers must be a JSON object")
        raise SystemExit(1)

    progress_queue: queue.Queue[tuple[str, str]] = queue.Queue()
    result_queue: queue.Queue[tuple[str, Any]] = queue.Queue()

    def on_progress(phase: str, message: str) -> None:
        progress_queue.put((phase, message))

    def worker() -> None:
        try:
            result = runtime.attempt_service.submit_attempt(
                question_set=question_set,
                answers={str(key): str(value) for key, value in answers.items()},
                started_at=started_at,
                is_history_retry=bool(payload.get("is_history_retry", False)),
                progress_callback=on_progress,
            )
        except Exception as exc:
            result_queue.put(("error", exc))
            return
        result_queue.put(("result", result))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    current_phase = "boot"
    phase_started_at = time.monotonic()
    total_started_at = phase_started_at
    next_heartbeat_at = total_started_at + 6

    while True:
        drained_progress = False
        while True:
            try:
                phase, message = progress_queue.get_nowait()
            except queue.Empty:
                break
            drained_progress = True
            current_phase = phase
            phase_started_at = time.monotonic()
            next_heartbeat_at = phase_started_at + 6
            _emit_event("progress", phase=phase, message=message)

        try:
            status, result_payload = result_queue.get(timeout=0.8)
        except queue.Empty:
            if not drained_progress:
                now = time.monotonic()
                if now >= next_heartbeat_at:
                    _emit_event(
                        "progress",
                        phase=current_phase,
                        message=_submit_heartbeat_message(
                            current_phase,
                            int(now - total_started_at),
                            int(now - phase_started_at),
                        ),
                    )
                    next_heartbeat_at = now + 6
            if thread.is_alive():
                continue
            thread.join(timeout=0.1)
            if result_queue.empty():
                _emit_event("error", error="评分线程已退出，但未返回结果。")
                raise SystemExit(1)
            continue

        while True:
            try:
                phase, message = progress_queue.get_nowait()
            except queue.Empty:
                break
            current_phase = phase
            phase_started_at = time.monotonic()
            _emit_event("progress", phase=phase, message=message)

        if status == "error":
            _emit_event("error", error=str(result_payload))
            raise SystemExit(1) from result_payload

        result = result_payload
        break

    _emit_event("result", ok=True, result=result.to_dict())


def command_history(args: argparse.Namespace) -> None:
    runtime = build_runtime()
    _success(history=runtime.stats_service.list_history(limit=args.limit))


def command_delete_history(args: argparse.Namespace) -> None:
    runtime = build_runtime()
    try:
        deleted = runtime.attempt_service.delete_attempt_history(args.attempt_id)
    except ValueError:
        _failure("attempt not found", details=args.attempt_id)
    _success(deleted=deleted)


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


def command_mock_exam_history(args: argparse.Namespace) -> None:
    runtime = build_runtime()
    _success(history=runtime.stats_service.list_mock_exam_history(limit=args.limit))


def command_mock_exam_review(args: argparse.Namespace) -> None:
    runtime = build_runtime()
    record = runtime.db.get_mock_exam_record(args.exam_id)
    if record is None:
        _failure("mock exam not found", details=args.exam_id)
    _success(mock_exam=record.to_dict())


def command_mock_exam_delete(args: argparse.Namespace) -> None:
    runtime = build_runtime()
    deleted = runtime.db.delete_mock_exam_record(args.exam_id)
    if deleted is None:
        _failure("mock exam not found", details=args.exam_id)
    runtime.weakness_service.rebuild_snapshots()
    _success(deleted=deleted)


def command_mock_exam_weakness(args: argparse.Namespace) -> None:
    runtime = build_runtime()
    _success(weakness=runtime.stats_service.list_mock_exam_weakness_snapshots(limit=args.limit))


def command_submit_mock_exam(_args: argparse.Namespace) -> None:
    runtime = build_runtime()
    payload = _read_json_stdin()
    level_raw = str(payload.get("level", "")).strip()
    if not level_raw:
        _failure("missing level")
    try:
        level = Level(level_raw)
    except ValueError as exc:
        _failure("invalid level", details=str(exc))
    started_at_raw = payload.get("started_at")
    if not started_at_raw:
        _failure("missing started_at")
    try:
        started_at = datetime.fromisoformat(started_at_raw)
    except ValueError as exc:
        _failure("invalid started_at", details=str(exc))
    sections = payload.get("sections", [])
    if not isinstance(sections, list) or not sections:
        _failure("sections must be a non-empty array")
    duration_seconds = payload.get("duration_seconds")
    duration_override: int | None = None
    if duration_seconds not in (None, ""):
        try:
            duration_override = max(1, int(duration_seconds))
        except (TypeError, ValueError) as exc:
            _failure("invalid duration_seconds", details=str(exc))
    record = runtime.mock_exam_service.submit_mock_exam(
        level=level,
        sections=sections,
        started_at=started_at,
        duration_seconds_override=duration_override,
    )
    _success(mock_exam=record.to_dict())


def command_submit_mock_exam_live(_args: argparse.Namespace) -> None:
    payload = _read_json_stdin()
    _emit_event("progress", phase="boot", message="正在加载模拟四六级考试评分运行时。")
    runtime = build_runtime()

    level_raw = str(payload.get("level", "")).strip()
    if not level_raw:
        _emit_event("error", error="missing level")
        raise SystemExit(1)
    try:
        level = Level(level_raw)
    except ValueError as exc:
        _emit_event("error", error=f"invalid level: {exc}")
        raise SystemExit(1) from exc

    started_at_raw = payload.get("started_at")
    if not started_at_raw:
        _emit_event("error", error="missing started_at")
        raise SystemExit(1)
    try:
        started_at = datetime.fromisoformat(started_at_raw)
    except ValueError as exc:
        _emit_event("error", error=f"invalid started_at: {exc}")
        raise SystemExit(1) from exc

    sections = payload.get("sections", [])
    if not isinstance(sections, list) or not sections:
        _emit_event("error", error="sections must be a non-empty array")
        raise SystemExit(1)

    duration_seconds = payload.get("duration_seconds")
    duration_override: int | None = None
    if duration_seconds not in (None, ""):
        try:
            duration_override = max(1, int(duration_seconds))
        except (TypeError, ValueError) as exc:
            _emit_event("error", error=f"invalid duration_seconds: {exc}")
            raise SystemExit(1) from exc

    progress_queue: queue.Queue[tuple[str, str]] = queue.Queue()
    result_queue: queue.Queue[tuple[str, Any]] = queue.Queue()

    def on_progress(phase: str, message: str) -> None:
        progress_queue.put((phase, message))

    def worker() -> None:
        try:
            progress_queue.put(("prepare", "正在整理整套试卷答案与评分维度。"))
            progress_queue.put(("score_request", "正在向 DeepSeek 请求整套模拟四六级考试评分。"))
            record = runtime.mock_exam_service.submit_mock_exam(
                level=level,
                sections=sections,
                started_at=started_at,
                duration_seconds_override=duration_override,
            )
        except Exception as exc:
            result_queue.put(("error", exc))
            return
        progress_queue.put(("analysis", "评分结果已返回，正在整理分项得分、弱势点与建议。"))
        result_queue.put(("result", record))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    current_phase = "boot"
    phase_started_at = time.monotonic()
    total_started_at = phase_started_at
    next_heartbeat_at = total_started_at + 6

    while True:
        drained_progress = False
        while True:
            try:
                phase, message = progress_queue.get_nowait()
            except queue.Empty:
                break
            drained_progress = True
            current_phase = phase
            phase_started_at = time.monotonic()
            next_heartbeat_at = phase_started_at + 6
            _emit_event("progress", phase=phase, message=message)

        try:
            status, result_payload = result_queue.get(timeout=0.8)
        except queue.Empty:
            if not drained_progress:
                now = time.monotonic()
                if now >= next_heartbeat_at:
                    _emit_event(
                        "progress",
                        phase=current_phase,
                        message=_submit_heartbeat_message(
                            current_phase,
                            int(now - total_started_at),
                            int(now - phase_started_at),
                        ),
                    )
                    next_heartbeat_at = now + 6
            if thread.is_alive():
                continue
            thread.join(timeout=0.1)
            if result_queue.empty():
                _emit_event("error", error="模拟四六级考试评分线程已退出，但未返回结果。")
                raise SystemExit(1)
            continue

        while True:
            try:
                phase, message = progress_queue.get_nowait()
            except queue.Empty:
                break
            current_phase = phase
            phase_started_at = time.monotonic()
            _emit_event("progress", phase=phase, message=message)

        if status == "error":
            _emit_event("error", error=str(result_payload))
            raise SystemExit(1) from result_payload

        record = result_payload
        break

    _emit_event("result", ok=True, mock_exam=record.to_dict())


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

    submit_live = subparsers.add_parser("submit-live")
    submit_live.set_defaults(func=command_submit_live)

    history = subparsers.add_parser("history")
    history.add_argument("--limit", type=int, default=30)
    history.set_defaults(func=command_history)

    delete_history = subparsers.add_parser("delete-history")
    delete_history.add_argument("--attempt-id", required=True)
    delete_history.set_defaults(func=command_delete_history)

    review = subparsers.add_parser("review")
    review.add_argument("--attempt-id", required=True)
    review.set_defaults(func=command_review)

    mock_exam_history = subparsers.add_parser("mock-exam-history")
    mock_exam_history.add_argument("--limit", type=int, default=20)
    mock_exam_history.set_defaults(func=command_mock_exam_history)

    mock_exam_review = subparsers.add_parser("mock-exam-review")
    mock_exam_review.add_argument("--exam-id", required=True)
    mock_exam_review.set_defaults(func=command_mock_exam_review)

    mock_exam_delete = subparsers.add_parser("mock-exam-delete")
    mock_exam_delete.add_argument("--exam-id", required=True)
    mock_exam_delete.set_defaults(func=command_mock_exam_delete)

    mock_exam_weakness = subparsers.add_parser("mock-exam-weakness")
    mock_exam_weakness.add_argument("--limit", type=int, default=20)
    mock_exam_weakness.set_defaults(func=command_mock_exam_weakness)

    submit_mock_exam = subparsers.add_parser("submit-mock-exam")
    submit_mock_exam.set_defaults(func=command_submit_mock_exam)

    submit_mock_exam_live = subparsers.add_parser("submit-mock-exam-live")
    submit_mock_exam_live.set_defaults(func=command_submit_mock_exam_live)

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
