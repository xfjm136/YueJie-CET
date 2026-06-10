from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.constants import DEFAULT_MODEL


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ENV_FILE = PROJECT_ROOT / ".env"


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_dir: Path
    db_path: Path
    deepseek_api_key: str | None
    deepseek_base_url: str
    deepseek_model: str
    request_timeout: float


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    _load_env_file(ENV_FILE)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    db_path = os.getenv("YUEJIE_DB_PATH")
    resolved_db = Path(db_path).expanduser() if db_path else DATA_DIR / "yuejie.db"
    return Settings(
        project_root=PROJECT_ROOT,
        data_dir=DATA_DIR,
        db_path=resolved_db,
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY") or None,
        deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/"),
        deepseek_model=os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL),
        request_timeout=float(os.getenv("YUEJIE_REQUEST_TIMEOUT", "120")),
    )

