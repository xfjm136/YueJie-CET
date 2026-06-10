from __future__ import annotations

import os
import subprocess
from pathlib import Path

from app.ui.app import YueJieApp


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _find_rust_frontend() -> Path | None:
    candidates = [
        PROJECT_ROOT / "target" / "release" / "yuejie-cet-rs",
        PROJECT_ROOT / "target" / "debug" / "yuejie-cet-rs",
    ]
    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate
    return None


def main() -> None:
    rust_binary = _find_rust_frontend()
    if rust_binary is not None:
        raise SystemExit(subprocess.run([str(rust_binary)], check=False).returncode)
    YueJieApp().run()


if __name__ == "__main__":
    main()
