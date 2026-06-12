from __future__ import annotations

import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CARGO_MANIFEST = PROJECT_ROOT / "Cargo.toml"


def _find_rust_frontend() -> Path | None:
    candidates = [
        PROJECT_ROOT / "target" / "release" / "yuejie-cet-rs",
        PROJECT_ROOT / "target" / "debug" / "yuejie-cet-rs",
    ]
    available = [
        candidate
        for candidate in candidates
        if candidate.exists() and os.access(candidate, os.X_OK)
    ]
    if not available:
        return None
    return max(available, key=lambda path: path.stat().st_mtime)


def main() -> None:
    rust_binary = _find_rust_frontend()
    if rust_binary is not None:
        raise SystemExit(subprocess.run([str(rust_binary)], check=False).returncode)
    if CARGO_MANIFEST.exists():
        raise SystemExit(
            subprocess.run(
                ["cargo", "run", "--release"],
                cwd=PROJECT_ROOT,
                check=False,
            ).returncode
        )
    raise SystemExit(
        "未找到 Rust 前端可执行文件，且当前目录不是可编译的 Cargo 项目。请先在仓库根目录执行 `cargo build --release`。"
    )


if __name__ == "__main__":
    main()
