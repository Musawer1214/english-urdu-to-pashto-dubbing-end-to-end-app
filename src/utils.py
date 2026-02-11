from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import shutil
import subprocess
from typing import Callable, Iterable


LogFn = Callable[[str], None]
ProgressFn = Callable[[float, str], None]


def noop_log(_: str) -> None:
    return None


def noop_progress(_: float, __: str) -> None:
    return None


def run_command(
    command: Iterable[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    log_fn: LogFn | None = None,
) -> None:
    logger = log_fn or noop_log
    process = subprocess.Popen(
        list(command),
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip()
        if line:
            logger(line)
    return_code = process.wait()
    if return_code != 0:
        cmd_text = " ".join(command)
        raise RuntimeError(f"Command failed ({return_code}): {cmd_text}")


def format_srt_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    td = timedelta(seconds=seconds)
    base = datetime(1, 1, 1) + td
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{base.hour:02}:{base.minute:02}:{base.second:02},{millis:03}"


def write_srt(segments: list[tuple[float, float, str]], output_path: Path) -> None:
    lines: list[str] = []
    for idx, (start_s, end_s, text) in enumerate(segments, start=1):
        lines.append(str(idx))
        lines.append(f"{format_srt_timestamp(start_s)} --> {format_srt_timestamp(end_s)}")
        lines.append((text or "").strip() or "[No text]")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def clean_and_mkdir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)

