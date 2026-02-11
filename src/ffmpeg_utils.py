from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

from .config import get_base_env, get_ffmpeg_exe, get_ffprobe_exe
from .utils import LogFn, run_command


def extract_mono_wav(video_path: Path, out_wav: Path, sample_rate: int = 16000, log_fn: LogFn | None = None) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        get_ffmpeg_exe(),
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-acodec",
        "pcm_s16le",
        str(out_wav),
    ]
    run_command(cmd, env=get_base_env(), log_fn=log_fn)


def probe_duration_seconds(media_path: Path) -> float:
    cmd = [
        get_ffprobe_exe(),
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(media_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, env=get_base_env(), check=True)
    data = json.loads(result.stdout)
    duration = float(data["format"]["duration"])
    return duration


def _build_atempo_filter(tempo: float) -> str:
    if tempo <= 0:
        raise ValueError("tempo must be > 0")
    factors: list[float] = []
    remaining = tempo
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    factors.append(remaining)
    return ",".join(f"atempo={f:.6f}" for f in factors)


def time_stretch_audio_to_target(
    input_wav: Path,
    output_wav: Path,
    target_duration_s: float,
    *,
    tolerance_s: float = 0.35,
    log_fn: LogFn | None = None,
) -> tuple[float, float]:
    src_duration = probe_duration_seconds(input_wav)
    if target_duration_s <= 0:
        shutil.copy2(input_wav, output_wav)
        return src_duration, src_duration

    if abs(src_duration - target_duration_s) <= tolerance_s:
        shutil.copy2(input_wav, output_wav)
        return src_duration, src_duration

    tempo = src_duration / target_duration_s
    atempo = _build_atempo_filter(tempo)
    cmd = [
        get_ffmpeg_exe(),
        "-y",
        "-i",
        str(input_wav),
        "-filter:a",
        atempo,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-acodec",
        "pcm_s16le",
        str(output_wav),
    ]
    run_command(cmd, env=get_base_env(), log_fn=log_fn)
    out_duration = probe_duration_seconds(output_wav)
    return src_duration, out_duration


def mux_audio_with_video(
    input_video: Path,
    input_audio_wav: Path,
    output_video: Path,
    log_fn: LogFn | None = None,
) -> None:
    output_video.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        get_ffmpeg_exe(),
        "-y",
        "-i",
        str(input_video),
        "-i",
        str(input_audio_wav),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(output_video),
    ]
    run_command(cmd, env=get_base_env(), log_fn=log_fn)
