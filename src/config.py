from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import sys


APP_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = APP_ROOT / "src"
OUTPUTS_DIR = APP_ROOT / "outputs"
TEMP_DIR = APP_ROOT / "temp"
MODELS_DIR = APP_ROOT / "models"
CONFIGS_DIR = APP_ROOT / "configs"
LOGS_DIR = APP_ROOT / "logs"
TOOLS_DIR = APP_ROOT / "tools"
FFMPEG_BIN_DIR = TOOLS_DIR / "ffmpeg" / "bin"
HF_CACHE_DIR = MODELS_DIR / "hf_cache"
SEAMLESS_LOCAL_DIR = MODELS_DIR / "seamless-m4t-v2-large"
WAV2LIP_MODELS_DIR = MODELS_DIR / "wav2lip"
WAV2LIP_REPO_DIR = APP_ROOT / "external" / "Wav2Lip"
ESPEAK_EXE_DEFAULT = Path(r"C:\Program Files\eSpeak NG\espeak-ng.exe")
DEFAULT_TARGET_LANG = "pbt"
DEFAULT_MODEL_NAME = "facebook/seamless-m4t-v2-large"


def ensure_layout() -> None:
    for path in (
        OUTPUTS_DIR,
        TEMP_DIR,
        MODELS_DIR,
        CONFIGS_DIR,
        LOGS_DIR,
        HF_CACHE_DIR,
        SEAMLESS_LOCAL_DIR,
        WAV2LIP_MODELS_DIR,
        FFMPEG_BIN_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def get_ffmpeg_exe() -> str:
    exe = FFMPEG_BIN_DIR / "ffmpeg.exe"
    return str(exe if exe.exists() else "ffmpeg")


def get_ffprobe_exe() -> str:
    exe = FFMPEG_BIN_DIR / "ffprobe.exe"
    return str(exe if exe.exists() else "ffprobe")


def get_base_env() -> dict[str, str]:
    env = os.environ.copy()
    env["HF_HOME"] = str(HF_CACHE_DIR)
    env["TRANSFORMERS_CACHE"] = str(HF_CACHE_DIR)
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    env["PATH"] = f"{FFMPEG_BIN_DIR}{os.pathsep}{env.get('PATH', '')}"
    return env


def get_espeak_exe() -> str:
    return str(ESPEAK_EXE_DEFAULT if ESPEAK_EXE_DEFAULT.exists() else "espeak-ng")


def get_python_exe() -> str:
    return sys.executable


@dataclass(slots=True)
class PipelineConfig:
    model_name: str = DEFAULT_MODEL_NAME
    model_local_dir: Path = SEAMLESS_LOCAL_DIR
    source_lang: str = "auto"  # auto | eng | urd
    target_lang: str = DEFAULT_TARGET_LANG
    device_policy: str = "auto"  # auto | cpu | cuda
    chunk_seconds: int = 20
    text_beams: int = 3
    enable_translation_verification: bool = True
    verification_margin: float = 0.03
    min_roundtrip_score: float = 0.55
    tts_backend: str = "edge_tts"  # edge_tts | espeak
    tts_voice: str = "ps-AF-LatifaNeural"
    tts_gender_mode: str = "auto"  # auto | male | female
    term_overrides_path: Path = CONFIGS_DIR / "term_overrides.json"
