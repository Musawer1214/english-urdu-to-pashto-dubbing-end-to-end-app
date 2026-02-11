from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from .utils import LogFn


@dataclass(slots=True)
class GenderDetectionResult:
    gender: str  # male | female | unknown
    confidence: float
    median_f0_hz: float
    voiced_ratio: float
    reason: str


def detect_primary_speaker_gender(audio_wav: Path, log_fn: LogFn | None = None) -> GenderDetectionResult:
    def _log(msg: str) -> None:
        if log_fn:
            log_fn(msg)

    y, sr = librosa.load(str(audio_wav), sr=16000, mono=True)
    if y.size < sr:
        return GenderDetectionResult(
            gender="unknown",
            confidence=0.0,
            median_f0_hz=0.0,
            voiced_ratio=0.0,
            reason="Audio too short for reliable gender detection.",
        )

    frame_length = 1024
    hop_length = 256
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    energy_threshold = max(float(np.percentile(rms, 30)), 0.0035)
    voiced_mask = rms > energy_threshold
    voiced_ratio = float(np.mean(voiced_mask)) if voiced_mask.size else 0.0
    if voiced_ratio < 0.08:
        return GenderDetectionResult(
            gender="unknown",
            confidence=0.0,
            median_f0_hz=0.0,
            voiced_ratio=voiced_ratio,
            reason="Not enough voiced frames for pitch analysis.",
        )

    f0 = librosa.yin(
        y=y,
        fmin=70.0,
        fmax=350.0,
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    keep = voiced_mask[: f0.shape[0]]
    voiced_f0 = f0[keep]
    voiced_f0 = voiced_f0[np.isfinite(voiced_f0)]
    if voiced_f0.size < 30:
        return GenderDetectionResult(
            gender="unknown",
            confidence=0.0,
            median_f0_hz=0.0,
            voiced_ratio=voiced_ratio,
            reason="Too few reliable pitch samples.",
        )

    median_f0 = float(np.median(voiced_f0))
    if median_f0 <= 160.0:
        conf = min(0.95, 0.55 + (160.0 - median_f0) / 60.0)
        result = GenderDetectionResult(
            gender="male",
            confidence=conf,
            median_f0_hz=median_f0,
            voiced_ratio=voiced_ratio,
            reason=f"Median pitch {median_f0:.1f} Hz suggests male voice.",
        )
    elif median_f0 >= 185.0:
        conf = min(0.95, 0.55 + (median_f0 - 185.0) / 60.0)
        result = GenderDetectionResult(
            gender="female",
            confidence=conf,
            median_f0_hz=median_f0,
            voiced_ratio=voiced_ratio,
            reason=f"Median pitch {median_f0:.1f} Hz suggests female voice.",
        )
    else:
        result = GenderDetectionResult(
            gender="unknown",
            confidence=0.35,
            median_f0_hz=median_f0,
            voiced_ratio=voiced_ratio,
            reason=f"Median pitch {median_f0:.1f} Hz is ambiguous.",
        )

    _log(
        "Gender detection: "
        f"{result.gender} (confidence={result.confidence:.2f}, "
        f"median_f0={result.median_f0_hz:.1f}Hz, voiced_ratio={result.voiced_ratio:.2f})"
    )
    return result

