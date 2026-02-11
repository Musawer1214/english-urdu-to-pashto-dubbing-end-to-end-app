from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import librosa
import numpy as np


@dataclass(slots=True)
class LipSyncGateDecision:
    should_lipsync: bool
    reason: str
    speech_ratio: float
    face_ratio: float
    dominant_box: tuple[int, int, int, int] | None = None  # y1, y2, x1, x2


def _median_box(boxes: Iterable[tuple[int, int, int, int]]) -> tuple[int, int, int, int] | None:
    box_list = list(boxes)
    if not box_list:
        return None
    arr = np.array(box_list, dtype=np.float32)
    med = np.median(arr, axis=0).astype(int).tolist()
    return med[0], med[1], med[2], med[3]


def _detect_faces(video_path: Path, sample_every_n: int = 2) -> tuple[list[tuple[int, int, int, int]], float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for face analysis: {video_path}")

    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if detector.empty():
        raise RuntimeError("OpenCV haarcascade frontal face detector is unavailable.")

    sampled = 0
    with_face = 0
    boxes: list[tuple[int, int, int, int]] = []
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % sample_every_n != 0:
                frame_idx += 1
                continue
            frame_idx += 1
            sampled += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
            if len(faces) == 0:
                continue

            # Use the largest face per sampled frame as the likely main speaker candidate.
            x, y, w, h = max(faces, key=lambda f: int(f[2]) * int(f[3]))
            with_face += 1
            boxes.append((int(y), int(y + h), int(x), int(x + w)))
    finally:
        cap.release()

    ratio = (with_face / sampled) if sampled > 0 else 0.0
    return boxes, ratio


def _estimate_speech_ratio(audio_wav: Path) -> float:
    samples, _ = librosa.load(str(audio_wav), sr=16000, mono=True)
    if samples.size == 0:
        return 0.0

    # 25ms window, 10ms hop for coarse speech/no-speech energy estimate.
    rms = librosa.feature.rms(y=samples, frame_length=400, hop_length=160, center=True)[0]
    if rms.size == 0:
        return 0.0
    threshold = max(0.008, float(np.percentile(rms, 65) * 0.6))
    speech_ratio = float(np.mean(rms > threshold))
    return speech_ratio


def evaluate_lipsync_gate(video_path: Path, source_audio_wav: Path) -> LipSyncGateDecision:
    speech_ratio = _estimate_speech_ratio(source_audio_wav)
    face_boxes, face_ratio = _detect_faces(video_path)
    dom_box = _median_box(face_boxes)

    if face_ratio < 0.95:
        return LipSyncGateDecision(
            should_lipsync=False,
            reason=f"Face not continuously visible enough for lip-sync safety (face_ratio={face_ratio:.2f}).",
            speech_ratio=speech_ratio,
            face_ratio=face_ratio,
            dominant_box=dom_box,
        )

    if speech_ratio < 0.10:
        return LipSyncGateDecision(
            should_lipsync=False,
            reason=f"Speech activity too low (speech_ratio={speech_ratio:.2f}).",
            speech_ratio=speech_ratio,
            face_ratio=face_ratio,
            dominant_box=dom_box,
        )

    return LipSyncGateDecision(
        should_lipsync=True,
        reason="Face and speech checks passed.",
        speech_ratio=speech_ratio,
        face_ratio=face_ratio,
        dominant_box=dom_box,
    )
