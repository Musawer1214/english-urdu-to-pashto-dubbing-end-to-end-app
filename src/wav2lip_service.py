from __future__ import annotations

from pathlib import Path
import shutil

from .config import WAV2LIP_MODELS_DIR, WAV2LIP_REPO_DIR, get_base_env, get_python_exe
from .utils import LogFn, run_command


class Wav2LipRunner:
    def __init__(self, log_fn: LogFn | None = None) -> None:
        self.log_fn = log_fn

    def _log(self, message: str) -> None:
        if self.log_fn:
            self.log_fn(message)

    def ensure_assets(self) -> Path:
        if not WAV2LIP_REPO_DIR.exists():
            raise RuntimeError(
                f"Wav2Lip repository not found at {WAV2LIP_REPO_DIR}. "
                "Clone/download it before running lip sync."
            )

        checkpoint = WAV2LIP_MODELS_DIR / "checkpoints" / "wav2lip_gan.pth"
        face_detector = WAV2LIP_MODELS_DIR / "face_detection" / "detection" / "sfd" / "s3fd.pth"
        if not checkpoint.exists():
            raise RuntimeError(f"Missing Wav2Lip checkpoint: {checkpoint}")
        if not face_detector.exists():
            raise RuntimeError(f"Missing face detection checkpoint: {face_detector}")

        repo_face_path = WAV2LIP_REPO_DIR / "face_detection" / "detection" / "sfd" / "s3fd.pth"
        repo_face_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(face_detector, repo_face_path)
        return checkpoint

    def run_lip_sync(
        self,
        *,
        input_video: Path,
        input_audio: Path,
        output_video: Path,
        device_policy: str,
        dominant_box: tuple[int, int, int, int] | None = None,
    ) -> None:
        checkpoint = self.ensure_assets()
        output_video.parent.mkdir(parents=True, exist_ok=True)
        env = get_base_env()

        if device_policy == "cpu":
            env["WAV2LIP_DEVICE"] = "cpu"
            wav2lip_batch = "32"
            face_det_batch = "4"
        else:
            env["WAV2LIP_DEVICE"] = "auto"
            wav2lip_batch = "64"
            face_det_batch = "8"

        attempts: list[list[str]] = []
        if dominant_box is not None:
            y1, y2, x1, x2 = dominant_box
            attempts.append(["--box", str(y1), str(y2), str(x1), str(x2), "--resize_factor", "1"])
        attempts.extend(
            [
                ["--pads", "0", "20", "0", "0", "--resize_factor", "1"],
                ["--pads", "0", "40", "0", "0", "--resize_factor", "1", "--nosmooth"],
                ["--pads", "0", "40", "0", "0", "--resize_factor", "2", "--nosmooth"],
            ]
        )
        last_exc: Exception | None = None
        for i, extra in enumerate(attempts, start=1):
            cmd = [
                get_python_exe(),
                "inference.py",
                "--checkpoint_path",
                str(checkpoint),
                "--face",
                str(input_video),
                "--audio",
                str(input_audio),
                "--outfile",
                str(output_video),
                "--wav2lip_batch_size",
                wav2lip_batch,
                "--face_det_batch_size",
                face_det_batch,
                *extra,
            ]
            self._log(f"Running Wav2Lip inference (attempt {i}/{len(attempts)})...")
            try:
                run_command(cmd, cwd=WAV2LIP_REPO_DIR, env=env, log_fn=self.log_fn)
                self._log("Wav2Lip inference complete.")
                return
            except Exception as exc:
                last_exc = exc
                self._log(f"Wav2Lip attempt {i} failed: {exc}")

        raise RuntimeError(f"Wav2Lip failed after retries: {last_exc}")
