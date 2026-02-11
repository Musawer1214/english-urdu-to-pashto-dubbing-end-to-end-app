from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download

APP_ROOT = Path(__file__).resolve().parents[1]
W2L_LOCAL = APP_ROOT / "models" / "wav2lip"


def main() -> int:
    W2L_LOCAL.mkdir(parents=True, exist_ok=True)
    files = [
        "checkpoints/wav2lip_gan.pth",
        "face_detection/detection/sfd/s3fd.pth",
    ]
    for filename in files:
        path = hf_hub_download(repo_id="camenduru/Wav2Lip", filename=filename, local_dir=W2L_LOCAL)
        print(f"Downloaded: {path}")
    print("Wav2Lip models ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

