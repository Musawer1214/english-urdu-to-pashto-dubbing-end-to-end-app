from __future__ import annotations

import argparse
from pathlib import Path
import sys

from src.config import OUTPUTS_DIR, PipelineConfig, SEAMLESS_LOCAL_DIR
from src.pipeline import VideoDubPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Pashto dubbing + lip-sync pipeline on one video.")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output-root", default=str(OUTPUTS_DIR), help="Directory where result folder is created")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--source-lang", choices=["auto", "eng", "urd"], default="auto")
    parser.add_argument("--target-lang", default="pbt", help="Target language code supported by SeamlessM4T")
    parser.add_argument("--model", default="facebook/seamless-m4t-v2-large")
    parser.add_argument("--model-local-dir", default=str(SEAMLESS_LOCAL_DIR))
    parser.add_argument("--chunk-seconds", type=int, default=20)
    parser.add_argument("--voice-gender", choices=["auto", "male", "female"], default="auto")
    parser.add_argument("--no-verify", action="store_true", help="Disable translation round-trip verification")
    args = parser.parse_args()

    cfg = PipelineConfig(
        model_name=args.model,
        model_local_dir=Path(args.model_local_dir),
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        device_policy=args.device,
        chunk_seconds=args.chunk_seconds,
        tts_gender_mode=args.voice_gender,
        enable_translation_verification=not args.no_verify,
    )

    def log_fn(msg: str) -> None:
        print(msg, flush=True)

    def progress_fn(p: float, msg: str) -> None:
        print(f"[{p * 100:6.2f}%] {msg}", flush=True)

    pipeline = VideoDubPipeline(cfg, log_fn=log_fn)
    result = pipeline.run(
        input_video=Path(args.input),
        output_root=Path(args.output_root),
        progress_fn=progress_fn,
    )
    print("\nPipeline finished successfully.")
    print(f"Final video: {result.final_video}")
    print(f"SRT file:    {result.translated_srt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
