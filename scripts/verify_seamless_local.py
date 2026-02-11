from __future__ import annotations

from pathlib import Path


REQUIRED_FILES = [
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "added_tokens.json",
    "preprocessor_config.json",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer_config.json",
]


def main() -> int:
    model_dir = Path(__file__).resolve().parents[1] / "models" / "seamless-m4t-v2-large"
    print(f"Checking: {model_dir}")
    missing = [name for name in REQUIRED_FILES if not (model_dir / name).exists()]
    if missing:
        print("Missing files:")
        for m in missing:
            print(f" - {m}")
        return 1
    print("All required Seamless local files are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
