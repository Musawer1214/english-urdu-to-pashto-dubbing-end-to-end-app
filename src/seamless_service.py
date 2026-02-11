from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
from typing import Callable

import librosa
import numpy as np
import torch
from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText, SeamlessM4Tv2ForTextToText

from .config import HF_CACHE_DIR, PipelineConfig
from .utils import ProgressFn, write_srt


@dataclass(slots=True)
class TextSegment:
    start_s: float
    end_s: float
    text: str
    source_text: str = ""
    verification_score: float = 0.0
    strategy: str = "s2tt"


@dataclass(slots=True)
class TranslationArtifacts:
    translated_srt: Path
    translated_text: Path
    chunks_count: int
    source_duration_s: float
    segments: list[TextSegment]


class SeamlessTranslator:
    def __init__(self, cfg: PipelineConfig, log_fn: Callable[[str], None]) -> None:
        self.cfg = cfg
        self.log_fn = log_fn
        self.speech_model: SeamlessM4Tv2ForSpeechToText | None = None
        self.text_model: SeamlessM4Tv2ForTextToText | None = None
        self.processor = None
        self.device = self._resolve_device(cfg.device_policy)
        self.term_replacements = self._load_term_replacements()

    @staticmethod
    def _resolve_device(policy: str) -> str:
        if policy == "cpu":
            return "cpu"
        if policy == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA selected, but CUDA is not available on this machine.")
            return "cuda"
        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _normalize_text(value: str) -> str:
        txt = (value or "").strip().lower()
        txt = re.sub(r"[^\w\s\u0600-\u06FF]", " ", txt)
        txt = re.sub(r"\s+", " ", txt).strip()
        return txt

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        na = SeamlessTranslator._normalize_text(a)
        nb = SeamlessTranslator._normalize_text(b)
        if not na or not nb:
            return 0.0
        return float(SequenceMatcher(None, na, nb).ratio())

    @staticmethod
    def _english_likeness(text: str) -> float:
        if not text:
            return 0.0
        chars = [c for c in text if c.isalpha()]
        if not chars:
            return 0.0
        latin = sum(1 for c in chars if ("a" <= c.lower() <= "z"))
        words = max(1, len(text.split()))
        return (latin / len(chars)) * (1.0 + min(words, 12) / 24.0)

    @staticmethod
    def _urdu_likeness(text: str) -> float:
        if not text:
            return 0.0
        chars = [c for c in text if c.isalpha()]
        if not chars:
            return 0.0
        arabic = sum(1 for c in chars if "\u0600" <= c <= "\u06FF")
        words = max(1, len(text.split()))
        return (arabic / len(chars)) * (1.0 + min(words, 12) / 24.0)

    @staticmethod
    def _latin_ratio(text: str) -> float:
        chars = [c for c in text if c.isalpha()]
        if not chars:
            return 0.0
        latin = sum(1 for c in chars if ("a" <= c.lower() <= "z"))
        return latin / len(chars)

    @staticmethod
    def _arabic_ratio(text: str) -> float:
        chars = [c for c in text if c.isalpha()]
        if not chars:
            return 0.0
        arabic = sum(1 for c in chars if "\u0600" <= c <= "\u06FF")
        return arabic / len(chars)

    def _resolve_model_source(self) -> str:
        local_dir = Path(self.cfg.model_local_dir)
        required = [
            "config.json",
            "generation_config.json",
            "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "tokenizer_config.json",
            "sentencepiece.bpe.model",
            "special_tokens_map.json",
            "preprocessor_config.json",
            "added_tokens.json",
        ]
        if all((local_dir / f).exists() for f in required):
            self.log_fn(f"Using local Seamless model folder: {local_dir}")
            return str(local_dir)
        self.log_fn(f"Using remote model id: {self.cfg.model_name}")
        return self.cfg.model_name

    def _load_speech_once(self) -> None:
        if self.speech_model is not None and self.processor is not None:
            return
        model_source = self._resolve_model_source()
        self.log_fn(f"Loading speech translation model: {model_source}")
        HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_source, cache_dir=str(HF_CACHE_DIR))
        self.speech_model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
            model_source,
            cache_dir=str(HF_CACHE_DIR),
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.speech_model.eval()
        self.log_fn(f"Speech model loaded on {self.device.upper()}")

    def _load_text_once(self) -> None:
        if self.text_model is not None:
            return
        model_source = self._resolve_model_source()
        self.log_fn(f"Loading text translation model: {model_source}")
        HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.text_model = SeamlessM4Tv2ForTextToText.from_pretrained(
            model_source,
            cache_dir=str(HF_CACHE_DIR),
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.text_model.eval()
        self.log_fn(f"Text model loaded on {self.device.upper()}")

    def _validate_target_language(self, target_lang: str) -> str:
        assert self.speech_model is not None
        lang_map = getattr(self.speech_model.generation_config, "text_decoder_lang_to_code_id", {}) or {}
        if target_lang in lang_map:
            return target_lang
        if "pbt" in lang_map:
            self.log_fn(f"Requested target language '{target_lang}' not available; using 'pbt'.")
            return "pbt"
        options = ", ".join(sorted(lang_map.keys())[:40])
        raise RuntimeError(f"Target language '{target_lang}' not available. Available sample: {options}")

    def _cleanup_device(self) -> None:
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def _speech_to_text(self, chunk_audio: np.ndarray, tgt_lang: str, max_new_tokens: int = 240) -> str:
        assert self.speech_model is not None and self.processor is not None
        model_inputs = self.processor(audios=chunk_audio, sampling_rate=16000, return_tensors="pt")
        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}
        with torch.inference_mode():
            outputs = self.speech_model.generate(
                **model_inputs,
                tgt_lang=tgt_lang,
                num_beams=self.cfg.text_beams,
                max_new_tokens=max_new_tokens,
            )
        text = self.processor.decode(outputs[0].tolist(), skip_special_tokens=True).strip()
        del model_inputs, outputs
        self._cleanup_device()
        return text

    def _text_to_text(self, text: str, src_lang: str, tgt_lang: str, max_new_tokens: int = 220) -> str:
        assert self.text_model is not None and self.processor is not None
        clean_text = (text or "").strip()
        if not clean_text:
            return ""
        model_inputs = self.processor(text=clean_text, src_lang=src_lang, return_tensors="pt")
        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}
        with torch.inference_mode():
            outputs = self.text_model.generate(
                **model_inputs,
                tgt_lang=tgt_lang,
                num_beams=self.cfg.text_beams,
                max_new_tokens=max_new_tokens,
            )
        out_text = self.processor.decode(outputs[0].tolist(), skip_special_tokens=True).strip()
        del model_inputs, outputs
        self._cleanup_device()
        return out_text

    def _detect_source_language(self, samples: np.ndarray, chunk_size: int) -> str:
        preferred = (self.cfg.source_lang or "auto").strip().lower()
        if preferred in {"eng", "urd"}:
            self.log_fn(f"Using user-selected source language: {preferred}")
            return preferred

        probe = samples[: min(samples.shape[0], chunk_size)]
        if probe.shape[0] < 1200:
            self.log_fn("Source language auto-detect fallback: audio probe too short, using 'eng'.")
            return "eng"

        eng_text = self._speech_to_text(probe, "eng")
        urd_text = self._speech_to_text(probe, "urd")
        eng_score = self._english_likeness(eng_text)
        urd_score = self._urdu_likeness(urd_text)
        eng_latin = self._latin_ratio(eng_text)
        urd_latin = self._latin_ratio(urd_text)
        eng_arabic = self._arabic_ratio(eng_text)
        urd_arabic = self._arabic_ratio(urd_text)
        if urd_arabic >= 0.25 and urd_arabic > eng_arabic + 0.08:
            lang = "urd"
        elif eng_latin >= 0.45 and eng_latin > urd_latin + 0.08:
            lang = "eng"
        else:
            lang = "urd" if urd_score > eng_score else "eng"
        self.log_fn(
            "Source language auto-detect: "
            f"eng_score={eng_score:.3f}, urd_score={urd_score:.3f}, "
            f"eng_latin={eng_latin:.3f}, urd_latin={urd_latin:.3f}, "
            f"eng_arabic={eng_arabic:.3f}, urd_arabic={urd_arabic:.3f}, selected={lang}"
        )
        return lang

    def _load_term_replacements(self) -> dict[str, str]:
        path = Path(self.cfg.term_overrides_path)
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            replacements = data.get("pashto_replace", {}) if isinstance(data, dict) else {}
            clean: dict[str, str] = {}
            for k, v in replacements.items():
                k2 = (str(k)).strip()
                v2 = (str(v)).strip()
                if k2 and v2:
                    clean[k2] = v2
            if clean:
                self.log_fn(f"Loaded {len(clean)} terminology replacements from {path}")
            return clean
        except Exception as exc:
            self.log_fn(f"Could not load term override file {path}: {exc}")
            return {}

    def _apply_term_replacements(self, text: str) -> str:
        fixed = text
        for wrong, correct in self.term_replacements.items():
            fixed = fixed.replace(wrong, correct)
        return fixed

    def translate_audio(
        self,
        source_wav: Path,
        output_srt: Path,
        output_text: Path,
        progress_fn: ProgressFn,
    ) -> TranslationArtifacts:
        self._load_speech_once()
        assert self.speech_model is not None and self.processor is not None

        target_lang = self._validate_target_language(self.cfg.target_lang)
        verify_enabled = bool(self.cfg.enable_translation_verification)
        if verify_enabled:
            try:
                self._load_text_once()
            except Exception as exc:
                verify_enabled = False
                self.log_fn(f"Verification disabled (text model failed to load): {exc}")

        samples, _ = librosa.load(str(source_wav), sr=16000, mono=True)
        if len(samples) == 0:
            raise RuntimeError("No audio samples were extracted from source video.")

        total_duration = float(len(samples)) / 16000.0
        chunk_size = int(self.cfg.chunk_seconds * 16000)
        chunk_size = max(chunk_size, 4 * 16000)
        source_lang = self._detect_source_language(samples, chunk_size)

        segments: list[TextSegment] = []
        translated_lines: list[str] = []

        starts = list(range(0, len(samples), chunk_size))
        chunk_count = len(starts)
        for idx, start in enumerate(starts, start=1):
            end = min(start + chunk_size, len(samples))
            source_chunk = samples[start:end].astype("float32", copy=False)
            if source_chunk.shape[0] < 400:
                source_chunk = np.pad(source_chunk, (0, 400 - source_chunk.shape[0]), mode="constant")
            start_s = start / 16000.0
            end_s = end / 16000.0

            progress_fn((idx - 1) / max(1, chunk_count), f"Translating chunk {idx}/{chunk_count}")

            source_text = self._speech_to_text(source_chunk, source_lang)
            direct_pashto = self._speech_to_text(source_chunk, target_lang)

            chosen = direct_pashto
            score = 1.0
            strategy = "s2tt"

            if verify_enabled and self.text_model is not None:
                try:
                    t2t_pashto = self._text_to_text(source_text, source_lang, target_lang)
                    back_direct = self._text_to_text(direct_pashto, target_lang, source_lang)
                    back_t2t = self._text_to_text(t2t_pashto, target_lang, source_lang)

                    score_direct = self._similarity(source_text, back_direct)
                    score_t2t = self._similarity(source_text, back_t2t)

                    if score_t2t > score_direct + float(self.cfg.verification_margin):
                        chosen = t2t_pashto
                        score = score_t2t
                        strategy = "s2tt+verify->t2tt"
                    else:
                        score = score_direct
                        strategy = "s2tt+verify->s2tt"

                    self.log_fn(
                        f"Chunk {idx}: verify score_direct={score_direct:.3f}, "
                        f"score_t2t={score_t2t:.3f}, strategy={strategy}"
                    )
                    if score < float(self.cfg.min_roundtrip_score):
                        self.log_fn(
                            f"Chunk {idx}: low translation confidence ({score:.3f}) after verification."
                        )
                except Exception as exc:
                    self.log_fn(f"Chunk {idx}: verification skipped due to error: {exc}")

            chosen = self._apply_term_replacements(chosen.strip())
            if not chosen:
                chosen = "[Untranslated chunk]"

            segments.append(
                TextSegment(
                    start_s=start_s,
                    end_s=end_s,
                    text=chosen,
                    source_text=source_text,
                    verification_score=score,
                    strategy=strategy,
                )
            )
            translated_lines.append(chosen)

        progress_fn(0.98, "Finalizing translated subtitles")
        srt_segments = [(s.start_s, s.end_s, s.text) for s in segments]
        write_srt(srt_segments, output_srt)
        output_text.write_text("\n".join(translated_lines), encoding="utf-8")
        progress_fn(1.0, "Translation complete")

        return TranslationArtifacts(
            translated_srt=output_srt,
            translated_text=output_text,
            chunks_count=chunk_count,
            source_duration_s=total_duration,
            segments=segments,
        )
