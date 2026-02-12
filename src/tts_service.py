from __future__ import annotations

import os
from pathlib import Path
import socket
import tempfile
import time
import threading

import librosa
import numpy as np

from .config import PipelineConfig, get_base_env, get_espeak_exe, get_ffmpeg_exe, get_python_exe
from .ffmpeg_utils import probe_duration_seconds, time_stretch_audio_to_target
from .seamless_service import TextSegment
from .utils import LogFn, PipelineCancelledError, run_command


class PashtoTTSService:
    def __init__(self, cfg: PipelineConfig, log_fn: LogFn | None = None) -> None:
        self.cfg = cfg
        self.log_fn = log_fn
        self.source_gender_hint = "unknown"

    def _log(self, msg: str) -> None:
        if self.log_fn:
            self.log_fn(msg)

    @staticmethod
    def _ensure_not_cancelled(cancel_event: threading.Event | None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise PipelineCancelledError("Operation cancelled by user.")

    @staticmethod
    def _is_near_silent(wav_path: Path) -> bool:
        samples, _ = librosa.load(str(wav_path), sr=16000, mono=True)
        if samples.size == 0:
            return True
        peak = float(np.max(np.abs(samples)))
        rms = float(np.sqrt(np.mean(samples * samples)))
        return peak < 0.003 or rms < 0.0008

    def is_near_silent(self, wav_path: Path) -> bool:
        return self._is_near_silent(wav_path)

    def set_source_gender_hint(self, gender: str) -> None:
        normalized = (gender or "").strip().lower()
        if normalized not in {"male", "female", "unknown"}:
            normalized = "unknown"
        self.source_gender_hint = normalized

    @staticmethod
    def _is_local_port_open(port: int, host: str = "127.0.0.1", timeout_s: float = 0.35) -> bool:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.settimeout(timeout_s)
            return sock.connect_ex((host, port)) == 0
        finally:
            sock.close()

    def _edge_proxy_candidates(self) -> list[str | None]:
        candidates: list[str | None] = [None]
        for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
            value = (os.environ.get(key) or "").strip()
            if value and value not in candidates:
                candidates.append(value)
        for port in (17890, 7890):
            if self._is_local_port_open(port):
                p = f"http://127.0.0.1:{port}"
                if p not in candidates:
                    candidates.append(p)
        return candidates

    def _voice_candidates(self) -> list[str]:
        female_voice = "ps-AF-LatifaNeural"
        male_voice = "ps-AF-GulNawazNeural"
        configured = (self.cfg.tts_voice or "").strip() or female_voice
        mode = (self.cfg.tts_gender_mode or "auto").strip().lower()

        if mode == "male":
            order = [male_voice, configured, female_voice]
        elif mode == "female":
            order = [female_voice, configured, male_voice]
        else:
            if self.source_gender_hint == "male":
                order = [male_voice, configured, female_voice]
            elif self.source_gender_hint == "female":
                order = [female_voice, configured, male_voice]
            else:
                order = [configured, female_voice, male_voice]

        dedup: list[str] = []
        for voice in order:
            if voice and voice not in dedup:
                dedup.append(voice)
        return dedup

    def _synthesize_edge_tts_to_wav(
        self,
        text: str,
        out_wav: Path,
        cancel_event: threading.Event | None = None,
    ) -> None:
        clean_text = " ".join((text or "").strip().split())
        if not clean_text:
            raise RuntimeError("Empty text passed to TTS.")

        voices = self._voice_candidates()
        proxies = self._edge_proxy_candidates()
        attempts: list[tuple[str, str | None, int]] = []
        for round_idx in range(4):
            for proxy in proxies:
                for voice in voices:
                    attempts.append((voice, proxy, round_idx))

        last_exc: Exception | None = None
        with tempfile.TemporaryDirectory() as td:
            text_file = Path(td) / "edge_input.txt"
            text_file.write_text(clean_text, encoding="utf-8")
            for idx, (voice, proxy, round_idx) in enumerate(attempts, start=1):
                self._ensure_not_cancelled(cancel_event)
                try:
                    mp3_path = Path(td) / f"tts_{idx}.mp3"
                    if mp3_path.exists():
                        mp3_path.unlink()
                    if out_wav.exists():
                        out_wav.unlink()
                    self._log(f"TTS synthesis attempt {idx}: voice={voice}, proxy={proxy or 'none'}")
                    edge_cmd = [
                        get_python_exe(),
                        "-m",
                        "edge_tts",
                        "--voice",
                        voice,
                        "--file",
                        str(text_file),
                        "--write-media",
                        str(mp3_path),
                    ]
                    if proxy:
                        edge_cmd.extend(["--proxy", proxy])
                    run_command(edge_cmd, env=get_base_env(), log_fn=self.log_fn, cancel_event=cancel_event)
                    if not mp3_path.exists() or mp3_path.stat().st_size < 2048:
                        raise RuntimeError("edge-tts output mp3 missing or too small.")
                    cmd = [
                        get_ffmpeg_exe(),
                        "-y",
                        "-i",
                        str(mp3_path),
                        "-ac",
                        "1",
                        "-ar",
                        "16000",
                        "-acodec",
                        "pcm_s16le",
                        str(out_wav),
                    ]
                    run_command(cmd, env=get_base_env(), log_fn=self.log_fn, cancel_event=cancel_event)
                    duration_s = probe_duration_seconds(out_wav)
                    if duration_s < 0.18:
                        raise RuntimeError(f"Generated edge-tts audio was too short ({duration_s:.2f}s).")
                    if self._is_near_silent(out_wav):
                        raise RuntimeError("Generated edge-tts audio was near-silent.")
                    return
                except Exception as exc:
                    last_exc = exc
                    self._log(
                        "TTS retry "
                        f"{idx}/{len(attempts)} failed (voice={voice}, round={round_idx + 1}, proxy={proxy or 'none'}): {exc}"
                    )
                    time.sleep(1.0 + 0.7 * round_idx)
        raise RuntimeError(f"Edge TTS failed after retries: {last_exc}")

    def _synthesize_espeak_to_wav(
        self,
        text: str,
        out_wav: Path,
        cancel_event: threading.Event | None = None,
    ) -> None:
        clean_text = " ".join((text or "").strip().split())
        if not clean_text:
            raise RuntimeError("Empty text passed to TTS.")
        cmd = [
            get_espeak_exe(),
            "-v",
            self.cfg.tts_voice,
            "-w",
            str(out_wav),
            clean_text,
        ]
        run_command(cmd, env=get_base_env(), log_fn=self.log_fn, cancel_event=cancel_event)
        if self._is_near_silent(out_wav):
            raise RuntimeError("Generated eSpeak audio was near-silent.")

    def _synthesize_to_wav(
        self,
        text: str,
        out_wav: Path,
        cancel_event: threading.Event | None = None,
    ) -> None:
        backend = self.cfg.tts_backend.lower().strip()
        if backend == "edge_tts":
            self._synthesize_edge_tts_to_wav(text, out_wav, cancel_event=cancel_event)
            return
        self._synthesize_espeak_to_wav(text, out_wav, cancel_event=cancel_event)

    def synthesize_segments(
        self,
        segments: list[TextSegment],
        out_wav: Path,
        temp_dir: Path,
        cancel_event: threading.Event | None = None,
    ) -> None:
        temp_dir.mkdir(parents=True, exist_ok=True)
        chunk_files: list[Path] = []

        for idx, seg in enumerate(segments, start=1):
            self._ensure_not_cancelled(cancel_event)
            self._log(f"TTS chunk {idx}/{len(segments)}")
            raw_wav = temp_dir / f"tts_raw_{idx:04d}.wav"
            fit_wav = temp_dir / f"tts_fit_{idx:04d}.wav"
            self._synthesize_to_wav(seg.text, raw_wav, cancel_event=cancel_event)
            target_len = max(0.25, seg.end_s - seg.start_s)
            time_stretch_audio_to_target(raw_wav, fit_wav, target_len, log_fn=self.log_fn, cancel_event=cancel_event)
            chunk_files.append(fit_wav)

        list_file = temp_dir / "concat_list.txt"
        lines = [f"file '{p.resolve().as_posix()}'" for p in chunk_files]
        list_file.write_text("\n".join(lines), encoding="utf-8")

        cmd = [
            get_ffmpeg_exe(),
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-acodec",
            "pcm_s16le",
            str(out_wav),
        ]
        run_command(cmd, env=get_base_env(), log_fn=self.log_fn, cancel_event=cancel_event)
