from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import shutil
import threading
import traceback

from .audio_profile import detect_primary_speaker_gender
from .config import OUTPUTS_DIR, PipelineConfig, ensure_layout
from .ffmpeg_utils import extract_mono_wav, mux_audio_with_video, probe_duration_seconds, time_stretch_audio_to_target
from .seamless_service import SeamlessTranslator
from .speaker_gate import evaluate_lipsync_gate
from .tts_service import PashtoTTSService
from .utils import PipelineCancelledError, ProgressFn
from .wav2lip_service import Wav2LipRunner


@dataclass(slots=True)
class PipelineResult:
    job_dir: Path
    source_audio: Path
    translated_audio: Path
    translated_audio_synced: Path
    translated_srt: Path
    translated_text: Path
    final_video: Path


class VideoDubPipeline:
    def __init__(
        self,
        config: PipelineConfig,
        log_fn,
        cancel_event: threading.Event | None = None,
    ) -> None:
        self.config = config
        self.log_fn = log_fn
        self.cancel_event = cancel_event or threading.Event()
        self.translator = SeamlessTranslator(config, log_fn=log_fn)
        self.tts = PashtoTTSService(config, log_fn=log_fn)
        self.lipsync = Wav2LipRunner(log_fn=log_fn)

    def request_cancel(self) -> None:
        self.cancel_event.set()

    def _ensure_not_cancelled(self) -> None:
        if self.cancel_event.is_set():
            raise PipelineCancelledError("Pipeline cancelled by user.")

    def run(self, input_video: Path, output_root: Path | None = None, progress_fn: ProgressFn | None = None) -> PipelineResult:
        progress = progress_fn or (lambda *_: None)
        ensure_layout()
        if output_root is None:
            output_root = OUTPUTS_DIR
        output_root = Path(output_root).resolve()
        output_root.mkdir(parents=True, exist_ok=True)

        input_video = input_video.resolve()
        if not input_video.exists():
            raise FileNotFoundError(f"Input video does not exist: {input_video}")
        self._ensure_not_cancelled()

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_dir = output_root / f"{input_video.stem}_pashto_{stamp}"
        temp_dir = job_dir / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)

        source_wav = temp_dir / "source_16k.wav"
        translated_wav = temp_dir / "pashto_raw.wav"
        synced_wav = temp_dir / "pashto_synced.wav"
        translated_srt = job_dir / "pashto_translation.srt"
        translated_text = job_dir / "pashto_translation.txt"
        lipsync_video = job_dir / f"{input_video.stem}_pashto_lipsync.mp4"
        fallback_video = job_dir / f"{input_video.stem}_pashto_dubbed.mp4"
        log_file = job_dir / "pipeline.log"

        def _log(msg: str) -> None:
            timestamp = datetime.now().strftime("%H:%M:%S")
            line = f"[{timestamp}] {msg}"
            self.log_fn(line)
            with log_file.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

        try:
            progress(0.03, "Extracting source audio")
            _log(f"Input video: {input_video}")
            extract_mono_wav(
                input_video,
                source_wav,
                sample_rate=16000,
                log_fn=_log,
                cancel_event=self.cancel_event,
            )
            self._ensure_not_cancelled()
            video_duration = probe_duration_seconds(input_video)
            _log(f"Video duration: {video_duration:.2f}s")
            gender = detect_primary_speaker_gender(source_wav, log_fn=_log)
            self.tts.set_source_gender_hint(gender.gender)
            _log(
                f"Voice profile selected from source: {gender.gender} "
                f"(confidence={gender.confidence:.2f}, f0={gender.median_f0_hz:.1f}Hz)"
            )

            progress(0.12, "Translating audio to Pashto text")
            artifacts = self.translator.translate_audio(
                source_wav=source_wav,
                output_srt=translated_srt,
                output_text=translated_text,
                progress_fn=lambda p, m: progress(0.12 + 0.43 * p, m),
                cancel_event=self.cancel_event,
            )
            _log(f"Translated {artifacts.chunks_count} chunks")
            self._ensure_not_cancelled()

            progress(0.56, "Synthesizing Pashto speech")
            self.tts.synthesize_segments(
                artifacts.segments,
                translated_wav,
                temp_dir / "tts_chunks",
                cancel_event=self.cancel_event,
            )
            self._ensure_not_cancelled()

            progress(0.72, "Aligning Pashto audio length to video")
            before_dur, after_dur = time_stretch_audio_to_target(
                translated_wav,
                synced_wav,
                video_duration,
                log_fn=_log,
                cancel_event=self.cancel_event,
            )
            _log(
                f"Audio duration adjusted {before_dur:.2f}s -> {after_dur:.2f}s to match video."
            )
            if self.tts.is_near_silent(synced_wav):
                raise RuntimeError(
                    "Generated Pashto audio is silent. TTS failed or produced invalid output. "
                    "Check internet connectivity for edge-tts and retry."
                )

            final_video = lipsync_video
            gate = evaluate_lipsync_gate(input_video, source_wav)
            _log(
                f"Lip-sync gate: should={gate.should_lipsync}, speech_ratio={gate.speech_ratio:.2f}, "
                f"face_ratio={gate.face_ratio:.2f}, reason={gate.reason}"
            )
            self._ensure_not_cancelled()
            if gate.should_lipsync:
                progress(0.78, "Running lip-sync")
                try:
                    self.lipsync.run_lip_sync(
                        input_video=input_video,
                        input_audio=synced_wav,
                        output_video=lipsync_video,
                        device_policy=self.config.device_policy,
                        dominant_box=gate.dominant_box,
                        cancel_event=self.cancel_event,
                    )
                except PipelineCancelledError:
                    raise
                except Exception as lipsync_exc:
                    _log(f"Lip-sync failed after retries, exporting dubbed fallback video: {lipsync_exc}")
                    mux_audio_with_video(
                        input_video,
                        synced_wav,
                        fallback_video,
                        log_fn=_log,
                        cancel_event=self.cancel_event,
                    )
                    final_video = fallback_video
            else:
                _log("Skipping lip-sync for this video and exporting dubbed fallback video.")
                mux_audio_with_video(
                    input_video,
                    synced_wav,
                    fallback_video,
                    log_fn=_log,
                    cancel_event=self.cancel_event,
                )
                final_video = fallback_video

            progress(1.0, "Done")
            _log(f"Final video: {final_video}")
            return PipelineResult(
                job_dir=job_dir,
                source_audio=source_wav,
                translated_audio=translated_wav,
                translated_audio_synced=synced_wav,
                translated_srt=translated_srt,
                translated_text=translated_text,
                final_video=final_video,
            )
        except PipelineCancelledError as exc:
            self.log_fn(f"[{datetime.now().strftime('%H:%M:%S')}] {exc}")
            if job_dir.exists():
                shutil.rmtree(job_dir, ignore_errors=True)
            raise
        except Exception as exc:
            _log(f"Pipeline failed: {exc}")
            _log(traceback.format_exc())
            raise
