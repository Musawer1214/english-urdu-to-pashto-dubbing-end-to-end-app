# 📘 Pashto Video Dubbing Desktop App - Project Report

## 0. 🧭 Repository Alignment Snapshot (February 12, 2026)

- 🔗 GitHub remote: `origin` -> `https://github.com/Musawer1214/English-Urdu-to-Pashto-dubbing-end-to-end-app.git`
- 🏷️ Active release baseline: tag `v1`
- 📚 Documentation baseline includes:
  - `README.md`
  - `PROJECT_REPORT.md`
  - `CONTRIBUTING.md`
  - `ROADMAP.md`
  - `issues/2026-02-11-issue-add-pashto-terminology-database.md`
- 📁 Repo root for this project is the `App/` folder (commands in docs assume this root).

---

## 1. 🎯 Project Overview

### 1.1 🌍 Mission
Build a Windows desktop app that takes an English or Urdu video as input and outputs a Pashto dubbed video in one end-to-end flow:
- Audio extraction
- Speech translation to Pashto
- Pashto speech synthesis
- Audio-video alignment
- Lip-sync (when safe and applicable)
- Final exported video + subtitles + translated text

### 1.2 🧩 Current Target
- Platform: Windows (desktop-first)
- Execution mode: CPU-first, with optional CUDA/GPU mode
- Target Pashto code in model pipeline: `pbt` (general Pashto available in SeamlessM4T)

### 1.3 📦 Current Product Outcome
For each run, the app generates:
- Final video (`*_pashto_lipsync.mp4` or fallback `*_pashto_dubbed.mp4`)
- Pashto subtitle file (`pashto_translation.srt`)
- Pashto text transcript (`pashto_translation.txt`)
- Full processing log (`pipeline.log`)

---

## 2. 🏗️ System Architecture

### 2.1 🚪 Entry Points
- GUI launcher: `run_gui.py`
- CLI runner: `run_pipeline.py`
- One-click Windows launcher: `start_gui.bat`

### 2.2 🧠 Core Modules
- Config and paths: `src/config.py`
- Orchestration pipeline: `src/pipeline.py`
- Translation and verification: `src/seamless_service.py`
- TTS generation and retries: `src/tts_service.py`
- Speaker gender estimation: `src/audio_profile.py`
- Lip-sync safety gate: `src/speaker_gate.py`
- Wav2Lip runtime integration: `src/wav2lip_service.py`
- FFmpeg utility layer: `src/ffmpeg_utils.py`
- Common process and SRT helpers: `src/utils.py`

### 2.3 🔌 External Dependencies
- SeamlessM4T v2 large (`facebook/seamless-m4t-v2-large`)
- Wav2Lip model checkpoints
- FFmpeg/FFprobe binaries
- Edge TTS Pashto neural voices (cloud)
- Optional eSpeak fallback (local, lower quality)

---

## 3. 🔄 End-to-End Processing Flow

1. Input video is selected (GUI or CLI).
2. Pipeline extracts mono 16k source audio from video.
3. Source speaker gender is estimated from pitch statistics.
4. Translation service:
   - Detects source language (`eng`/`urd`) if `auto`.
   - Splits audio into chunks.
   - Produces Pashto text per chunk.
   - Optionally verifies chunk quality using round-trip strategy checks.
   - Applies manual term overrides from `configs/term_overrides.json`.
5. TTS service synthesizes Pashto audio per chunk:
   - Chooses voice candidates with gender-aware priority.
   - Retries across voices and proxy options if needed.
   - Rejects invalid/near-silent output.
6. Chunk audio is duration-aligned and concatenated.
7. Final Pashto audio is stretched to video duration.
8. Lip-sync gate evaluates if video is suitable:
   - Face continuity threshold
   - Speech activity threshold
   - Dominant face box estimation
9. If gate passes, Wav2Lip runs with retries and fallback attempt settings.
10. If gate fails or Wav2Lip fails, app exports safe dubbed fallback (video + Pashto audio mux).
11. Artifacts are written in a timestamped job folder under `outputs/`.

---

## 4. ✨ What We Built (Major Capabilities)

### 4.1 ⚙️ CPU-First with Optional GPU
- Default behavior supports systems without CUDA.
- Device policy can be switched between `auto`, `cpu`, and `cuda`.

### 4.2 🤖 Local-First Seamless Model Loading
- If required files exist in `models/seamless-m4t-v2-large`, app loads locally.
- Prevents repeated download pressure and improves portability.

### 4.3 ✅ Translation Verification Layer
- Added per-chunk quality check using:
  - direct speech-to-Pashto output
  - alternative text-to-text output from source transcript
  - back-translation similarity scoring
- Best strategy is selected chunk-by-chunk using configurable score margin.

### 4.4 👤 Gender-Aware Voice Selection
- Added speaker gender detection and voice-priority control.
- `tts_gender_mode` supports `auto`, `male`, `female`.
- Reduces male/female mismatch in dubbed output.

### 4.5 🛡️ Crash Resistance and Fallback Strategy
- Added checks to detect silent/invalid synthesized audio.
- Robust retries for cloud TTS and Wav2Lip inference.
- Always produce dubbed fallback video when lip-sync is unsuitable.

### 4.6 👄 Lip-Sync Safety Gating
- Prevents lip-sync attempts when no stable visible face or insufficient speech exists.
- Avoids poor output and unnecessary failures on non-speaking/no-face video parts.

### 4.7 🖥️ Usable Desktop GUI
- File picker for input/output.
- Queue support for multiple videos.
- Real-time logs and progress bar.
- Completion status and direct path to last output file.
- Controls for language source mode, gender mode, chunk size, and verification toggle.

---

## 5. 🧯 Problems Solved and How

### 5.1 🔇 Silent Output Video
- Problem: Some produced videos had no audible dubbed sound.
- Fixes:
  - Added near-silence detection after TTS and post-alignment audio.
  - Added stricter output validation (file size, duration, silence checks).
  - Fixed FFmpeg concat list behavior by using resolved absolute chunk paths.
  - Added hard failure path when audio remains invalid (instead of exporting bad output).

### 5.2 🗣️ Gender Mismatch in Dubbed Voice
- Problem: Source male/female was not reflected in Pashto TTS voice.
- Fixes:
  - Added source voice profiling (`src/audio_profile.py`).
  - Introduced gender-aware voice candidate ordering in TTS.
  - Exposed manual override in GUI/CLI (`auto|male|female`).

### 5.3 🌐 Urdu/English Detection Quality
- Problem: Auto language detection could choose wrong source language in edge cases.
- Fixes:
  - Added script-ratio heuristics (Latin vs Arabic/Urdu).
  - Combined with likeness scoring from probe transcripts.
  - Logged detection metrics for troubleshooting.

### 5.4 🧭 Lip-Sync Failure and Relative Path Issues
- Problem: Lip-sync step failed in some runs due to path resolution and video conditions.
- Fixes:
  - Output root is resolved to absolute path before job creation.
  - Gate added to skip lip-sync for unsuitable media and safely export dubbed fallback.
  - Added multi-attempt Wav2Lip strategy with alternative inference options.

### 5.5 ⏱️ Operational Stability for Long Processing
- Problem: Heavy pipeline requires robust behavior on slower CPU paths.
- Fixes:
  - Stage-wise progress reporting and detailed logging.
  - Retry loops for network-dependent TTS.
  - Defensive checks at each stage before moving forward.

---

## 6. ⚠️ Current Constraints / Known Gaps

1. Pashto accent/dialect control is limited to available `pbt` resources and voice inventory.
2. Translation verification improves reliability but cannot guarantee perfect semantic equivalence for all chunks.
3. Multi-speaker scenes are only partially handled:
   - dominant face lock exists
   - true active-speaker identity tracking is not fully implemented yet
4. Cloud TTS quality and availability depend on internet/network conditions.
5. Wav2Lip may still produce artifacts in complex motion, occlusion, or profile-face shots.

---

## 7. 🧪 Testing and Validation Status

### 7.1 ✅ Baseline Verification Commands
- Compile sanity check:
  - `.\.venv\Scripts\python.exe -m compileall .\src .\run_pipeline.py .\run_gui.py`
- Seamless local model presence check:
  - `.\.venv\Scripts\python.exe .\scripts\verify_seamless_local.py`
- CLI option check:
  - `.\.venv\Scripts\python.exe .\run_pipeline.py --help`
- GUI startup check:
  - `.\.venv\Scripts\python.exe .\run_gui.py`

### 7.2 📤 Output Location Pattern
- `outputs/<input_name>_pashto_<timestamp>/`
- Final file will be either:
  - `<input_name>_pashto_lipsync.mp4` (if gate + lip-sync pass)
  - `<input_name>_pashto_dubbed.mp4` (safe fallback)

---

## 8. 🚀 Future Plan (Implementation Roadmap)

### Phase 1 - 🎯 Accuracy and Voice Quality
1. Add optional second-pass translation QA using constrained glossary correction per domain.
2. Expand Pashto pronunciation lexicon and automated phonetic fixes.
3. Add confidence annotations in SRT for manual reviewer targeting.

### Phase 2 - 👥 Speaker-Aware Lip Sync
1. Add active speaker diarization + mouth-motion correlation.
2. Track speaker identity across frames.
3. Apply lip-sync only to the speaking person in multi-person scenes.

### Phase 3 - ⚡ Robustness and Performance
1. Add checkpoint/restart support for long videos.
2. Add persistent job metadata and resume from last successful stage.
3. Add optional low-memory mode and chunk scheduling tuning.

### Phase 4 - 📦 Productization and Portability
1. Package Windows installer with dependency checks.
2. Add configuration profiles (quality-first vs speed-first).
3. Prepare abstraction layer for Linux/macOS expansion.

### Phase 5 - 📈 Quality Monitoring
1. Build regression test set (English + Urdu -> Pashto expected behavior).
2. Add automated smoke tests for:
   - non-silent audio
   - subtitle generation
   - expected output file creation
3. Track metrics over time:
   - translation confidence distribution
   - TTS retry rate
   - lip-sync success/fallback ratio

---

## 9. 📝 Suggested Working Document Practice

Use this file as the project baseline and keep it updated each sprint:
1. Append new solved issues under Section 5.
2. Move completed roadmap items from Section 8 into Section 4.
3. Add measurable quality numbers in Section 7 after each test cycle.

This will keep implementation history, technical decisions, and next actions in one place for future contributors.
