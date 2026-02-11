# English/Urdu to Pashto Dubbing End-to-End App

Windows desktop app to translate spoken English/Urdu videos into Pashto and generate a dubbed output in one pipeline.

## Project aim

This project is open source and built for community contribution.  
Its mission is practical Pashto inclusion in modern AI/video tooling, where Pashto is still underrepresented.

## What this app does

- Takes a video input (`.mp4`, `.mkv`, `.mov`, `.avi`)
- Extracts audio and translates speech to Pashto (`pbt`)
- Synthesizes Pashto voice audio
- Aligns dubbed audio to original video timing
- Applies lip sync when face/speech safety checks pass
- Produces:
  - `*_pashto_lipsync.mp4` (if lip-sync succeeds)
  - or fallback `*_pashto_dubbed.mp4` (safe non-lipsync output)
  - `pashto_translation.srt`
  - `pashto_translation.txt`
  - `pipeline.log`

## Current key features

- CPU-first by default, optional CUDA mode (`auto|cpu|cuda`)
- Source language mode (`auto|eng|urd`)
- Voice gender mode (`auto|male|female`)
- Source speaker gender estimation for better voice matching
- Chunk-level translation verification with round-trip confidence checks
- Term override file for manual correction (`configs/term_overrides.json`)
- Robust fallback behavior to avoid pipeline crashes

## Tech stack

- Translation: `facebook/seamless-m4t-v2-large`
- TTS: `edge-tts` Pashto neural voices (`ps-AF-*`)
- Lip sync: Wav2Lip
- Media processing: FFmpeg
- UI: Tkinter

## Repository policy (why heavy files are excluded)

To keep this repo portable and GitHub-friendly, heavy files are intentionally not tracked:

- model weights/checkpoints
- generated outputs and temp files
- local virtual environment
- test videos
- large binaries

You must download/install these separately (instructions below).

## Setup (Windows)

### 1. Clone repo

```powershell
git clone <your-repo-url>
cd <repo-folder>
```

### 2. Create virtual env and install deps

Preferred:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1
```

Manual:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install torch==2.5.1+cpu torchaudio==2.5.1+cpu torchvision==0.20.1+cpu --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
```

### 3. Install FFmpeg

Option A:
- Install FFmpeg globally and ensure `ffmpeg` + `ffprobe` are on PATH.

Option B:
- Place binaries in `tools/ffmpeg/bin/`:
  - `ffmpeg.exe`
  - `ffprobe.exe`

### 4. Download SeamlessM4T model files

Create folder:

`models/seamless-m4t-v2-large`

Download from Hugging Face repo `facebook/seamless-m4t-v2-large`:

- `config.json`
- `generation_config.json`
- `model.safetensors.index.json`
- `model-00001-of-00002.safetensors`
- `model-00002-of-00002.safetensors`
- `added_tokens.json`
- `preprocessor_config.json`
- `sentencepiece.bpe.model`
- `special_tokens_map.json`
- `tokenizer_config.json`

Verify:

```powershell
.\.venv\Scripts\python.exe .\scripts\verify_seamless_local.py
```

### 5. Wav2Lip assets

Required:

- local Wav2Lip code at `external/Wav2Lip`
- checkpoints:
  - `models/wav2lip/checkpoints/wav2lip_gan.pth`
  - `models/wav2lip/face_detection/detection/sfd/s3fd.pth`

You can download checkpoints via:

```powershell
.\.venv\Scripts\python.exe .\scripts\download_models.py
```

## Run the app

### GUI

```powershell
.\.venv\Scripts\python.exe .\run_gui.py
```

Or run:

`start_gui.bat`

### CLI

```powershell
.\.venv\Scripts\python.exe .\run_pipeline.py --input <path-to-video> --device cpu
```

Useful flags:

- `--source-lang auto|eng|urd`
- `--voice-gender auto|male|female`
- `--no-verify`

## Output structure

Each run creates:

`outputs/<video_name>_pashto_<timestamp>/`

Containing final video + subtitles + translated text + logs.

## Known limitations

- Pashto quality depends on currently available model/voice resources.
- Lip sync can degrade in complex scenes or fast motion.
- Multi-speaker active-speaker targeting is partially handled and still evolving.

## Contributing

Contributions are welcome:

- Bug reports
- Better Pashto terminology/glossary suggestions
- Translation quality improvements
- Speaker-aware lip-sync improvements
- Performance and stability enhancements

Please open an issue or PR with clear reproduction steps and expected behavior.

## Ethical and license note

Use dubbing/lip-sync responsibly and only where you have rights/permission to process media.  
Follow upstream model/tool licenses, especially Wav2Lip usage restrictions.
