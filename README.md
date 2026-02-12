# 🎬 English/Urdu to Pashto Dubbing End-to-End App

🖥️ Windows desktop app to translate spoken English/Urdu videos into Pashto and generate dubbed output in one pipeline.

## 🚀 Current baseline (v1)

- 🎞️ Input: video (`.mp4`, `.mkv`, `.mov`, `.avi`)
- 🌐 Translation target: Pashto (`pbt`) via SeamlessM4T
- 🗣️ TTS: Pashto neural voices via `edge-tts` (with fallback handling)
- 📦 Output:
  - `*_pashto_lipsync.mp4` (when lip-sync gate and Wav2Lip succeed), or
  - `*_pashto_dubbed.mp4` (safe fallback mux)
  - `pashto_translation.srt`
  - `pashto_translation.txt`
  - `pipeline.log`

## ✨ Features in this release

- ⚙️ CPU-first default, optional CUDA mode (`auto|cpu|cuda`)
- 🧭 Source language mode (`auto|eng|urd`)
- 👤 Voice gender mode (`auto|male|female`) with source-speaker pitch hinting
- 🧩 Chunked translation with optional round-trip verification
- 📝 Term overrides via `configs/term_overrides.json`
- 🛡️ Lip-sync safety gate with fallback export when conditions are not suitable

## 🧠 Tech stack

- 🌐 Translation: `facebook/seamless-m4t-v2-large`
- 🔊 TTS: `edge-tts` (`ps-AF-*` voices)
- 👄 Lip sync: Wav2Lip
- 🎛️ Media processing: FFmpeg / FFprobe
- 🪟 GUI: Tkinter

## 📚 Repository notes

- 📁 The Git repository root is this `App/` folder.
- 🧳 Heavy files are intentionally excluded from git: models, generated outputs, local venv, external repos, binaries, test media.
- 📄 Additional project docs:
  - `PROJECT_REPORT.md`
  - `ROADMAP.md`
  - `CONTRIBUTING.md`

## 🛠️ Setup (Windows)

### 1) 📥 Clone and enter repo

```powershell
git clone https://github.com/Musawer1214/English-Urdu-to-Pashto-dubbing-end-to-end-app.git
cd English-Urdu-to-Pashto-dubbing-end-to-end-app
```

### 2) 🐍 Create venv and install dependencies

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

### 3) 🎞️ Install FFmpeg

Option A ✅ (recommended):
- Install FFmpeg globally and ensure `ffmpeg` + `ffprobe` are on `PATH`.

Option B 📁:
- Place binaries in `tools/ffmpeg/bin/`:
  - `ffmpeg.exe`
  - `ffprobe.exe`

### 4) 🤖 Download local SeamlessM4T model files

Create:
- `models/seamless-m4t-v2-large/`

Download required files from `facebook/seamless-m4t-v2-large`:
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

Verify local model folder ✅:

```powershell
.\.venv\Scripts\python.exe .\scripts\verify_seamless_local.py
```

### 5) 👄 Prepare Wav2Lip assets

Required:
- Wav2Lip repository at `external/Wav2Lip`
- checkpoints:
  - `models/wav2lip/checkpoints/wav2lip_gan.pth`
  - `models/wav2lip/face_detection/detection/sfd/s3fd.pth`

Download checkpoints ⬇️:

```powershell
.\.venv\Scripts\python.exe .\scripts\download_models.py
```

## ▶️ Run

### 🪟 GUI

```powershell
.\.venv\Scripts\python.exe .\run_gui.py
```

or:

```powershell
.\start_gui.bat
```

### ⌨️ CLI

```powershell
.\.venv\Scripts\python.exe .\run_pipeline.py --input <path-to-video>
```

Supported CLI options 🧾:
- `--output-root <dir>`
- `--device auto|cpu|cuda`
- `--source-lang auto|eng|urd`
- `--target-lang <lang-code>` (default `pbt`)
- `--model <hf-model-id>`
- `--model-local-dir <path>`
- `--chunk-seconds <int>`
- `--voice-gender auto|male|female`
- `--no-verify`

## 📤 Output layout

Each run creates 🗂️:
- `outputs/<video_name>_pashto_<timestamp>/`

Artifacts inside that folder 📌:
- final video (`*_pashto_lipsync.mp4` or `*_pashto_dubbed.mp4`)
- `pashto_translation.srt`
- `pashto_translation.txt`
- `pipeline.log`

## ⚠️ Known limitations

- 🎯 Pashto quality depends on currently available translation/voice resources.
- 🎥 Wav2Lip can degrade on complex motion, profile faces, and occlusions.
- 👥 Multi-speaker active-speaker targeting is partial and still evolving.
- 🌐 Cloud TTS requires stable internet access.

## ⚖️ Ethical and license note

Use dubbing/lip-sync responsibly and only where you have legal rights/permission to process the media.  
Follow upstream model/tool licenses, especially Wav2Lip usage restrictions.
