$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    if (Test-Path "C:\Users\musaw\AppData\Local\Programs\Python\Python310\python.exe") {
        & "C:\Users\musaw\AppData\Local\Programs\Python\Python310\python.exe" -m venv "$root\.venv"
    } else {
        throw "Python 3.10 was not found. Install Python 3.10 and re-run this script."
    }
}

& $venvPython -m pip install --upgrade pip setuptools wheel
& $venvPython -m pip install torch==2.5.1+cpu torchaudio==2.5.1+cpu torchvision==0.20.1+cpu --index-url https://download.pytorch.org/whl/cpu
& $venvPython -m pip install -r (Join-Path $root "requirements.txt")
& $venvPython (Join-Path $root "scripts\download_models.py")

Write-Host "Environment setup complete."
Write-Host "Run GUI with: .\.venv\Scripts\python.exe .\run_gui.py"
