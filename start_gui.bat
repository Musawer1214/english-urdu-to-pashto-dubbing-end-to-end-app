@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Virtual environment not found at .venv\Scripts\python.exe
  echo Run scripts\setup_env.ps1 first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" run_gui.py
endlocal
