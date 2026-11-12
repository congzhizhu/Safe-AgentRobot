@echo off
title Safety_Agent Launcher
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
if not exist ".venv\Scripts\python.exe" (
  echo First run: creating virtual environment and installing dependencies...
  echo This may take 5-10 minutes. Please keep network connection.
  python -m venv .venv
  if errorlevel 1 (
    echo [ERROR] Failed to create venv. Please install Python 3.10+ first.
    pause
    exit /b 1
  )
  .venv\Scripts\python.exe -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
)
if not exist ".env" (
  echo [WARN] .env not found. Copy .env.example to .env and fill in your API key.
)
echo ==============================================
echo   Safety_Agent - starting services...
echo   WebUI: http://127.0.0.1:7860
echo   Stop : press Ctrl + C in this window
echo ==============================================
.venv\Scripts\python.exe start_all_windows.py
pause
