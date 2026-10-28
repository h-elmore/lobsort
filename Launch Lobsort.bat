@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\lobsort.exe (
  py -m venv .venv
  .venv\Scripts\python.exe -m pip install --upgrade pip
  .venv\Scripts\python.exe -m pip install -e .
  if errorlevel 1 (pause & exit /b 1)
)
.venv\Scripts\lobsort.exe %*
