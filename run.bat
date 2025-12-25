@echo off
setlocal
set "ROOT_DIR=%~dp0"
set "PYTHON_EXE=%ROOT_DIR%runtime\python.exe"

if not exist "%PYTHON_EXE%" (
    echo Runtime not found. Running setup_runtime.ps1...
    powershell -ExecutionPolicy Bypass -File "%ROOT_DIR%setup_runtime.ps1"
)

echo Starting PID Auto Tuner...
"%PYTHON_EXE%" -m streamlit run src/app.py
pause
