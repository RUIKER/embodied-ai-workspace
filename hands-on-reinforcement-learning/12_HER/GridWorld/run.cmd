@echo off
setlocal
set "HERE=%~dp0"
set "PY=%HERE%..\..\.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo [ERROR] venv python not found: %PY%
    echo Run: ../../.venv/Scripts/python.exe -m pip install -r requirements.txt
    exit /b 1
)
"%PY%" "%HERE%main.py" %*
endlocal
