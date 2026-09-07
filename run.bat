@echo off
REM Всегда использует Python из .venv, минуя заглушку Microsoft Store.
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo Сначала создайте окружение: py -3.11 -m venv .venv
  echo Затем: .venv\Scripts\python.exe -m pip install -r requirements.txt
  exit /b 1
)
"%PY%" "%~dp0main.py" %*
