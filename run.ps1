# Всегда запускает main.py через .venv, минуя заглушку Microsoft Store (`python` → «Python»).
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Error "Нет $py. Создайте окружение: py -3.11 -m venv .venv"
}
& $py (Join-Path $root "main.py") @args
exit $LASTEXITCODE
