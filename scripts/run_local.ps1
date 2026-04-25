param(
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if (-not (Test-Path "tests\assets\deterministic_sample.png") -or -not (Test-Path "tests\assets\deterministic_sample.mp4")) {
    python tests\generate_deterministic_media.py
}

uvicorn app.main:app --host $HostAddress --port $Port --reload
