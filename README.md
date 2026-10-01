# AI Streamer

Local real-time AI Twitch streamer: reads chat, talks back with a character voice, and lets the human host interrupt instantly via Push-to-Talk. Everything that needs a GPU runs on the stream PC (target: **RTX 4080 16 GB** + **Ryzen 5900X**, Windows 10/11).

The bot never writes to Twitch chat. It answers with speech.

## Tech stack

| Layer | Stack |
|---|---|
| Runtime | Python 3.11, `asyncio`, Pydantic v2, Loguru |
| Event bus | In-process async bus with preemption (PTT has absolute priority) |
| Twitch | `twitchio` 2.x IRC, `chat:read` only |
| Host PTT / STT | Global hotkey (`keyboard`), PortAudio (`sounddevice`), Faster-Whisper `large-v3`, Silero VAD |
| LLM | OpenAI-compatible HTTP client → vLLM (WSL2), llama.cpp server, or Ollama |
| TTS | Kokoro-82M streaming at 24 kHz, optional RVC (`rvc-python`) |
| Memory | Redis (short-term dialogue) + Qdrant (long-term facts) + `intfloat/multilingual-e5-base` |
| Audio I/O | `sounddevice` / PortAudio (persistent output stream; PTT must not kill the mic) |
| Infra | Docker Compose for Redis + Qdrant |

VRAM budget (see `VRAM__TOTAL_BUDGET_GB`, default **15 GB** of 16):

- LLM ~ 8–9 GB (14B AWQ/Q4)
- STT Faster-Whisper `large-v3` ~ 3 GB
- TTS Kokoro ~ 0.5–2.5 GB, plus optional RVC
- Embedder e5-base ~ 0.4 GB
- Reserve for VTube Studio / OBS ~ 2–3 GB

---

## What you get on a live stream

1. A viewer writes in Twitch chat (or bits / sub / raid).
2. Memory looks up known facts about that viewer or topic.
3. The LLM streams a reply in English, split on sentence pauses.
4. Kokoro (and optional RVC) synthesizes each clause and plays it on the selected output device.
5. If the host holds the PTT key, **LLM and TTS stop immediately**. The mic was already open; Whisper transcribes after release (`task=translate` → English even if you spoke Russian).

Priority order: **PTT (host) > donations/subs/raids > chat**.

---

## Hardware and OS

Designed for one Windows streaming box:

- NVIDIA GPU with **~16 GB VRAM** (RTX 4080 is the reference)
- **CUDA 12.x** PyTorch wheels (`cu124`)
- Fast SSD (first launch downloads several GB of model weights)
- Microphone + headphones / VB-Cable / mix for OBS
- Docker Desktop (Redis + Qdrant)

Linux works for the Python app, but **vLLM is Linux-only**. On Windows the LLM server is usually **Ollama**, **llama.cpp**, or **vLLM inside WSL2**.

---

## Prerequisites (clean PC)

Install these **before** Python packages:

1. **Git**
2. **Python 3.11 x64** from [python.org](https://www.python.org/downloads/)  
   During setup: enable **“Add python.exe to PATH”**.  
   Do **not** rely on the Microsoft Store `python` stub (it prints “Python was not found”).
3. **NVIDIA Game Ready / Studio driver** recent enough for CUDA 12.
4. **Docker Desktop** with the engine running.
5. **eSpeak NG** (Kokoro / `phonemizer` need it for G2P):  
   [https://github.com/espeak-ng/espeak-ng/releases](https://github.com/espeak-ng/espeak-ng/releases) — install the Windows `.msi`, then reboot or open a new terminal so `espeak-ng` is on `PATH`.
6. Optional but useful: **Visual C++ Redistributable** (latest x64).
7. An **OpenAI-compatible LLM server** that will listen on localhost (see below).
8. A Twitch **OAuth token** with scope `chat:read` — easiest: [https://twitchapps.com/tmi/](https://twitchapps.com/tmi/). The `oauth:` prefix is optional.

No system-wide CUDA Toolkit is required if PyTorch CUDA wheels are installed: the app loads `cublas` / `cudnn` from `torch/lib` on Windows.

---

## 1. Get the code

```powershell
git clone <this-repo-url> ai-streamer
cd ai-streamer
```

---

## 2. Python environment

Always use the venv interpreter. `run.ps1` / `run.bat` do that for you.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip wheel
```

If execution policy blocks `Activate.ps1`:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Or skip activation and call `.\.venv\Scripts\python.exe` everywhere.

---

## 3. Install dependencies (CUDA PyTorch first)

PyTorch must come from the CUDA 12.4 index **before** the rest of the stack:

```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
```

Check the GPU is visible:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

You want `True` and your NVIDIA card name.

---

## 4. Memory services (Redis + Qdrant)

From the repo root:

```powershell
docker compose up -d
docker compose ps
```

Defaults (already in `.env.example`):

- Redis: `redis://127.0.0.1:6379/0`
- Qdrant: `http://127.0.0.1:6333`

If Docker is down, the app still starts but memory degrades (facts are not stored / recalled). Data lives in named volumes; `docker compose down` keeps it, `docker compose down -v` wipes lore.

---

## 5. LLM server (separate process)

This repo is only the **client**. Point `LLM__BASE_URL` at any OpenAI-compatible `/v1` endpoint.

**Ollama (simplest on Windows)**

```powershell
ollama pull qwen2.5:7b
ollama serve
```

```
LLM__BASE_URL=http://127.0.0.1:11434/v1
LLM__MODEL=qwen2.5:7b
```

A 7B model leaves more VRAM for Whisper + Kokoro + OBS on a 16 GB card. A 14B AWQ/Q4 model is the architecture default and needs ~8.5 GB by itself.

**llama.cpp server**

Run the server with `--host 127.0.0.1` and an OpenAI-compatible path, then set `LLM__BASE_URL=http://127.0.0.1:8000/v1` (or whatever port you chose) and `LLM__MODEL` to the name the server expects.

**vLLM** — Linux / WSL2 only, not a native Windows process.

Keep the LLM on the same machine or a LAN box; the client streams tokens and must stay low-latency.

---

## 6. Configuration (`.env`)

```powershell
copy .env.example .env
```

Edit `.env`. Nested keys use `__` (example: `TTS__OUTPUT_DEVICE`). Flat aliases like `TWITCH_TOKEN` also work.

Minimum for a live stream:

| Variable | Purpose |
|---|---|
| `TWITCH_CHANNEL` | Channel whose chat is read (no `#`) |
| `TWITCH_TOKEN` | OAuth token, `chat:read` |
| `PTT__HOTKEY` | Host interrupt key (`f13` by default; change if your keyboard has no F13, e.g. `f8`) |
| `LLM__BASE_URL` | OpenAI-compatible server |
| `LLM__MODEL` | Model id that server exposes |
| `LLM__CHARACTER_NAME` | Substituted into `{name}` in the system prompt |
| `LLM__SYSTEM_PROMPT` | Character prompt (multiline, double quotes; escape inner `"` as `\"`). Appended user lines force **English** replies. |
| `STT__TASK` | `translate` (English transcript) or `transcribe` |
| `TTS__OUTPUT_DEVICE` | Playback device index or name; empty = Windows default |

Optional RVC after Kokoro:

```
TTS__RVC__ENABLED=true
TTS__RVC__MODEL_PATH=models/rvc/egirl.pth
TTS__RVC__INDEX_PATH=models/rvc/egirl.index
TTS__RVC__PITCH=0
TTS__RVC__DEVICE=cuda:0
TTS__RVC__F0_METHOD=rmvpe
```

Put `.pth` / `.index` under `models/rvc/` (that folder is gitignored). First RVC start may download Hubert / RMVPE into the package cache.

`.env` is gitignored. Never commit tokens.

**List audio devices:**

```powershell
.\.venv\Scripts\python.exe -c "import sounddevice as sd; print(sd.query_devices())"
```

Use a line with output channels for `TTS__OUTPUT_DEVICE`, and an input line for `PTT__INPUT_DEVICE` if the default mic is wrong.

---

## 7. First-run downloads

On the first full start the app (and Hugging Face Hub) will pull:

- Faster-Whisper `large-v3` (or whatever `STT__MODEL` is)
- Kokoro voice weights
- Sentence-Transformers `multilingual-e5-base`
- Optional RVC extras if enabled

On Windows, symlink creation in the HF cache often fails without Developer Mode. The app sets `HF_HUB_DISABLE_SYMLINKS=1` so files are copied instead. The first run can take a long time and several GB of disk.

---

## 8. Run

Prefer the wrappers so you never hit the Store stub:

```powershell
.\run.ps1
```

or:

```powershell
.\run.bat
```

or:

```powershell
.\.venv\Scripts\python.exe main.py
```

On Windows, a **global** PTT hook (`keyboard`) typically needs **Run as administrator**. Without elevation the hotkey may do nothing.

Ctrl+C stops the process.

### Launch modes

| Command | What starts |
|---|---|
| `python main.py --demo` | Event bus + mocks; no GPU, no Twitch, no Docker |
| `python main.py --twitch-only` | Live chat echoed to the console |
| `python main.py --ptt-only` | Mic + Whisper only |
| `python main.py --llm-only` | Twitch + PTT + LLM; `TTS_REQUEST` clauses printed, no speech |
| `python main.py` | Full pipeline: Twitch + PTT + STT + LLM + TTS + Memory |

Demo duration: `python main.py --demo --duration 20`.

---

## 9. Smoke tests (optional)

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Hydra (pulled in by `rvc-python`) is disabled via `pyproject.toml` pytest addopts.

---

## Troubleshooting

| Symptom | What to check |
|---|---|
| `python` opens the Store / “Python was not found” | Use `.\.venv\Scripts\python.exe` or `.\run.ps1` |
| `torch.cuda.is_available() is False` | NVIDIA driver + `pip install torch --index-url https://download.pytorch.org/whl/cu124` in **this** venv |
| VRAM / CUDA OOM | Lower LLM size (7B), disable RVC, close extra GPU apps, keep `VRAM__TOTAL_BUDGET_GB=15` |
| Config error “VRAM budget exceeded” | Sum of module `vram_gb` fields vs budget; do not inflate `TTS`/`STT` reservations blindly |
| PTT does nothing | Run the terminal as Administrator; confirm `PTT__HOTKEY`; test `--ptt-only` |
| Mic “hangs” ~10 s after interrupt | Do not call `sd.stop()` / stream abort from the UI thread — current `SoundPlayer` already avoids that |
| Kokoro / phonemizer errors | Install eSpeak NG and restart the terminal |
| Empty or Russian LLM replies | Keep `STT__TASK=translate` and a system prompt that stays in English; the client also appends `Respond in English.` |
| Twitch silent | Token scope `chat:read`, channel name without `#`, `--twitch-only` first |
| Redis/Qdrant connection errors | `docker compose ps`; ports bound to `127.0.0.1` |

Logs: console + `logs/ai-streamer.log` (path from `LOG_FILE` / default in settings).

---

## Quick Start

On a clean Windows PC with an NVIDIA GPU:

1. Install **Git**, **Python 3.11**, **NVIDIA driver**, **Docker Desktop**, **eSpeak NG**.
2. `git clone` this repo and `cd` into it.
3. `py -3.11 -m venv .venv` then `.\.venv\Scripts\Activate.ps1`.
4. `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124`
5. `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124`
6. `docker compose up -d`
7. Start an LLM server (e.g. Ollama) and note its `/v1` URL.
8. `copy .env.example .env` and fill `TWITCH_CHANNEL`, `TWITCH_TOKEN`, `LLM__BASE_URL`, `LLM__MODEL`, persona fields, `PTT__HOTKEY`, optional `TTS__OUTPUT_DEVICE`.
9. Sanity: `python main.py --demo`, then `--twitch-only`, then `--ptt-only`.
10. Full stream (admin terminal): `.\run.ps1`

You should hear Kokoro on the chosen output device, see chat in the logs, and be able to cut speech instantly with PTT.
