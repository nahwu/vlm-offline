# vlm-offline

REST API project for running `Qwen/Qwen2.5-VL-7B-Instruct` with image or video input, plus a web application UI and deterministic test assets.

## What is Included
- FastAPI server for multimodal query inference:
  - `POST /v1/query` accepts `query + single image/video`.
  - `POST /v1/query/batch` accepts `query + multiple files`.
  - `video_pipeline` supports `sampled_frames` and `full_temporal`.
- Web application UI:
  - `GET /` serves `app/static/index.html`.
  - Frontend assets: `app/static/styles.css` and `app/static/app.js`.
  - Supports multi-file selection, server batch vs client sequential coordination, request history, and live metrics polling.
- Deterministic media generator:
  - `tests/generate_deterministic_media.py`.
- Programmatic REST tests:
  - `tests/test_api_requests.py`.
- Runtime observability:
  - Request IDs (`X-Request-ID` response header).
  - JSON request logs with latency and GPU memory.
  - `GET /metrics` endpoint with counters and latency stats.
- Deployment plan:
  - `docs/DEPLOYMENT_PLAN.md`.

## Single-Machine Deployment (Docker Preferred)

### CPU container
```bash
docker compose up --build vlm-api
```

### GPU container (NVIDIA)
```bash
docker compose --profile gpu up --build vlm-api-gpu -d
```

Then open `http://127.0.0.1:8000/`.

Check that the container is actually using CUDA:
```bash
curl http://127.0.0.1:8000/health
```

The response should show `cuda_available: true`, a `gpu_name`, and a non-empty `torch_cuda` value. If it reports CPU, verify Docker GPU access first:
```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
```

On a 16 GB RTX 5080, the GPU service defaults to 4-bit NF4 loading for `Qwen2.5-VL-7B-Instruct`. This keeps the 7B model on the GPU instead of offloading layers to CPU, which is usually much faster than fp16 offload while preserving more accuracy than switching to a smaller model.

## Web Application
- Open `http://127.0.0.1:8000/`.
- Build a request with one or more image/video files.
- Choose:
  - `video_pipeline`: `sampled_frames` or `full_temporal`
  - `coordination`: `server_batch` or `client_sequential`
- Submit and review:
  - Per-file response cards
  - `X-Request-ID` in status
  - Runtime stats from `/metrics`
  - Local request history in the UI

## Quick Start (Windows PowerShell)
```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python tests\generate_deterministic_media.py
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Open browser: `http://127.0.0.1:8000/`

## API Usage
### Health check
```bash
curl http://127.0.0.1:8000/health
```

### Image query
```bash
curl -X POST "http://127.0.0.1:8000/v1/query" \
  -F "query=Describe the main objects in the image." \
  -F "video_pipeline=sampled_frames" \
  -F "file=@tests/assets/deterministic_sample.png"
```

### Video query
```bash
curl -X POST "http://127.0.0.1:8000/v1/query" \
  -F "query=Summarize what changes in the video." \
  -F "video_pipeline=full_temporal" \
  -F "max_video_frames=8" \
  -F "file=@tests/assets/deterministic_sample.mp4"
```

### Batch query
```bash
curl -X POST "http://127.0.0.1:8000/v1/query/batch" \
  -F "query=Answer each file with one concise summary." \
  -F "video_pipeline=sampled_frames" \
  -F "files=@tests/assets/deterministic_sample.png" \
  -F "files=@tests/assets/deterministic_sample.mp4"
```

### Runtime metrics
```bash
curl http://127.0.0.1:8000/metrics
```

## Scripted Test
Run both deterministic image/video requests:
```powershell
python tests\test_api_requests.py --base-url http://127.0.0.1:8000
```

Run additional batch check:
```powershell
python tests\test_api_requests.py --base-url http://127.0.0.1:8000 --run-batch
```

## Config
Copy `.env.example` to `.env` if needed:
- `MODEL_NAME` default: `Qwen/Qwen2.5-VL-7B-Instruct`
- `MAX_NEW_TOKENS` default: `128`
- `IMAGE_MIN_PIXELS` default: `50176` (`224 * 224`)
- `IMAGE_MAX_PIXELS` default: `1003520` (`1280 * 28 * 28`) for the GPU service; CPU Compose uses `602112`
- `MODEL_QUANTIZATION` default: `nf4` for GPU Compose, `none` for CPU Compose
- `MODEL_COMPUTE_DTYPE` default: `bfloat16` for GPU Compose
- `ATTN_IMPLEMENTATION` default: `sdpa` for GPU Compose
- `MAX_VIDEO_FRAMES` default: `8`
- `VIDEO_FPS` default: `1.0` (used in `full_temporal` pipeline)
- `IMAGE_MAX_PIXELS` controls visual tokens and latency. Use `401408` (`512 * 28 * 28`) for faster coarse descriptions, `602112` for balanced image QA, and the default `1003520` for small text or fine visual detail.

## Notes
- First run downloads model weights and may take several minutes.
- GPU is strongly recommended for practical latency.
- The Docker build installs CUDA-enabled PyTorch wheels. The container does not need a full CUDA toolkit unless you enable optional Flash Attention builds.
- If startup logs show parameters on `cpu` or `meta`, the model is offloading and latency will suffer. Keep `MODEL_QUANTIZATION=nf4`, reduce `IMAGE_MAX_PIXELS`, or use a smaller model.
- Server batch mode processes files sequentially by default for stability; the HTML UI also supports `client_sequential` coordination.
