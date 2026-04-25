import json
import logging
import tempfile
import time
import uuid
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.metrics import RuntimeMetrics
from app.model_service import VLMService, load_config_from_env


app = FastAPI(title="Qwen2.5-VL-7B REST API", version="0.1.0")
service = VLMService(config=load_config_from_env())
metrics = RuntimeMetrics()
logger = logging.getLogger("uvicorn.error")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.middleware("http")
async def request_tracking_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    started = time.perf_counter()
    gpu_allocated_mb = 0.0
    gpu_reserved_mb = 0.0

    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception:
        status_code = 500
        raise
    finally:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if torch.cuda.is_available():
            gpu_allocated_mb = float(torch.cuda.memory_allocated() / (1024 * 1024))
            gpu_reserved_mb = float(torch.cuda.memory_reserved() / (1024 * 1024))

        metrics.record(
            latency_ms=elapsed_ms,
            status_code=status_code,
            gpu_allocated_mb=gpu_allocated_mb,
            gpu_reserved_mb=gpu_reserved_mb,
        )

        logger.info(
            json.dumps(
                {
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": status_code,
                    "latency_ms": round(elapsed_ms, 2),
                    "gpu_allocated_mb": round(gpu_allocated_mb, 2),
                    "gpu_reserved_mb": round(gpu_reserved_mb, 2),
                }
            )
        )

    response.headers["X-Request-ID"] = request_id
    return response


def _save_upload_to_temp(upload: UploadFile) -> str:
    suffix = Path(upload.filename or "upload").suffix or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        upload.file.seek(0)
        tmp.write(upload.file.read())
        return tmp.name


def _run_inference_for_upload(upload: UploadFile, query: str, max_video_frames: int, video_pipeline: str) -> dict:
    tmp_path = _save_upload_to_temp(upload)
    filename = upload.filename or "uploaded"

    try:
        answer = service.infer_from_file(
            file_path=tmp_path,
            content_type=upload.content_type or "",
            query=query,
            video_pipeline=video_pipeline,
            max_video_frames=max_video_frames,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return {
        "filename": filename,
        "content_type": upload.content_type or "",
        "answer": answer,
    }


@app.get("/")
def index() -> RedirectResponse:
    return RedirectResponse(url="/static/index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "device": service.device}


@app.get("/metrics")
def get_metrics() -> dict:
    return metrics.snapshot()


@app.post("/v1/query")
async def query_vlm(
    query: str = Form(...),
    file: UploadFile = File(...),
    max_video_frames: int = Form(8),
    video_pipeline: str = Form("sampled_frames"),
) -> dict:
    if video_pipeline not in {"sampled_frames", "full_temporal"}:
        raise HTTPException(status_code=400, detail="video_pipeline must be sampled_frames or full_temporal")

    try:
        result = _run_inference_for_upload(
            upload=file,
            query=query,
            max_video_frames=max_video_frames,
            video_pipeline=video_pipeline,
        )
    except HTTPException:
        raise
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex)) from ex

    return {
        "model": service.config.model_name,
        "query": query,
        "video_pipeline": video_pipeline,
        "filename": result["filename"],
        "answer": result["answer"],
    }


@app.post("/v1/query/batch")
async def query_vlm_batch(
    query: str = Form(...),
    files: list[UploadFile] = File(...),
    max_video_frames: int = Form(8),
    video_pipeline: str = Form("sampled_frames"),
) -> JSONResponse:
    if video_pipeline not in {"sampled_frames", "full_temporal"}:
        raise HTTPException(status_code=400, detail="video_pipeline must be sampled_frames or full_temporal")
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required.")

    results: list[dict] = []
    errors: list[dict] = []

    for index, upload in enumerate(files):
        try:
            result = _run_inference_for_upload(
                upload=upload,
                query=query,
                max_video_frames=max_video_frames,
                video_pipeline=video_pipeline,
            )
            results.append(result)
        except Exception as ex:
            errors.append(
                {
                    "index": index,
                    "filename": upload.filename or "uploaded",
                    "error": str(ex),
                }
            )

    payload = {
        "model": service.config.model_name,
        "query": query,
        "video_pipeline": video_pipeline,
        "results": results,
        "errors": errors,
    }

    status = 207 if errors and results else 200
    if errors and not results:
        status = 400
    return JSONResponse(content=payload, status_code=status)
