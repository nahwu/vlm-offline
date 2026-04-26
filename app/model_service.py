import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

logger = logging.getLogger("vlm.model_service")

import cv2
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration


def _flash_attn_available() -> bool:
    """Return True only if flash-attn is importable (it is an optional extra)."""
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        return False


@dataclass
class ModelConfig:
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    max_new_tokens: int = 128
    image_min_pixels: int | None = 224 * 224
    image_max_pixels: int | None = 1280 * 28 * 28
    model_quantization: str = "none"
    model_compute_dtype: str = "float16"
    attn_implementation: str = "auto"
    default_max_video_frames: int = 8
    default_video_fps: float = 1.0


def _resolve_torch_dtype(name: str, device: str) -> torch.dtype:
    normalized = name.strip().lower()
    if normalized in {"auto", ""}:
        return torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported MODEL_COMPUTE_DTYPE: {name}")


class VLMService:
    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        self._model = None
        self._processor = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._has_offload = False

    def load(self) -> None:
        """Eagerly load the model and run a CUDA warmup pass at startup.

        The first forward pass through a PyTorch model on CUDA compiles and
        caches device kernels, adding 20-30 s of latency.  Running a tiny
        dummy inference here moves that cost to startup so the first real
        user request is not penalised.
        """
        self._ensure_model_loaded()
        self._warmup()

    def _ensure_model_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        torch_dtype = _resolve_torch_dtype(self.config.model_compute_dtype, self._device) if self._device == "cuda" else torch.float32
        quantization = self.config.model_quantization.strip().lower()
        if self._device != "cuda":
            quantization = "none"

        logger.info(
            "[model] download started model=%s device=%s dtype=%s",
            self.config.model_name,
            self._device,
            str(torch_dtype),
        )
        _dl_start = time.perf_counter()

        flash_attn_available = _flash_attn_available()
        requested_attention = self.config.attn_implementation.strip().lower()
        if requested_attention == "auto":
            attention_implementation = "flash_attention_2" if (self._device == "cuda" and flash_attn_available) else "sdpa"
        else:
            attention_implementation = requested_attention
        if attention_implementation == "flash_attention_2" and not flash_attn_available:
            logger.warning("[model] flash_attention_2 requested but flash-attn is not installed; falling back to sdpa")
            attention_implementation = "sdpa"

        quantization_config = None
        if quantization in {"nf4", "4bit"}:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
            )
        elif quantization in {"int8", "8bit"}:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        elif quantization not in {"none", "fp16", "bf16"}:
            raise ValueError(f"Unsupported MODEL_QUANTIZATION: {self.config.model_quantization}")

        logger.info(
            "[model] runtime torch=%s torch_cuda=%s cuda_available=%s attention=%s flash_attn_installed=%s quantization=%s compute_dtype=%s",
            torch.__version__,
            torch.version.cuda,
            torch.cuda.is_available(),
            attention_implementation,
            flash_attn_available,
            quantization,
            str(torch_dtype),
        )

        model_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": "auto",
            "attn_implementation": attention_implementation,
        }
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.config.model_name, **model_kwargs)
        if hasattr(self._model, "generation_config"):
            self._model.generation_config.do_sample = False
            self._model.generation_config.temperature = None
            self._model.generation_config.top_p = None

        processor_kwargs = {}
        if self.config.image_min_pixels is not None:
            processor_kwargs["min_pixels"] = self.config.image_min_pixels
        if self.config.image_max_pixels is not None:
            processor_kwargs["max_pixels"] = self.config.image_max_pixels

        self._processor = AutoProcessor.from_pretrained(self.config.model_name, **processor_kwargs)
        logger.info(
            "[model] processor image pixels min=%s max=%s",
            self.config.image_min_pixels,
            self.config.image_max_pixels,
        )

        _dl_elapsed = time.perf_counter() - _dl_start
        logger.info(
            "[model] download completed model=%s elapsed_s=%.1f",
            self.config.model_name,
            _dl_elapsed,
        )

        _actual_device = str(self._model.device) if hasattr(self._model, "device") else self._device
        _on_gpu = _actual_device.startswith("cuda")
        if _on_gpu:
            gpu_name = torch.cuda.get_device_name(torch.device(_actual_device))
            logger.info(
                "[model] started on GPU device=%s gpu_name=%s",
                _actual_device,
                gpu_name,
            )
            # Log per-device parameter distribution so CPU offloading is visible
            _device_counts: dict = {}
            for _p in self._model.parameters():
                _dev = str(_p.device)
                _device_counts[_dev] = _device_counts.get(_dev, 0) + _p.numel()
            for _dev, _count in sorted(_device_counts.items()):
                logger.info("[model] parameters device=%s count=%d (%.1fB)", _dev, _count, _count / 1e9)
            self._has_offload = any(not d.startswith("cuda") for d in _device_counts)
            if self._has_offload:
                logger.warning(
                    "[model] some layers are offloaded from CUDA - check available VRAM; inference will be slow"
                )
            # Enable cuDNN autotuner for fixed-shape ops
            torch.backends.cudnn.benchmark = True
        else:
            logger.warning(
                "[model] started on CPU device=%s — inference will be slow",
                _actual_device,
            )

        logger.info(
            "[model] ready model=%s device=%s",
            self.config.model_name,
            _actual_device,
        )

    def _warmup(self) -> None:
        """Pre-compile CUDA kernels using realistic image sizes.

        Qwen2.5-VL tiles images into visual patches.  A 64x64 dummy produces
        only ~4 patches and warms up different cuDNN kernel shapes than a real
        640x480 image (~300 patches).  We run two sizes to cover typical inputs.
        """
        if self._device == "cpu":
            return
        if self._has_offload:
            logger.warning("[model] warmup skipped because model layers are offloaded from CUDA")
            return
        logger.info("[model] warmup started — pre-compiling CUDA kernels with realistic sizes")
        _t = time.perf_counter()
        for _w, _h in [(448, 448), (640, 480)]:
            try:
                dummy = Image.new("RGB", (_w, _h), color=(128, 128, 128))
                self._generate(query="warmup", images=[dummy])
                logger.info("[model] warmup pass size=%dx%d elapsed_s=%.1f", _w, _h, time.perf_counter() - _t)
            except Exception:
                logger.warning("[model] warmup pass size=%dx%d failed (non-fatal)", _w, _h, exc_info=True)
        logger.info("[model] warmup completed total_elapsed_s=%.1f", time.perf_counter() - _t)

    @property
    def device(self) -> str:
        return self._device

    @property
    def runtime_info(self) -> dict:
        info = {
            "device": self._device,
            "torch_version": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "image_min_pixels": self.config.image_min_pixels,
            "image_max_pixels": self.config.image_max_pixels,
            "max_new_tokens": self.config.max_new_tokens,
            "model_quantization": self.config.model_quantization,
            "model_compute_dtype": self.config.model_compute_dtype,
            "attn_implementation": self.config.attn_implementation,
            "has_offload": self._has_offload,
        }
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
        return info

    def infer_from_image(self, image: Image.Image, query: str) -> str:
        self._ensure_model_loaded()
        return self._generate(query=query, images=[image.convert("RGB")])

    def infer_from_video(self, video_path: str, query: str, max_frames: int | None = None) -> str:
        self._ensure_model_loaded()
        return self.infer_from_video_with_pipeline(
            video_path=video_path,
            query=query,
            pipeline="sampled_frames",
            max_frames=max_frames,
        )

    def infer_from_video_with_pipeline(
        self,
        video_path: str,
        query: str,
        pipeline: str,
        max_frames: int | None = None,
    ) -> str:
        self._ensure_model_loaded()

        if pipeline == "full_temporal":
            return self._generate_from_video_temporal(video_path=video_path, query=query)

        frames = self._extract_video_frames(
            video_path=video_path,
            max_frames=max_frames or self.config.default_max_video_frames,
        )
        if not frames:
            raise ValueError("No frames could be read from the input video.")
        return self._generate(query=query, images=frames)

    def infer_from_file(
        self,
        file_path: str,
        content_type: str,
        query: str,
        video_pipeline: str,
        max_video_frames: int,
    ) -> str:
        path = Path(file_path)
        content_type = (content_type or "").lower()
        is_video = content_type.startswith("video/") or path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}
        is_image = content_type.startswith("image/") or path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

        media_type = "image" if is_image else "video" if is_video else "unknown"
        logger.info(
            "[inference] call received file=%s media=%s pipeline=%s query_len=%d",
            path.name,
            media_type,
            video_pipeline,
            len(query),
        )
        _inf_start = time.perf_counter()

        try:
            if is_image:
                with Image.open(path) as image:
                    result = self.infer_from_image(image=image, query=query)
            elif is_video:
                result = self.infer_from_video_with_pipeline(
                    video_path=str(path),
                    query=query,
                    pipeline=video_pipeline,
                    max_frames=max_video_frames,
                )
            else:
                raise ValueError("Unsupported file type. Upload an image or video.")
        except Exception:
            logger.exception(
                "[inference] call failed file=%s media=%s",
                path.name,
                media_type,
            )
            raise

        _inf_elapsed = time.perf_counter() - _inf_start
        logger.info(
            "[inference] call completed file=%s media=%s elapsed_s=%.2f answer_len=%d",
            path.name,
            media_type,
            _inf_elapsed,
            len(result),
        )
        return result

    def _generate(self, query: str, images: List[Image.Image]) -> str:
        if not query.strip():
            raise ValueError("Query text cannot be empty.")

        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": query})

        messages = [{"role": "user", "content": content}]
        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self._processor(
            text=[text],
            images=images,
            padding=True,
            return_tensors="pt",
        )

        inputs = {k: v.to(self._model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        _t_generate = time.perf_counter()
        with torch.inference_mode():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        _generate_s = time.perf_counter() - _t_generate

        _t_decode = time.perf_counter()
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        _decode_s = time.perf_counter() - _t_decode

        logger.debug(
            "[inference] timing generate_s=%.2f decode_s=%.2f tokens=%d",
            _generate_s,
            _decode_s,
            generated_ids_trimmed[0].shape[0] if generated_ids_trimmed else 0,
        )

        return output_text[0].strip()

    def _generate_from_video_temporal(self, video_path: str, query: str) -> str:
        if not query.strip():
            raise ValueError("Query text cannot be empty.")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "fps": self.config.default_video_fps,
                    },
                    {"type": "text", "text": query},
                ],
            }
        ]

        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = {k: v.to(self._model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.inference_mode():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]

        output_text = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return output_text[0].strip()

    @staticmethod
    def _extract_video_frames(video_path: str, max_frames: int) -> List[Image.Image]:
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise ValueError("Unable to open the input video file.")

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            capture.release()
            return []

        stride = max(total_frames // max_frames, 1)
        sampled: List[Image.Image] = []
        frame_index = 0

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if frame_index % stride == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                sampled.append(Image.fromarray(rgb))
                if len(sampled) >= max_frames:
                    break

            frame_index += 1

        capture.release()
        return sampled


def _optional_int_env(name: str, default: int | None) -> int | None:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    if not value or value.lower() in {"none", "null"}:
        return None
    return int(value)


def load_config_from_env() -> ModelConfig:
    return ModelConfig(
        model_name=os.getenv("MODEL_NAME", "Qwen/Qwen2.5-VL-7B-Instruct"),
        max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "128")),
        image_min_pixels=_optional_int_env("IMAGE_MIN_PIXELS", ModelConfig.image_min_pixels),
        image_max_pixels=_optional_int_env("IMAGE_MAX_PIXELS", ModelConfig.image_max_pixels),
        model_quantization=os.getenv("MODEL_QUANTIZATION", ModelConfig.model_quantization),
        model_compute_dtype=os.getenv("MODEL_COMPUTE_DTYPE", ModelConfig.model_compute_dtype),
        attn_implementation=os.getenv("ATTN_IMPLEMENTATION", ModelConfig.attn_implementation),
        default_max_video_frames=int(os.getenv("MAX_VIDEO_FRAMES", "8")),
        default_video_fps=float(os.getenv("VIDEO_FPS", "1.0")),
    )
