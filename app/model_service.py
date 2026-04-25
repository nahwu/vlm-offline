import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


@dataclass
class ModelConfig:
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    max_new_tokens: int = 256
    default_max_video_frames: int = 8
    default_video_fps: float = 1.0


class VLMService:
    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        self._model = None
        self._processor = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def _ensure_model_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        torch_dtype = torch.float16 if self._device == "cuda" else torch.float32

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.config.model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
        )
        self._processor = AutoProcessor.from_pretrained(self.config.model_name)

    @property
    def device(self) -> str:
        return self._device

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

        if is_image:
            with Image.open(path) as image:
                return self.infer_from_image(image=image, query=query)
        if is_video:
            return self.infer_from_video_with_pipeline(
                video_path=str(path),
                query=query,
                pipeline=video_pipeline,
                max_frames=max_video_frames,
            )
        raise ValueError("Unsupported file type. Upload an image or video.")

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

        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
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

        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
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


def load_config_from_env() -> ModelConfig:
    return ModelConfig(
        model_name=os.getenv("MODEL_NAME", "Qwen/Qwen2.5-VL-7B-Instruct"),
        max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "256")),
        default_max_video_frames=int(os.getenv("MAX_VIDEO_FRAMES", "8")),
        default_video_fps=float(os.getenv("VIDEO_FPS", "1.0")),
    )
