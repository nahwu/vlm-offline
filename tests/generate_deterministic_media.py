from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
IMAGE_PATH = ASSETS / "deterministic_sample.png"
VIDEO_PATH = ASSETS / "deterministic_sample.mp4"


def make_image() -> None:
    width, height = 640, 360
    image = Image.new("RGB", (width, height), (242, 243, 232))
    draw = ImageDraw.Draw(image)

    draw.rectangle((40, 40, 280, 180), fill=(18, 91, 80), outline=(0, 0, 0), width=3)
    draw.ellipse((330, 70, 550, 250), fill=(242, 191, 121), outline=(0, 0, 0), width=3)
    draw.text((48, 200), "Deterministic image", fill=(20, 20, 20))
    draw.text((48, 228), "Qwen2.5-VL test asset", fill=(20, 20, 20))

    image.save(IMAGE_PATH)


def make_video() -> None:
    width, height = 640, 360
    fps = 8
    frames = 24

    writer = cv2.VideoWriter(
        str(VIDEO_PATH),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    for i in range(frames):
        frame = np.full((height, width, 3), (220, 232, 239), dtype=np.uint8)

        x = 40 + i * 18
        y = 90 + int((i % 8) * 8)
        cv2.rectangle(frame, (x, y), (x + 120, y + 90), (20, 90, 80), -1)

        cv2.putText(
            frame,
            f"Frame {i:02d}",
            (30, 320),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (30, 30, 30),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)

    writer.release()


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    make_image()
    make_video()
    print(f"Generated: {IMAGE_PATH}")
    print(f"Generated: {VIDEO_PATH}")


if __name__ == "__main__":
    main()
