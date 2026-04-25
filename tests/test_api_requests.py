import argparse
from pathlib import Path

import requests


def send_query(
    base_url: str,
    file_path: Path,
    query: str,
    max_video_frames: int = 8,
    video_pipeline: str = "sampled_frames",
) -> dict:
    with file_path.open("rb") as f:
        response = requests.post(
            f"{base_url.rstrip('/')}/v1/query",
            data={
                "query": query,
                "max_video_frames": str(max_video_frames),
                "video_pipeline": video_pipeline,
            },
            files={"file": (file_path.name, f)},
            timeout=180,
        )

    response.raise_for_status()
    return response.json()


def send_batch_query(
    base_url: str,
    file_paths: list[Path],
    query: str,
    max_video_frames: int = 8,
    video_pipeline: str = "sampled_frames",
) -> dict:
    files = []
    opened = []
    try:
        for path in file_paths:
            handle = path.open("rb")
            opened.append(handle)
            files.append(("files", (path.name, handle)))

        response = requests.post(
            f"{base_url.rstrip('/')}/v1/query/batch",
            data={
                "query": query,
                "max_video_frames": str(max_video_frames),
                "video_pipeline": video_pipeline,
            },
            files=files,
            timeout=300,
        )
    finally:
        for handle in opened:
            handle.close()

    response.raise_for_status()
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="Send deterministic image/video REST test queries.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--query-image", default="Describe the main objects in this image.")
    parser.add_argument("--query-video", default="Summarize what changes across the video frames.")
    parser.add_argument("--video-pipeline", default="sampled_frames", choices=["sampled_frames", "full_temporal"])
    parser.add_argument("--run-batch", action="store_true")
    args = parser.parse_args()

    assets = Path(__file__).resolve().parent / "assets"
    image_path = assets / "deterministic_sample.png"
    video_path = assets / "deterministic_sample.mp4"

    if not image_path.exists() or not video_path.exists():
        raise FileNotFoundError(
            "Missing deterministic assets. Run: python tests/generate_deterministic_media.py"
        )

    print("Sending image query...")
    image_result = send_query(
        args.base_url,
        image_path,
        args.query_image,
        video_pipeline=args.video_pipeline,
    )
    print("Image answer:\n", image_result["answer"], "\n")

    print("Sending video query...")
    video_result = send_query(
        args.base_url,
        video_path,
        args.query_video,
        video_pipeline=args.video_pipeline,
    )
    print("Video answer:\n", video_result["answer"], "\n")

    if args.run_batch:
        print("Sending batch query (image + video)...")
        batch_result = send_batch_query(
            args.base_url,
            [image_path, video_path],
            "Answer each file with one concise summary.",
            video_pipeline=args.video_pipeline,
        )
        print("Batch result:\n", batch_result, "\n")


if __name__ == "__main__":
    main()
