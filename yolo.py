from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
DEFAULT_MODEL = "yolov8n.pt"
OUTPUT_DIR = PROJECT_DIR / "yolo_results"


def find_image_in_project() -> Path | None:
    image_paths = [
        path
        for path in PROJECT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]

    if not image_paths:
        return None

    image_paths.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return image_paths[0]


def get_image_path() -> Path | None:
    if len(sys.argv) >= 2:
        image_path = Path(sys.argv[1])
        if not image_path.is_absolute():
            image_path = PROJECT_DIR / image_path
        return image_path

    return find_image_in_project()


def load_yolo_model():
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError:
        print("Missing package: ultralytics")
        print("Install it with:")
        print("  pip install ultralytics")
        return None

    return YOLO(DEFAULT_MODEL)


def main() -> None:
    image_path = get_image_path()

    if image_path is None:
        print("No image found in the project folder.")
        print("Put a .png/.jpg/.jpeg/.bmp/.webp file next to yolo.py, then run: python yolo.py")
        return

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return

    model = load_yolo_model()
    if model is None:
        return

    print(f"Using image: {image_path.name}")
    print(f"Using model: {DEFAULT_MODEL}")
    print("Running YOLO object detection...")

    results = model.predict(
        source=str(image_path),
        save=True,
        project=str(OUTPUT_DIR),
        name="predict",
        exist_ok=True,
    )

    result = results[0]
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        print("No objects detected.")
    else:
        print("Detected objects:")
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            confidence = float(box.conf[0])
            print(f"  {class_name}: {confidence:.0%}")

    print(f"Result image saved in: {OUTPUT_DIR / 'predict'}")


if __name__ == "__main__":
    main()
