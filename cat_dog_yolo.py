from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
DEFAULT_MODEL = "yolov8n.pt"
OUTPUT_DIR = PROJECT_DIR / "cat_dog_results"
TARGET_CLASSES = {"cat", "dog"}


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
        print(
            "Put a .png/.jpg/.jpeg/.bmp/.webp file next to cat_dog_yolo.py, "
            "then run: python cat_dog_yolo.py"
        )
        return

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return

    model = load_yolo_model()
    if model is None:
        return

    print(f"Using image: {image_path.name}")
    print(f"Using model: {DEFAULT_MODEL}")
    print("Running YOLO cat/dog classification...")

    results = model.predict(
        source=str(image_path),
        save=True,
        project=str(OUTPUT_DIR),
        name="predict",
        exist_ok=True,
    )

    result = results[0]
    boxes = result.boxes

    cat_dog_results = []
    if boxes is not None:
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            confidence = float(box.conf[0])

            if class_name in TARGET_CLASSES:
                cat_dog_results.append((class_name, confidence))

    if not cat_dog_results:
        print("No cat or dog detected in this image.")
    else:
        cat_dog_results.sort(key=lambda item: item[1], reverse=True)
        best_class, best_confidence = cat_dog_results[0]

        print(f"Classification result: {best_class}")
        print(f"Confidence: {best_confidence:.0%}")
        print("All cat/dog detections:")
        for class_name, confidence in cat_dog_results:
            print(f"  {class_name}: {confidence:.0%}")

    print(f"Result image saved in: {OUTPUT_DIR / 'predict'}")


if __name__ == "__main__":
    main()
