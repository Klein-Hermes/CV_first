from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


PROJECT_DIR = Path(__file__).resolve().parent
MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
MNIST_FILE = PROJECT_DIR / "mnist.npz"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
TRAIN_SAMPLES = 30000
K_NEIGHBORS = 7


def ensure_mnist_dataset() -> Path:
    if not MNIST_FILE.exists():
        print("Downloading MNIST dataset...")
        urllib.request.urlretrieve(MNIST_URL, MNIST_FILE)
    return MNIST_FILE


def load_mnist() -> tuple[np.ndarray, np.ndarray]:
    dataset_path = ensure_mnist_dataset()
    with np.load(dataset_path) as data:
        x_train = data["x_train"][:TRAIN_SAMPLES].astype(np.float32) / 255.0
        y_train = data["y_train"][:TRAIN_SAMPLES]
    return x_train.reshape(TRAIN_SAMPLES, -1), y_train


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


def preprocess_image(image_path: Path) -> np.ndarray:
    image = Image.open(image_path).convert("L")
    image_array = np.asarray(image, dtype=np.float32) / 255.0

    # Convert to the MNIST style: bright digit on dark background.
    if image_array.mean() > 0.5:
        image_array = 1.0 - image_array

    image_array[image_array < 0.18] = 0.0

    rows, cols = np.where(image_array > 0.05)
    if len(rows) == 0 or len(cols) == 0:
        return np.zeros(28 * 28, dtype=np.float32)

    top, bottom = rows.min(), rows.max() + 1
    left, right = cols.min(), cols.max() + 1
    digit = image_array[top:bottom, left:right]

    digit_image = Image.fromarray((digit * 255).astype(np.uint8))
    digit_image = digit_image.resize((20, 20), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(digit_image, (4, 4))

    centered = np.asarray(canvas, dtype=np.float32) / 255.0

    return centered.reshape(-1)


def predict_digit(
    image_vector: np.ndarray, train_images: np.ndarray, train_labels: np.ndarray
) -> tuple[int, float]:
    distances = np.sum((train_images - image_vector) ** 2, axis=1)
    nearest_indices = np.argsort(distances)[:K_NEIGHBORS]
    nearest_labels = train_labels[nearest_indices]

    nearest_distances = distances[nearest_indices]
    weights = 1.0 / (nearest_distances + 1e-6)
    scores = np.bincount(nearest_labels, weights=weights, minlength=10)
    prediction = int(np.argmax(scores))
    confidence = float(scores[prediction] / scores.sum())
    return prediction, confidence


def main() -> None:
    image_path = get_image_path()

    if image_path is None:
        print("No image found in the project folder.")
        print("Put a .png/.jpg/.jpeg/.bmp file next to test.py, then run: python test.py")
        return

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return

    print(f"Using image: {image_path.name}")
    print("Loading MNIST training data...")
    train_images, train_labels = load_mnist()

    print("Recognizing handwritten digit...")
    image_vector = preprocess_image(image_path)
    prediction, confidence = predict_digit(image_vector, train_images, train_labels)

    print(f"Predicted digit: {prediction}")
    print(f"Confidence from {K_NEIGHBORS}-nearest vote: {confidence:.0%}")


if __name__ == "__main__":
    main()
