from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
MNIST_FILE = Path(__file__).with_name("mnist.npz")
TRAIN_SAMPLES = 10000
K_NEIGHBORS = 5


def ensure_mnist_dataset() -> Path:
    if not MNIST_FILE.exists():
        print("正在下载 MNIST 数据集...")
        urllib.request.urlretrieve(MNIST_URL, MNIST_FILE)
    return MNIST_FILE


def load_mnist() -> tuple[np.ndarray, np.ndarray]:
    dataset_path = ensure_mnist_dataset()
    with np.load(dataset_path) as data:
        x_train = data["x_train"][:TRAIN_SAMPLES].astype(np.float32) / 255.0
        y_train = data["y_train"][:TRAIN_SAMPLES]
    return x_train.reshape(TRAIN_SAMPLES, -1), y_train


def preprocess_image(image_path: str) -> np.ndarray:
    image = Image.open(image_path).convert("L")
    image = ImageOps.invert(image)
    image = ImageOps.fit(image, (28, 28), method=Image.Resampling.LANCZOS)

    image_array = np.asarray(image, dtype=np.float32) / 255.0

    # 白底黑字和黑底白字都尽量兼容。
    if image_array.mean() > 0.5:
        image_array = 1.0 - image_array

    return image_array.reshape(-1)


def predict_digit(
    image_vector: np.ndarray, train_images: np.ndarray, train_labels: np.ndarray
) -> tuple[int, float]:
    distances = np.sum((train_images - image_vector) ** 2, axis=1)
    nearest_indices = np.argsort(distances)[:K_NEIGHBORS]
    nearest_labels = train_labels[nearest_indices]

    counts = np.bincount(nearest_labels, minlength=10)
    prediction = int(np.argmax(counts))
    confidence = float(counts[prediction] / K_NEIGHBORS)
    return prediction, confidence


def main() -> None:
    if len(sys.argv) != 2:
        print("用法: python test.py 图片路径")
        print(r"示例: python test.py D:\images\digit.png")
        return

    image_path = sys.argv[1]
    if not Path(image_path).exists():
        print(f"图片不存在: {image_path}")
        return

    print("正在加载 MNIST 训练数据...")
    train_images, train_labels = load_mnist()

    print("正在识别图片中的手写数字...")
    image_vector = preprocess_image(image_path)
    prediction, confidence = predict_digit(image_vector, train_images, train_labels)

    print(f"预测结果: {prediction}")
    print(f"置信度(基于 {K_NEIGHBORS} 邻近样本投票): {confidence:.0%}")


if __name__ == "__main__":
    main()
