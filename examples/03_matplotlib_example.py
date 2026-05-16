"""
Matplotlib 入门示例

运行方式：
    python examples/03_matplotlib_example.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def main():
    print("开始绘图...")

    # 这里拿到“当前脚本所在的目录”。
    # 这样后面保存图片时，无论你是在哪个工作目录执行这个脚本，
    # 都会稳定地保存到当前 .py 文件旁边，不会因为相对路径不同而报错。
    current_dir = Path(__file__).resolve().parent
    output_dir = current_dir / "res"

    # 如果 res 目录还不存在，就自动创建它。
    # exist_ok=True 表示：如果目录已经存在，也不要报错。
    output_dir.mkdir(parents=True, exist_ok=True)

    # 这里准备一些示例数据。
    # x 表示横坐标，使用 linspace 在 0 到 2π 之间均匀取 100 个点。
    x = np.linspace(0, 2 * np.pi, 100)

    # 使用 NumPy 计算 sin 和 cos 的值，作为纵坐标。
    y_sin = np.sin(x)
    y_cos = np.cos(x)

    # figure 表示整张画布。
    # figsize=(10, 5) 表示图像宽 10 英寸，高 5 英寸。
    plt.figure(figsize=(10, 5))

    # plot 用来画折线图。
    # label 会在图例中显示。
    plt.plot(x, y_sin, label="sin(x)", color="blue", linewidth=2)
    plt.plot(x, y_cos, label="cos(x)", color="orange", linestyle="--")

    # title、xlabel、ylabel 分别设置标题、x 轴名称、y 轴名称。
    plt.title("Sine And Cosine Curves")
    plt.xlabel("x")
    plt.ylabel("y")

    # legend 用来显示图例，帮助区分不同曲线。
    plt.legend()

    # grid 用来显示网格线，阅读图形时会更方便。
    plt.grid(True, linestyle=":", alpha=0.6)

    # tight_layout 会自动调整边距，避免标题或标签被遮住。
    plt.tight_layout()

    # 保存图片到文件。
    output_path = output_dir / "matplotlib_line_plot.png"
    plt.savefig(output_path, dpi=150)
    print(f"折线图已保存到: {output_path}")

    # show 会弹出图形窗口。
    # 如果你在某些无图形界面的环境运行，看不到窗口也没关系，
    # 因为上面已经把图片保存到本地了。
    plt.show()

    print("\n继续绘制柱状图...")

    students = ["张三", "李四", "王五", "赵六"]
    scores = [88, 92, 75, 100]

    plt.figure(figsize=(8, 5))
    plt.bar(students, scores, color=["#4C78A8", "#F58518", "#54A24B", "#E45756"])
    plt.title("Student Scores")
    plt.xlabel("Student")
    plt.ylabel("Score")
    plt.ylim(0, 110)  # 设置 y 轴范围，便于观察

    # 给每个柱子标上具体分数
    for name, score in zip(students, scores):
        plt.text(name, score + 2, str(score), ha="center")

    plt.tight_layout()

    output_path_bar = output_dir / "matplotlib_bar_plot.png"
    plt.savefig(output_path_bar, dpi=150)
    print(f"柱状图已保存到: {output_path_bar}")

    plt.show()


if __name__ == "__main__":
    main()
