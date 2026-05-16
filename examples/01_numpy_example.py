"""
NumPy 入门示例

运行方式：
    python examples/01_numpy_example.py
"""

import numpy as np


# 为了让浮点数输出更整齐一些，这里设置 NumPy 的打印精度。
# suppress=True 的意思是：尽量不用科学计数法显示较小的数字。
np.set_printoptions(precision=2, suppress=True)


def main():
    print("===== 1. 创建数组 =====")

    # NumPy 的核心对象叫 ndarray（N 维数组）。
    # 和 Python 原生 list 相比，ndarray 更适合做数值计算，速度通常也更快。
    arr1 = np.array([1, 2, 3, 4])
    print("一维数组 arr1:", arr1)
    print("arr1 的类型:", type(arr1))
    print("arr1 的形状 shape:", arr1.shape)  # shape 表示每个维度有多少个元素
    print("arr1 的数据类型 dtype:", arr1.dtype)

    # 创建二维数组。可以把它理解成“表格”或“矩阵”。
    arr2 = np.array([[1, 2, 3], [4, 5, 6]])
    print("\n二维数组 arr2:\n", arr2)
    print("arr2 的形状:", arr2.shape)  # 2 行 3 列

    print("\n===== 2. 快速生成常用数组 =====")

    # zeros：创建全 0 数组
    zeros_arr = np.zeros((2, 3))
    print("全 0 数组:\n", zeros_arr)

    # ones：创建全 1 数组
    ones_arr = np.ones((2, 2))
    print("全 1 数组:\n", ones_arr)

    # arange：类似 Python 的 range，但返回的是 NumPy 数组
    range_arr = np.arange(0, 10, 2)  # 从 0 到 10（不含 10），步长为 2
    print("arange 生成的数组:", range_arr)

    # linspace：在指定区间内，平均生成若干个点
    line_arr = np.linspace(0, 1, 5)
    print("linspace 生成的数组:", line_arr)

    print("\n===== 3. 数组运算 =====")

    a = np.array([1, 2, 3])
    b = np.array([10, 20, 30])

    # NumPy 支持“逐元素运算”。
    # 也就是说，a + b 不是把两个数组拼起来，而是对应位置相加。
    print("a + b =", a + b)
    print("a - b =", a - b)
    print("a * b =", a * b)  # 对应位置相乘，不是矩阵乘法
    print("a / b =", a / b)

    # 数组和标量（单个数字）也可以直接运算。
    print("a * 10 =", a * 10)
    print("a + 5 =", a + 5)

    print("\n===== 4. 索引和切片 =====")

    arr = np.array([10, 20, 30, 40, 50])
    print("原数组:", arr)
    print("第 0 个元素:", arr[0])
    print("最后一个元素:", arr[-1])
    print("切片 arr[1:4]:", arr[1:4])  # 取下标 1、2、3

    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("\n二维数组 matrix:\n", matrix)
    print("第 2 行:", matrix[1])
    print("第 2 行第 3 列的元素:", matrix[1, 2])
    print("第 1 列:", matrix[:, 0])  # : 表示“这一维全取”
    print("前两行、前两列:\n", matrix[:2, :2])

    print("\n===== 5. 形状变换 =====")

    # reshape 可以重新组织数组形状，但元素总数必须一致。
    nums = np.arange(1, 13)
    print("原一维数组:", nums)

    reshaped = nums.reshape(3, 4)
    print("变成 3x4 矩阵后:\n", reshaped)

    # flatten 可以把多维数组拍平成一维。
    flat = reshaped.flatten()
    print("重新拍平成一维:", flat)

    print("\n===== 6. 统计计算 =====")

    scores = np.array([88, 92, 75, 100, 95])
    print("分数数组:", scores)
    print("平均值 mean:", np.mean(scores))
    print("总和 sum:", np.sum(scores))
    print("最大值 max:", np.max(scores))
    print("最小值 min:", np.min(scores))
    print("标准差 std:", np.std(scores))

    print("\n===== 7. 布尔筛选 =====")

    # 条件表达式会生成一个布尔数组（True / False）。
    # 再用这个布尔数组去筛选原数组，就能拿到符合条件的元素。
    big_scores = scores[scores >= 90]
    print("分数 >= 90 的元素:", big_scores)

    print("\n===== 8. 广播机制（Broadcasting） =====")

    # 广播是 NumPy 非常重要的概念。
    # 它允许不同形状但“兼容”的数组自动扩展后再运算。
    mat = np.array([[1, 2, 3], [4, 5, 6]])
    vec = np.array([10, 20, 30])
    print("矩阵 mat:\n", mat)
    print("向量 vec:", vec)

    # vec 会被自动“广播”为：
    # [[10, 20, 30],
    #  [10, 20, 30]]
    # 然后与 mat 逐元素相加。
    print("mat + vec 的结果:\n", mat + vec)


if __name__ == "__main__":
    main()
