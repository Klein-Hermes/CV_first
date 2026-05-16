"""
SciPy 入门示例

运行方式：
    python examples/04_scipy_example.py
"""

import numpy as np
from scipy import integrate, optimize, stats


def func_for_integral(x):
    # 这是一个待积分函数 f(x) = x^2
    return x**2


def func_for_minimize(x):
    # 这是一个待优化函数：(x - 3)^2 + 4
    # 当 x = 3 时，这个函数取得最小值 4。
    return (x - 3) ** 2 + 4


def main():
    print("===== 1. 数值积分 integrate =====")

    # quad 用来计算定积分。
    # 这里计算积分：∫(0 到 2) x^2 dx
    # 理论结果是 8/3，大约等于 2.6667。
    integral_result, error_estimate = integrate.quad(func_for_integral, 0, 2)
    print("积分结果:", integral_result)
    print("误差估计:", error_estimate)

    print("\n===== 2. 求最小值 optimize =====")

    # minimize_scalar 用来求单变量函数的最小值。
    # 返回结果里会包含最优点 x、最小值 fun 等信息。
    result = optimize.minimize_scalar(func_for_minimize)
    print("最优解对象:", result)
    print("最优点 x:", result.x)
    print("最小函数值:", result.fun)

    print("\n===== 3. 插值（简单演示） =====")

    # 插值的意思可以简单理解为：
    # 已知一组离散点，希望估计中间未知位置的值。
    from scipy.interpolate import interp1d

    x = np.array([0, 1, 2, 3])
    y = np.array([0, 2, 4, 6])

    # kind="linear" 表示线性插值
    f = interp1d(x, y, kind="linear")
    x_new = 1.5
    print(f"当 x = {x_new} 时，插值结果为:", float(f(x_new)))

    print("\n===== 4. 统计模块 stats =====")

    data = np.array([88, 92, 75, 100, 95, 90, 84])
    print("原始数据:", data)

    # 均值
    print("均值:", np.mean(data))

    # 中位数
    print("中位数:", np.median(data))

    # 众数
    mode_result = stats.mode(data, keepdims=False)
    print("众数:", mode_result.mode)
    print("众数出现次数:", mode_result.count)

    # 标准差
    print("标准差:", np.std(data))

    # zscore 可以把数据标准化成“均值为 0，标准差为 1”的形式
    z_scores = stats.zscore(data)
    print("标准化后的 z-score:", z_scores)

    print("\n===== 5. 正态分布概率 =====")

    # norm.cdf 表示累计分布函数。
    # 下面这句的意思是：在标准正态分布 N(0,1) 下，P(X <= 1.96) 的概率是多少。
    probability = stats.norm.cdf(1.96, loc=0, scale=1)
    print("标准正态分布下 P(X <= 1.96) =", probability)


if __name__ == "__main__":
    main()
