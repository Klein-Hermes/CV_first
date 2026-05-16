"""
PyTorch 入门示例

运行方式：
    python examples/06_pytorch_example.py
"""

import torch


def main():
    print("===== 1. 张量 Tensor 基础 =====")

    # Tensor（张量）是 PyTorch 的核心数据结构。
    # 你可以把它理解成“带有自动求导能力的多维数组”。
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    print("张量 x:\n", x)
    print("x 的形状:", x.shape)
    print("x 的数据类型:", x.dtype)

    print("\n===== 2. 张量运算 =====")

    y = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    print("张量 y:\n", y)
    print("x + y:\n", x + y)
    print("x * y（逐元素乘法）:\n", x * y)

    # matmul 表示矩阵乘法
    print("x @ y（矩阵乘法）:\n", torch.matmul(x, y))

    print("\n===== 3. 自动求导 autograd =====")

    # requires_grad=True 表示告诉 PyTorch：
    # 后续要跟踪这个张量上的运算，以便之后自动求导。
    a = torch.tensor(2.0, requires_grad=True)
    b = a**2 + 3 * a + 1  # b = a^2 + 3a + 1

    # backward() 会从结果 b 反向传播，自动计算 db/da。
    b.backward()

    print("a 的值:", a.item())
    print("b 的值:", b.item())
    print("db/da 的值:", a.grad.item())  # 理论上导数是 2a + 3，所以这里是 7

    print("\n===== 4. 用 PyTorch 做一个线性回归 =====")

    # 这里我们手工构造一组简单数据，关系大致是 y = 2x + 1
    X = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y_true = torch.tensor([[3.0], [5.0], [7.0], [9.0], [11.0]])

    # 定义一个线性层，输入维度 1，输出维度 1。
    # 它内部会自动创建参数 weight 和 bias。
    model = torch.nn.Linear(in_features=1, out_features=1)

    # 定义损失函数：均方误差（MSE）
    criterion = torch.nn.MSELoss()

    # 定义优化器：随机梯度下降 SGD
    # lr 是学习率，决定每次参数更新步子有多大。
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    print("训练前参数:")
    print("weight =", model.weight.data.item())
    print("bias =", model.bias.data.item())

    # 开始训练 1000 轮
    for epoch in range(1000):
        # 1. 前向传播：把输入喂给模型，得到预测值
        y_pred = model(X)

        # 2. 计算损失：预测值和真实值差多远
        loss = criterion(y_pred, y_true)

        # 3. 清空上一轮的梯度
        optimizer.zero_grad()

        # 4. 反向传播：自动计算每个参数的梯度
        loss.backward()

        # 5. 更新参数：沿着让损失下降的方向调整 weight / bias
        optimizer.step()

        # 每 100 轮打印一次，方便观察训练过程
        if (epoch + 1) % 100 == 0:
            print(f"第 {epoch + 1} 轮，loss = {loss.item():.6f}")

    print("\n训练后参数:")
    print("weight =", model.weight.data.item())
    print("bias =", model.bias.data.item())

    print("\n===== 5. 用训练好的模型预测 =====")

    test_input = torch.tensor([[6.0]])
    test_output = model(test_input)
    print("输入:", test_input.item())
    print("预测输出:", test_output.item())


if __name__ == "__main__":
    main()
