"""
学习顺序建议和总入口文件

运行方式：
    python examples/07_all_in_one_note.py
"""


def main():
    print("建议学习顺序：")
    print("1. NumPy：先学数组、索引、广播、统计运算")
    print("2. Pandas：再学表格数据处理、筛选、分组、导出")
    print("3. Matplotlib：学会把数据画出来")
    print("4. SciPy：接触优化、积分、统计这些科学计算能力")
    print("5. scikit-learn：入门机器学习流程")
    print("6. PyTorch：入门深度学习和自动求导")

    print("\n对应示例文件：")
    print("examples/01_numpy_example.py")
    print("examples/02_pandas_example.py")
    print("examples/03_matplotlib_example.py")
    print("examples/04_scipy_example.py")
    print("examples/05_sklearn_example.py")
    print("examples/06_pytorch_example.py")

    print("\n如果某个库还没有安装，可以用下面的命令安装：")
    print("pip install numpy pandas matplotlib scipy scikit-learn torch")


if __name__ == "__main__":
    main()
