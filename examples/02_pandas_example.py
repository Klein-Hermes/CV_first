"""
Pandas 入门示例

运行方式：
    python examples/02_pandas_example.py
"""

from pathlib import Path

import pandas as pd
import cv2
def main():
    # 统一把示例输出文件放到 examples/res 目录下。
    # 使用脚本自身路径来定位目录，可以避免“从不同工作目录运行时找不到路径”的问题。
    current_dir = Path(__file__).resolve().parent
    output_dir = current_dir / "res"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("===== 1. 创建 Series =====")

    # Series 可以理解成“带索引的一列数据”。
    # 很像 Excel 中的一列，或者数据库中的一个字段。
    s = pd.Series([88, 92, 75, 100], index=["张三", "李四", "王五", "赵六"])
    print(s)
    print("Series 的平均值:", s.mean())
    print("通过标签访问 '李四':", s["李四"])

    print("\n===== 2. 创建 DataFrame =====")

    # DataFrame 是 Pandas 最核心的数据结构。
    # 你可以把它理解成“带行列标签的二维表格”。
    data = {
        "姓名": ["张三", "李四", "王五", "赵六"],
        "年龄": [20, 21, 19, 22],
        "数学": [88, 92, 75, 100],
        "英语": [90, 85, 78, 95],
    }
    df = pd.DataFrame(data)
    print(df)

    print("\n===== 3. 查看基础信息 =====")
    print("前 2 行数据:")
    print(df.head(2))

    print("\n数据形状（行数, 列数）:", df.shape)
    print("\n列名:")
    print(df.columns)

    print("\n每列的数据类型:")
    print(df.dtypes)

    print("\n===== 4. 选择列和行 =====")

    # 取单列，返回的是 Series
    print("只取 '姓名' 列:")
    print(df["姓名"])

    # 取多列，返回的是 DataFrame
    print("\n只取 '姓名' 和 '数学' 两列:")
    print(df[["姓名", "数学"]])

    # loc 按“标签”选取；iloc 按“位置”选取。
    print("\n使用 loc 选第 0 行:")
    print(df.loc[0])

    print("\n使用 iloc 选前 2 行前 3 列:")
    print(df.iloc[:2, :3])

    print("\n===== 5. 新增列 =====")

    # 直接对新列赋值即可创建新列。
    # 这里计算总分和平均分。
    df["总分"] = df["数学"] + df["英语"]
    df["平均分"] = df["总分"] / 2
    print(df)

    print("\n===== 6. 条件筛选 =====")

    # 筛选数学成绩大于等于 90 的同学
    top_math = df[df["数学"] >= 90]
    print("数学 >= 90 的同学:")
    print(top_math)

    print("\n===== 7. 排序 =====")

    # 按总分从高到低排序
    sorted_df = df.sort_values(by="总分", ascending=False)
    print("按总分降序排序后:")
    print(sorted_df)

    print("\n===== 8. 分组统计 =====")

    # 新增一个“班级”字段，演示 groupby 的用法
    df["班级"] = ["一班", "一班", "二班", "二班"]
    print("加入班级后的数据:")
    print(df)

    # groupby 的意思是“按某个字段分组，再做统计”
    group_result = df.groupby("班级")[["数学", "英语", "总分"]].mean()
    print("\n按班级求平均分:")
    print(group_result)

    print("\n===== 9. 缺失值处理 =====")

    # 人工构造一个带缺失值的 DataFrame
    df_missing = pd.DataFrame(
        {
            "姓名": ["小明", "小红", "小刚"],
            "成绩": [95, None, 82],
        }
    )
    print("原始数据:")
    print(df_missing)

    print("\n判断哪些位置缺失:")
    print(df_missing.isna())

    # fillna 可以用指定值填充缺失值
    filled_df = df_missing.fillna(0)
    print("\n把缺失值填充为 0 后:")
    print(filled_df)

    print("\n===== 10. 导出 CSV（示例） =====")

    # 这里演示如何保存为 CSV 文件。
    # index=False 的意思是：保存时不写入默认行号。
    output_path = output_dir / "pandas_students_output.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"已导出 CSV 文件到: {output_path}")


if __name__ == "__main__":
    main()
