"""
scikit-learn 入门示例

运行方式：
    python examples/05_sklearn_example.py
"""

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def main():
    print("===== 1. 加载数据集 =====")

    # Iris（鸢尾花）数据集是机器学习入门非常经典的数据集。
    # 它包含 150 条样本，每条样本有 4 个特征：
    # 花萼长度、花萼宽度、花瓣长度、花瓣宽度。
    iris = load_iris()

    # X 表示特征（输入）
    # y 表示标签（输出，也就是类别）
    X = iris.data
    y = iris.target

    print("特征名称:", iris.feature_names)
    print("类别名称:", iris.target_names)
    print("X 的形状:", X.shape)
    print("y 的形状:", y.shape)

    print("\n===== 2. 划分训练集和测试集 =====")

    # train_test_split 用来把数据拆成两部分：
    # 训练集：给模型学习用
    # 测试集：评估模型效果用
    #
    # test_size=0.2 表示 20% 数据作为测试集
    # random_state=42 表示固定随机种子，方便复现结果
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("训练集大小:", X_train.shape[0])
    print("测试集大小:", X_test.shape[0])

    print("\n===== 3. 创建并训练模型 =====")

    # KNeighborsClassifier 是 K 近邻分类器。
    # 它的思路很直观：
    # 看一个新样本“离谁近”，再根据附近样本的类别投票决定它属于哪一类。
    model = KNeighborsClassifier(n_neighbors=3)

    # fit 就是训练模型。
    # 对 KNN 来说，它本质上是“记住训练数据”。
    model.fit(X_train, y_train)

    print("模型训练完成。")

    print("\n===== 4. 进行预测 =====")

    # predict 会对测试集给出预测类别
    y_pred = model.predict(X_test)
    print("前 10 个预测结果:", y_pred[:10])
    print("前 10 个真实结果:", y_test[:10])

    print("\n===== 5. 评估模型 =====")

    # accuracy_score 是最基础的分类指标：预测正确的比例
    acc = accuracy_score(y_test, y_pred)
    print("准确率 accuracy:", acc)

    # classification_report 会输出更详细的分类指标：
    # precision（精确率）、recall（召回率）、f1-score 等
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    print("===== 6. 预测新样本 =====")

    # 随便构造一个新样本，格式必须和训练数据一致：
    # [花萼长度, 花萼宽度, 花瓣长度, 花瓣宽度]
    new_sample = [[5.1, 3.5, 1.4, 0.2]]
    pred_class = model.predict(new_sample)[0]
    pred_name = iris.target_names[pred_class]

    print("新样本:", new_sample)
    print("预测类别编号:", pred_class)
    print("预测类别名称:", pred_name)


if __name__ == "__main__":
    main()
