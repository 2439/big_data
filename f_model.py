import numpy
import pandas
# import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import sys


class Data:
    def __init__(self, file):
        # 读取数据，获取表头
        f = open(file)
        self.df = pandas.read_csv(f)
        cols = self.df.columns.values
        # print(cols)

        # 数据转为二维数组和DataFrame格式
        df_array = []
        for i in range(len(self.df)):
            df_array.append(self.df.iloc[i].values)
        df_array = numpy.array(df_array)
        # print(df_array)

        df_frame = {}
        for i in range(len(cols)):
            df_frame[cols[i]] = df_array[:, i]
        self.df = pandas.DataFrame(df_frame)
        # print(self.df)

    def data_pre(self):
        # print(self.df.describe())
        # 重复值删除
        self.df.duplicated()

        # id无关因素，BMI分类代表身高和体重，因素冗余
        del(self.df["id"])
        del(self.df["身高"])
        del(self.df["孕前体重"])

        for line in self.df.columns:
            # SNP缺失值等0填充
            if "SNP" in line or line in ["孕次", "产次", "DM家族史", "ACEID"]:
                self.df[line] = self.df[line].fillna(0).astype(int)
                # print(self.df[line].value_counts())
            # 年龄身高等用平均数填充
            if line in ["年龄", "收缩压", "舒张压", "ALT", "AST"]:
                self.df[line] = self.df[line].fillna(self.df[line].mean()).astype(int)
            if line in ["孕前BMI", "分娩时", "糖筛孕周", "VAR00007", "wbc"]\
                    or line in self.df.columns[37:47]:
                self.df[line] = self.df[line].fillna(self.df[line].mean())
            # BMI分类用众数填充
            if line == "BMI分类":
                self.df[line] = self.df[line].fillna(self.df[line].mode()[0]).astype(int)

        # RBP40填充
        self.df["RBP4"] = self.df["RBP4"].fillna(0)

        # 年龄数据离散化，根据数据和折线图，最小17，最大48，分箱
        self.df["年龄"] = self.df["年龄"].astype(int)
        # print(self.df["年龄"].value_counts().sort_index())
        # plt.plot(self.df["年龄"].value_counts().sort_index())
        # plt.show()
        bins = [16, 25, 30, 35, 40, 50]
        self.df["年龄"] = pandas.cut(self.df["年龄"], bins, labels=False)

        # RBP4归一化
        for line in self.df.columns:
            if line in ["RBP4", "孕前BMI", "收缩压", "舒张压", "分娩时",
                        "糖筛孕周", "VAR00007", "wbc", "ALT", "AST"]\
                    or line in self.df.columns[37:47]:
                self.df[line] = (self.df[line] - self.df[line].min()) / (self.df[line].max() - self.df[line].min())

        # 写入文件
        self.df.to_csv("f_train_pre.csv", encoding="gbk", index=False)


class Model:
    def __init__(self):
        self.dtc_model = DecisionTreeClassifier(criterion='entropy', max_depth=5)

    def train(self, train_x, train_y):
        self.dtc_model.fit(train_x, train_y.astype('str'))

    def predict(self, test_x):
        return self.dtc_model.predict(test_x).astype(float)


if __name__ == '__main__':
    # print(sys.argv[1])
    # 读取数据位置
    if len(sys.argv) == 2:
        s = sys.argv[1]
    else:
        s = input("文件位置：")

    # 数据预处理
    data = Data(s)
    data.data_pre()

    df = pandas.read_csv(open("f_train_pre.csv"))
    # 数据划分
    df = numpy.array(df)
    df = df[:, 1:]  # id对结果无影响
    X = df[:, :df.shape[1] - 1]
    y = df[:, df.shape[1] - 1]

    X_train = X[:700]
    y_train = y[:700]
    X_test = X[300:]
    y_test = y[300:]

    # 模型训练
    model = Model()
    model.train(X_train, y_train)
    pre_y = model.predict(X_test)

    # 分析结果
    correct1 = 0
    for i in range(len(y_test)):
        if y_test[i] == 1 and y_test[i] == pre_y[i]:
            correct1 += 1
    correct_rate = correct1 / pre_y.sum()
    recall_rate = correct1 / y_test.sum()
    print((2 * correct_rate * recall_rate) / (correct_rate + recall_rate))
