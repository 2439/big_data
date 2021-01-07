from sklearn.tree import DecisionTreeClassifier
import numpy
import pandas
import sys
# import matplotlib.pyplot as plt


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

        # 删除id，id对结果无影响
        # del(self.df['id'])

        # 性别转为数值型
        sex = self.df["性别"].unique()
        sex_mapping = {sex[i]: i for i in range(sex.shape[0])}
        self.df['性别'] = self.df["性别"].map(sex_mapping)
        # print(self.df)

        # 年龄数据离散化，根据数据和折线图，最小3，最大93，分箱
        self.df["年龄"] = self.df["年龄"].astype(int)
        # print(self.df["年龄"].value_counts().sort_index())
        # plt.plot(self.df["年龄"].value_counts().sort_index())
        # plt.show()
        bins = [2, 20, 30, 40, 50, 60, 70, 94]
        self.df["年龄"] = pandas.cut(self.df["年龄"], bins, labels=False)

        # 日期转为距离2018.1.1的天数，体检日期为2017年
        self.df["体检日期"] = pandas.to_datetime(self.df['体检日期'], format="%d/%m/%Y")
        self.df["体检日期"] = pandas.to_datetime('1/1/2018', format="%d/%m/%Y") - self.df["体检日期"]
        self.df["体检日期"] = self.df["体检日期"].dt.days
        # print(self.df["体检日期"].sort_values())

        # 缺失值填充，除乙肝相关用0填充外，其余用平均值填充，归一化
        for line in self.df.iloc[range(len(self.df)), range(4, 41)]:
            if "乙肝" in line:
                self.df[line] = self.df[line].fillna(0)
            else:
                self.df[line] = self.df[line].fillna(self.df[line].mean())
            self.df[line] = (self.df[line] - self.df[line].min()) / (self.df[line].max() - self.df[line].min())

        # 写入文件
        self.df.to_csv("d_train_pre.csv", encoding="gbk", index=False)


class Model:
    def __init__(self):
        self.dtc_model = DecisionTreeClassifier(criterion='entropy', max_depth=5)

    def train(self, train_x, train_y):
        self.dtc_model.fit(train_x, train_y.astype('str'))

    def predict(self, test_x):
        return self.dtc_model.predict(test_x).astype(float)


if __name__ == '__main__':
    # 读取数据位置
    if len(sys.argv) == 2:
        s = sys.argv[1]
    else:
        s = input("文件位置：")

    # 数据预处理
    data = Data(s)
    data.data_pre()

    # 数据划分
    df = pandas.read_csv(open("d_train_pre.csv"))
    df = numpy.array(df)
    df = df[:, 1:]    # id对结果无影响
    X = df[:, :df.shape[1]-1]
    y = df[:, df.shape[1]-1]

    X_train = X[:5000]
    y_train = y[:5000]
    X_test = X[5000:]
    y_test = y[5000:]

    # 模型训练
    model = Model()
    model.train(X_train, y_train)
    pre_y = model.predict(X_test)

    # 分析结果
    sum_result = 0
    for i in range(len(y_test)):
        sum_result += pow(pre_y[i]-y_test[i], 2)
    sum_result /= (2 * len(y_test))
    print(sum_result)
