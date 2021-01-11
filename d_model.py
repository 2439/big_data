from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import numpy
import pandas
import sys
import matplotlib.pyplot as plt


class Data:
    def __init__(self, file):
        self.name = file
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
        # print(self.df.isnull().any())
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
        for line in self.df.columns:
            if "乙肝" in line:
                self.df[line] = self.df[line].fillna(0)
            else:
                self.df[line] = self.df[line].fillna(self.df[line].mean())
            self.df[line] = (self.df[line] - self.df[line].min()) / (self.df[line].max() - self.df[line].min())

    def to_csv(self):
        # 写入文件
        self.name = self.name.split('.')[0] + '_pre.csv'
        self.df.to_csv("d_train_pre.csv", encoding="gbk", index=False)


class Model:
    def __init__(self):
        # self.model = DecisionTreeClassifier(criterion='entropy', max_depth=5)
        # self.model = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0,
        #                                 fit_intercept=True, intercept_scaling=1, class_weight=None,
        #                                 random_state=None, solver='liblinear', max_iter=100, multi_class='ovr',
        #                                 verbose=0, warm_start=False, n_jobs=1)
        # self.model = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=4)   # 1.0944
        self.model = LinearRegression(fit_intercept=True, normalize=True, copy_X=True, n_jobs=4)   # 1.0936

    def train(self, train_x, train_y):
        self.model.fit(train_x, train_y.astype('str'))

    def predict(self, test_x):
        return self.model.predict(test_x).astype(float)


if __name__ == '__main__':
    # 读取数据位置
    if len(sys.argv) == 2:
        s = sys.argv[1]
    else:
        s = input("文件位置：")
    # s = 'd_train.csv'

    # 数据预处理
    data_train = Data('d_train.csv')
    data_train.data_pre()
    data_train.to_csv()

    # 模型训练
    df_train = data_train.df
    y_train = numpy.array(df_train['血糖'])
    del(df_train["血糖"])
    X_train = numpy.array(df_train)

    model = Model()
    model.train(X_train, y_train)

    # 模型训练
    data_test = Data(s)
    data_test.data_pre()
    df_test = data_test.df
    y_test = numpy.array(df_test['血糖'])
    del(df_test["血糖"])
    X_test = numpy.array(df_test)

    pre_y = model.predict(X_test)

    # 分析结果
    sum_result = 0
    for i in range(len(y_test)):
        sum_result += pow(pre_y[i]-y_test[i], 2)
    sum_result /= (2 * len(y_test))
    print(sum_result)

    # plt.figure(figsize=(10, 3))
    # plt.plot(y_test, label='y_test')
    # plt.plot(pre_y, label='pre_y')
    # plt.legend()
    # plt.show()
