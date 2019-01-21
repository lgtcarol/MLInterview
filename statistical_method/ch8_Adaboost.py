# -*- coding: utf-8 -*-
# __author__ = 'lgtcarol '
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# data
def create_data():
    iris = load_iris()  # lgt加载的结果是sklearn.utils.Bunch,iris.data是narray型
    # 将内置数据集转为dataframe格式
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0, 1, -1]])  # 选取前100行，第一二和倒数第一列
    for i in range(len(data)):
        if data[i, -1] == 0:  # 将label进行修改
            data[i, -1] = -1
    # print(data)
    return data[:, :2], data[:, -1]


X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

plt.scatter(X[:50, 0], X[:50, 1], label='0')  # 横纵坐标，前50个点标记为0
plt.scatter(X[50:, 0], X[50:, 1], label='1')
plt.legend()


class AdaBoost:
    def __init__(self, n_estimators=50, learning_rate=1.0):
        self.clf_num = n_estimators  # 期望训得的弱学习器数目
        self.learning_rate = learning_rate

    # 计算alpha lgt:弱分类器的权重计算公式
    def _alpha(self, error):
        return 0.5 * np.log((1 - error) / error)

    # 规范化因子
    def _Z(self, weights, a, clf):
        return sum([weights[i] * np.exp(-1 * a * self.Y[i] * clf[i]) for i in range(self.M)])

    # 权值更新 lgt:权值分布更新公式
    def _w(self, a, clf, Z):
        for i in range(self.M):
            self.weights[i] = self.weights[i] * np.exp(-1 * a * self.Y[i] * clf[i]) / Z

    # lgt:单属性分类的弱分类器训练
    def _G(self, features, labels, weights):  # features应为某属性向量
        m = len(features)
        error = 100000.0  # 无穷大
        best_v = 0.0
        # 单维features
        features_min = min(features)
        features_max = max(features)
        n_step = (features_max - features_min + self.learning_rate) // self.learning_rate
        # print('n_step:{}'.format(n_step))
        direct, compare_array = None, None
        for i in range(1, int(n_step)):
            v = features_min + self.learning_rate * i

            if v not in features:  #
                # 误分类计算
                compare_array_positive = np.array([1 if features[k] > v else -1 for k in range(m)])  # 按该阈值为正样本标注1
                weight_error_positive = sum(
                    [weights[k] for k in range(m) if compare_array_positive[k] != labels[k]])  # 统计正样本中标记出错的总数

                compare_array_nagetive = np.array([-1 if features[k] > v else 1 for k in range(m)])
                weight_error_nagetive = sum(
                    [weights[k] for k in range(m) if compare_array_nagetive[k] != labels[k]])  # 该阈值对应的负样本同上处理

                if weight_error_positive < weight_error_nagetive:
                    weight_error = weight_error_positive
                    _compare_array = compare_array_positive
                    direct = 'positive'
                else:
                    weight_error = weight_error_nagetive
                    _compare_array = compare_array_nagetive
                    direct = 'nagetive'

                # print('v:{} error:{}'.format(v, weight_error))
                if weight_error < error:  # 前向误差更小时，进行误差更新以及权值更新
                    error = weight_error
                    compare_array = _compare_array
                    best_v = v
        return best_v, direct, error, compare_array

    # lgt:分类器结果判断
    def G(self, x, v, direct):
        if direct == 'positive':
            return 1 if x > v else -1
        else:
            return -1 if x > v else 1

            # # G(x)的线性组合
            # def _f(self, alpha, clf_sets):
            #     pass

    def init_args(self, datasets, labels):
        self.X = datasets
        self.Y = labels
        self.M, self.N = datasets.shape  # M为样本个数，N为属性维度

        # 弱分类器数目和集合
        self.clf_sets = []

        # 初始化weights
        self.weights = [1.0 / self.M] * self.M  # M为样本个数

        # G(x)系数 alpha
        self.alpha = []

    def fit(self, X, y):
        self.init_args(X, y)
        # 对应Adaboost伪代码部分，最外层控制弱分类器个数，内层的系数和权值更新很明显，
        # 关键是117-131弱分类器和训练误差分析过程，其实就是函数_G实现原理和跟弱学习器具体衔接！
        for epoch in range(self.clf_num):
            best_clf_error, best_v, clf_result = 100000, None, None
            # 根据特征维度, 选择误差最小的
            for j in range(self.N):
                features = self.X[:, j]  # 取出训练集中某列特征
                # 分类阈值，分类误差，分类结果
                v, direct, error, compare_array = self._G(features, self.Y, self.weights)
                # lgt:取出一个属性计算该属性最好的阈值v,（>）对应的正负，错误率和该属性该阈值下的分类结果
                if error < best_clf_error:
                    best_clf_error = error
                    best_v = v
                    final_direct = direct
                    clf_result = compare_array
                    axis = j
                # lgt:若
                # print('epoch:{}/{} feature:{} error:{} v:{}'.format(epoch, self.clf_num, j, error, best_v))
                if best_clf_error == 0:
                    break

            # 计算G(x)系数a
            a = self._alpha(best_clf_error)
            self.alpha.append(a)  # 把每个弱分类器对应的系数存储到列表
            # 记录分类器
            self.clf_sets.append((axis, best_v, final_direct))  # 把每个弱分类器依据的属性，阈值，阈值正负关系进行存储
            # 规范化因子
            Z = self._Z(self.weights, a, clf_result)  # 目前权重，本轮分类器系数，当前模型？？（clf_result）
            # 权值更新
            self._w(a, clf_result, Z)

    # print('classifier:{}/{} error:{:.3f} v:{} direct:{} a:{:.5f}'.format(epoch+1, self.clf_num, error, best_v,
    # final_direct, a)) print('weight:{}'.format(self.weights)) print('\n')

    # lgt: 单属性预测（多个弱分类器加权结果进行预测，即集成学习分类器的预测）
    def predict(self, feature):
        result = 0.0
        for i in range(len(self.clf_sets)):
            axis, clf_v, direct = self.clf_sets[i]
            f_input = feature[axis]
            result += self.alpha[i] * self.G(f_input, clf_v, direct)  #
        # result lgt:每个样本按弱分类器个数（len(self.clf_sets)）进行累计判断，判断结果为result
        return 1 if result > 0 else -1

    # lgt:预测正确的样本个数所占比重
    def score(self, X_test, y_test):
        right_count = 0
        for i in range(len(X_test)):
            feature = X_test[i]
            if self.predict(feature) == y_test[i]:
                right_count += 1

        return right_count / len(X_test)


'''例8.1'''
X = np.arange(10).reshape(10, 1)
y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1])

clf = AdaBoost(n_estimators=3, learning_rate=0.5)
clf.fit(X, y)

X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

clf = AdaBoost(n_estimators=10, learning_rate=0.2)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)

# 100次结果
result = []
for i in range(1, 101):
    X, y = create_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    clf = AdaBoost(n_estimators=100, learning_rate=0.2)
    clf.fit(X_train, y_train)
    r = clf.score(X_test, y_test)
    # print('{}/100 score：{}'.format(i, r))
    result.append(r)

print("average score: %.3f" % (sum(result) / len(result)))
