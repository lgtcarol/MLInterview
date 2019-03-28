import pandas as pd  # 数据分析
import numpy as np  # 科学计算
import matplotlib.pyplot as plt
import warnings

from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.model_selection import cross_validate, cross_val_score

data_train = pd.read_csv("bais_classfication/ads_train.csv", index_col=0)
data_test = pd.read_csv("bais_classfication/ads_test.csv", index_col=0)
#train_length = data_train.shape[0]
#print('原始数据分布：', train_length, (data_train[data_train['y_buy'] == 1].shape[0]))  # 38209 172数据高度倾斜

'''预处理'''
# 上采样
def up_sample_data(df):
    data0 = df[df['y_buy'] == 0]  # 将多数类别的样本放在data0
    data1 = df[df['y_buy'] == 1]  # 将少数类别的样本放在data1

    index = np.random.randint(len(data1), size=(len(data0) - len(data1)))  # 随机给定上采样取出样本的序号
    up_data1 = data1.iloc[list(index)]  # 上采样

    return (pd.concat([up_data1, df]))

data_train = up_sample_data(data_train)
tmp = data_train.copy()
tmp.drop(columns=['y_buy'], inplace=True)
data_train_x = tmp.reset_index()
data_train_y = data_train['y_buy'].reset_index()
data_train_x.drop(columns=['index'], inplace=True)
data_train_y.drop(columns=['index'], inplace=True)

# data = pd.concat([data_train_x, data_test], ignore_index=True)
data_train_x.drop(columns=['last_visit'], inplace=True)
data_test.drop(columns=['last_visit'], inplace=True)
data_train_x = data_train_x.fillna(0)
data_test = data_test.fillna(0)

def float_to_int(df, column_name):
    df[column_name] = df[column_name].round().astype(int)
    return df
data_train_x = float_to_int(data_train_x, 'buy_freq')
data_train_x = float_to_int(data_train_x, 'buy_interval')
data_train_x = float_to_int(data_train_x, 'sv_interval')
data_train_x = float_to_int(data_train_x, 'expected_time_buy')
data_train_x = float_to_int(data_train_x, 'expected_time_visit')


# 将uniq_urls中的-1当异常值处理，变为均值
def reset_uniq_urls(df):
    average = df['uniq_urls'][df['uniq_urls'] >= 0].mean()
    df['uniq_urls'][df['uniq_urls'] < 0] = average
    return df

data_train_x = reset_uniq_urls(data_train_x)

# data.info()

def reset_uniq_urls(df):
    average = df['uniq_urls'][df['uniq_urls'] >= 0].mean()
    df['uniq_urls'][df['uniq_urls'] < 0] = average
    return df

data_train_x = reset_uniq_urls(data_train_x)


# data.info()

# 分桶
def getbins_buyorsv_interval(df, column_name):
    df[column_name][df[column_name] > 150] = 0
    df[column_name][(df[column_name] > 120) & (df[column_name] <= 150)] = 1
    df[column_name][(df[column_name] > 90) & (df[column_name] <= 120)] = 2
    df[column_name][(df[column_name] > 60) & (df[column_name] <= 90)] = 3
    df[column_name][(df[column_name] > 30) & (df[column_name] <= 60)] = 4
    df[column_name][df[column_name] <= 30] = 5
    return df

data_train_x = getbins_buyorsv_interval(data_train_x, 'buy_interval')
data_train_x = getbins_buyorsv_interval(data_train_x, 'sv_interval')


def getbins_last_buyorvisit(df, column_name):
    df[column_name][df[column_name] > 150] = 0
    df[column_name][(df[column_name] > 120) & (df[column_name] <= 150)] = 1
    df[column_name][(df[column_name] > 90) & (df[column_name] <= 120)] = 2
    df[column_name][(df[column_name] > 60) & (df[column_name] <= 90)] = 3
    df[column_name][df[column_name] <= 60] = 4
    return df


data_train_x = getbins_last_buyorvisit(data_train_x, 'last_buy')


# 分桶
def getbins_buyorvisit_freq(df, column_name):
    df[column_name][df[column_name] >= 13] = 0
    df[column_name][(df[column_name] >= 11) & (df[column_name] < 13)] = 1
    df[column_name][(df[column_name] >= 9) & (df[column_name] < 11)] = 2
    df[column_name][(df[column_name] >= 7) & (df[column_name] < 9)] = 3
    df[column_name][(df[column_name] >= 5) & (df[column_name] < 7)] = 4
    df[column_name][(df[column_name] >= 3) & (df[column_name] < 5)] = 5
    df[column_name][(df[column_name] >= 1) & (df[column_name] < 3)] = 6
    df[column_name][df[column_name] < 1] = 7
    return df


data_train_x = getbins_buyorvisit_freq(data_train_x, 'buy_freq')
data_train_x = getbins_buyorvisit_freq(data_train_x, 'visit_freq')
print(data_train_x['visit_freq'][data_train_x['visit_freq'] == 5])


# 独热编码
def encode_onehot(df, column_name):
    feature_df = pd.get_dummies(df[column_name], prefix=column_name)
    all = pd.concat([df.drop([column_name], axis=1), feature_df], axis=1)
    return all


data_train_x = encode_onehot(data_train_x, 'isbuyer')
data_train_x = encode_onehot(data_train_x, 'buy_freq')
data_train_x = encode_onehot(data_train_x, 'visit_freq')
data_train_x = encode_onehot(data_train_x, 'multiple_buy')
data_train_x = encode_onehot(data_train_x, 'multiple_visit')
data_train_x = encode_onehot(data_train_x, 'buy_interval')
data_train_x = encode_onehot(data_train_x, 'sv_interval')
data_train_x = encode_onehot(data_train_x, 'last_buy')


def urls_of_checkins(df):
    df['urls_of_checkins'] = df['uniq_urls'] / df['num_checkins']
    return df

data_train_x = urls_of_checkins(data_train_x)

''' 随机森林 '''
clf = DecisionTreeClassifier()
# X_test = ss.transform(X_test)
scoring = ['precision_macro', 'recall_macro', 'roc_auc']
scores = cross_validate(clf, data_train_x, data_train_y, cv=5, scoring=scoring)
# predictions = clf.predict(survived_df.as_matrix()[train_length:])
print(scores['test_precision_macro'])
print(scores['test_recall_macro'])
print(scores['test_roc_auc'])

'''逻辑回归'''
#3 7分
from sklearn.linear_model import LogisticRegression
ss = StandardScaler()
data_train_x = ss.fit_transform(data_train_x)
clf = LogisticRegression()
data_train_xtrain = data_train_x[:53251]
data_train_ytrain = data_train_y[:53251]
data_train_xval = data_train_x[53251:]
data_train_yval = data_train_y[53251:]
clf.fit(data_train_xtrain, data_train_ytrain)
pred_y = clf.predict_proba(data_train_xval)
pred_y0 = [x[0] for x in pred_y]
# tmp = clf.predict(data_test)
precision, recall, thresholds = precision_recall_curve(data_train_yval, pred_y0)
# 交叉验证
scoring = ['precision_macro', 'recall_macro', 'roc_auc']
scores = cross_validate(clf, data_train_x, data_train_y, cv=5, scoring=scoring)
# predictions = clf.predict(survived_df.as_matrix()[train_length:])
print(scores['test_precision_macro'])
print(scores['test_recall_macro'])
print(scores['test_roc_auc'])
# lgt
import sklearn.preprocessing
from sklearn.linear_model import LogisticRegression

ss = sklearn.preprocessing.StandardScaler()
X_train = ss.fit_transform(data_train_x)
X_test = ss.transform(data_test)

# 合并训练集与测试集

true_label = pd.DataFrame(data_train['y_buy'])
data_train.drop(columns=['y_buy'], inplace=True)
data = pd.concat([data_train, data_test], ignore_index=True)
# print(data.columns)
data.drop(columns=['Unnamed: 0', 'last_visit'], inplace=True)
data = data.fillna(0)


def float_to_int(df, column_name):
    df[column_name] = df[column_name].round().astype(int)
    return df


data = float_to_int(data, 'buy_freq')
data = float_to_int(data, 'buy_interval')
data = float_to_int(data, 'sv_interval')
data = float_to_int(data, 'expected_time_buy')
data = float_to_int(data, 'expected_time_visit')
data.info()


# 将uniq_urls中的-1当异常值处理，变为均值
def reset_uniq_urls(df):
    average = df['uniq_urls'][df['uniq_urls'] >= 0].mean()
    df['uniq_urls'][df['uniq_urls'] < 0] = average
    return df


data = reset_uniq_urls(data)
# data.info()

X = data.as_matrix()[:train_length]
y = true_label.as_matrix()
clf = DecisionTreeClassifier()
# clf.fit(X,y)
scoring = ['precision_macro', 'recall_macro', 'roc_auc']
scores = cross_validate(clf, X, y, cv=5, scoring=scoring)
# predictions = clf.predict(survived_df.as_matrix()[train_length:])
print(scores['test_precision_macro'])
print(scores['test_recall_macro'])
print(scores['test_roc_auc'])


# scores = cross_val_score(clf,X,y,cv=5,scoring='roc_auc')
# print(scores)


# 独热编码
def encode_onehot(df, column_name):
    feature_df = pd.get_dummies(df[column_name], prefix=column_name)
    all = pd.concat([df.drop([column_name], axis=1), feature_df], axis=1)
    return all


# 标签编码LabelEncoder对不连续的数字进行编号
def encode_count(df, column_name):
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(df[column_name].values))
    df[column_name] = lbl.transform(list(df[column_name].values))
    return df


def merge_count(df, columns, value, cname):
    add = pd.DataFrame(df.groupby(columns)[value].count()).reset_index()
    add.columns = columns + [cname]
    df = df.merge(add, on=columns, how="left")
    return df


def feat_count(df, df_feature, fe, value, name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].count()).reset_index()
    if not name:
        df_count.columns = fe + [value + "_%s_count" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df
