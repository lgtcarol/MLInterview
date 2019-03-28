# -*- coding: utf-8 -*-
# __author__ = 'lgtcarol'

import pandas as pd  # 数据分析
import numpy as np  # 科学计算
from sklearn.model_selection import StratifiedKFold


data_train = pd.read_csv("bais_classfication/ads_train.csv", index_col=0)
#data_test = pd.read_csv("bais_classfication/ads_test.csv", index_col=0)

'''特征工程'''
tmp = data_train.copy()
tmp.drop(columns=['y_buy'], inplace=True)
data_train_x = tmp.reset_index()
data_train_y = data_train['y_buy'].reset_index()
data_train_x.drop(columns=['index'], inplace=True)
data_train_y.drop(columns=['index'], inplace=True)
#data_train_x.drop(columns=['last_visit'], inplace=True)
#data_test.drop(columns=['last_visit'], inplace=True)
data_train_x = data_train_x.fillna(0)
#data_test = data_test.fillna(0)

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

def reset_uniq_urls(df):
    average = df['uniq_urls'][df['uniq_urls'] >= 0].mean()
    df['uniq_urls'][df['uniq_urls'] < 0] = average
    return df
data_train_x = reset_uniq_urls(data_train_x)

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
#print(data_train_x['visit_freq'][data_train_x['visit_freq'] == 5])

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

'''train集划分'''
skf = StratifiedKFold(n_splits=5,shuffle=True)
skf.get_n_splits(data_train_x, data_train_y)
for train_index,test_index in skf.split(data_train_x,data_train_y):
    X_train, X_test = data_train_x.iloc[train_index], data_train_x.iloc[test_index] #30568*21, 7641
    y_train, y_test = data_train_y.iloc[train_index], data_train_y.iloc[test_index]

'''A组上采样'''
def up_sample_data(df):
    data0 = df[df['y_buy'] == 0]  # 将多数类别的样本放在data0
    data1 = df[df['y_buy'] == 1]  # 将少数类别的样本放在data1
    index = np.random.randint(len(data1), size=(len(data0) - len(data1)))  # 随机给定上采样取出样本的序号
    up_data1 = data1.iloc[list(index)]  # 上采样
    return (pd.concat([up_data1, df]))

Xy_train = pd.concat([X_train, y_train], axis=1)
Xy_train = up_sample_data(Xy_train) #60860*22

data_train_x = Xy_train.iloc[:, :-1]
data_train_y = Xy_train.iloc[:, -1:]

'''逻辑回归'''
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, f1_score

ss = StandardScaler()
data_train_x = ss.fit_transform(data_train_x)
clf = LogisticRegression()
clf.fit(data_train_x, data_train_y)
y_pred = clf.predict_proba(X_test)
y0_pred = [x[0] for x in y_pred]
#precision, recall, thresholds = precision_recall_curve(y_test, y0_pred)
y_pred = clf.predict(X_test)
f1_score(y_test, y_pred)


