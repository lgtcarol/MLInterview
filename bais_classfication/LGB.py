# -*- coding: utf-8 -*-
# __author__ = 'lgtcarol'


# 归一化处理，并增加了一些特征处理
import pandas as pd  # 数据分析
import numpy as np  # 科学计算
import matplotlib.pyplot as plt
import warnings

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore')
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

if __name__ == '__main__':

    data_train = pd.read_csv("bais_classfication/ads_train.csv")
    #data_test = pd.read_csv("D:/DC/alibaba/ads_test.csv")
    train_length = data_train.shape[0]
    print('原始数据分布：',train_length,(data_train[data_train['y_buy'] == 1].shape[0])) # 38209 172数据高度倾斜

    tmp = data_train.copy()
    tmp.drop(columns=['y_buy'], inplace=True)
    data_train_x = tmp.reset_index()
    data_train_y = data_train['y_buy'].reset_index()
    data_train_x.drop(columns=['index'], inplace=True)
    data_train_y.drop(columns=['index'], inplace=True)

    '''train集划分'''
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    skf.get_n_splits(data_train_x, data_train_y)
    for train_index, test_index in skf.split(data_train_x, data_train_y):
        X_train, X_test = data_train_x.iloc[train_index], data_train_x.iloc[test_index]  # 30568*21, 7641
        y_train, y_test = data_train_y.iloc[train_index], data_train_y.iloc[test_index]

    # 上采样
    def up_sample_data(df):
        data0 = df[df['y_buy'] == 0]  # 将多数类别的样本放在data0
        data1 = df[df['y_buy'] == 1]  # 将少数类别的样本放在data1
        index = np.random.randint(len(data1), size=(len(data0)-len(data1)))  # 随机给定上采样取出样本的序号
        up_data1 = data1.iloc[list(index)]  # 上采样
        return (pd.concat([up_data1, df]))
    Xy_train = pd.concat([X_train, y_train], axis=1)
    Xy_train = up_sample_data(Xy_train)

    train_length = Xy_train.shape[0]
    #print('上采样后数据分布：',train_length,(data_train[data_train['y_buy'] == 1].shape[0])) # 数据高度倾斜问题解决


    # 合并训练集与测试集
    true_label = pd.DataFrame(Xy_train['y_buy'])
    data = Xy_train.iloc[:, :-1]
    data = pd.concat([data, X_test])
    #data = pd.concat([data_train,data_test],ignore_index=True)
    # print(data.columns)
    data.drop(columns=['Unnamed: 0','last_visit'],inplace=True)
    data = data.fillna(0)



    def float_to_int(df,column_name):
        df[column_name] = df[column_name].round().astype(int)
        return df

    data = float_to_int(data,'buy_freq')
    data = float_to_int(data,'buy_interval')
    data = float_to_int(data,'sv_interval')
    data = float_to_int(data,'expected_time_buy')
    data = float_to_int(data,'expected_time_visit')
    data.info()

    # 将uniq_urls中的-1当异常值处理，变为均值
    def reset_uniq_urls(df):
        average = df['uniq_urls'][df['uniq_urls'] >= 0].mean()
        df['uniq_urls'][df['uniq_urls'] < 0] = average
        return df
    data = reset_uniq_urls(data)
    # data.info()

    def whole_time(df):
        df['whole_time_buy'] = df['buy_freq'] * df['buy_interval']
        df['whole_time_visit'] =  df['visit_freq'] * df['sv_interval']
        return df
    data = whole_time(data)

    def dev_of_lastandinterval(df):
        df['dev_lastandinterval1'] = df['last_buy'] - df['buy_interval']
        df['dev_lastandinterval2'] = df['last_buy'] - df['sv_interval']
        return df
    data = dev_of_lastandinterval(data)

    # 分桶
    def getbins_buyorsv_interval(df,column_name):
        df.loc[df[column_name] > 150,column_name+'_label'] = 0
        df.loc[(df[column_name] > 120) & (df[column_name] <= 150),column_name+'_label'] = 1
        df.loc[(df[column_name] > 90) & (df[column_name] <= 120),column_name+'_label'] = 2
        df.loc[(df[column_name] > 60) & (df[column_name] <= 90),column_name+'_label'] = 3
        df.loc[(df[column_name] > 30) & (df[column_name] <= 60),column_name+'_label'] = 4
        df.loc[df[column_name] <= 30,column_name+'_label'] = 5
        return df

    data = getbins_buyorsv_interval(data, 'buy_interval')
    data = getbins_buyorsv_interval(data, 'sv_interval')


    def getbins_last_buyorvisit(df,column_name):
        df.loc[df[column_name] > 150,column_name+'_label'] = 0
        df.loc[(df[column_name] > 120) & (df[column_name] <= 150),column_name+'_label'] = 1
        df.loc[(df[column_name] > 90) & (df[column_name] <= 120),column_name+'_label'] = 2
        df.loc[(df[column_name] > 60) & (df[column_name] <= 90),column_name+'_label'] = 3
        df.loc[df[column_name] <= 60,column_name+'_label'] = 4
        return df
    data = getbins_last_buyorvisit(data,'last_buy')

    # 分桶
    def getbins_buyorvisit_freq(df,column_name):
        df.loc[df[column_name] >= 13,column_name+'_label'] = 0
        df.loc[(df[column_name] >= 11) & (df[column_name] < 13),column_name+'_label'] = 1
        df.loc[(df[column_name] >= 9) & (df[column_name] < 11),column_name+'_label'] = 2
        df.loc[(df[column_name] >= 7) & (df[column_name] < 9),column_name+'_label'] = 3
        df.loc[(df[column_name] >= 5) & (df[column_name] < 7),column_name+'_label'] = 4
        df.loc[(df[column_name] >= 3) & (df[column_name] < 5),column_name+'_label'] = 5
        df.loc[(df[column_name] >= 1) & (df[column_name] < 3),column_name+'_label'] = 6
        df.loc[df[column_name] < 1,column_name+'_label'] = 7
        return df
    data = getbins_buyorvisit_freq(data,'buy_freq')
    data = getbins_buyorvisit_freq(data, 'visit_freq')

    # 独热编码
    def encode_onehot(df, column_name):
        feature_df = pd.get_dummies(df[column_name], prefix=column_name)
        all = pd.concat([df.drop([column_name], axis=1), feature_df], axis=1)
        return all

    # data = encode_onehot(data,'isbuyer')
    data = encode_onehot(data,'buy_freq_label')
    data = encode_onehot(data,'visit_freq_label')
    # data = encode_onehot(data,'multiple_buy')
    # data = encode_onehot(data,'multiple_visit')
    data = encode_onehot(data,'buy_interval_label')
    data = encode_onehot(data,'sv_interval_label')
    data = encode_onehot(data,'last_buy_label')

    def urls_of_checkins(df):
        df['urls_of_checkins'] = df['uniq_urls'] / df['num_checkins']
        return df
    data = urls_of_checkins(data)

    import sklearn.preprocessing as preprocessing
    scaler = preprocessing.StandardScaler()

    dev_lastandinterval1_scale_param = scaler.fit(data['dev_lastandinterval1'].values.reshape(-1,1))
    data['dev_lastandinterval1_scaled'] = scaler.fit_transform(data['dev_lastandinterval1'].values.reshape(-1,1),dev_lastandinterval1_scale_param)

    dev_lastandinterval2_scale_param = scaler.fit(data['dev_lastandinterval2'].values.reshape(-1,1))
    data['dev_lastandinterval2_scaled'] = scaler.fit_transform(data['dev_lastandinterval2'].values.reshape(-1,1),dev_lastandinterval2_scale_param)

    uniq_urls_scale_param = scaler.fit(data['uniq_urls'].values.reshape(-1,1))
    data['uniq_urls_scaled'] = scaler.fit_transform(data['uniq_urls'].values.reshape(-1,1),uniq_urls_scale_param)

    num_checkins_scale_param = scaler.fit(data['num_checkins'].values.reshape(-1,1))
    data['num_checkins_scaled'] = scaler.fit_transform(data['num_checkins'].values.reshape(-1,1),num_checkins_scale_param)

    urls_of_checkins_scale_param = scaler.fit(data['urls_of_checkins'].values.reshape(-1,1))
    data['urls_of_checkins_scaled'] = scaler.fit_transform(data['urls_of_checkins'].values.reshape(-1,1),urls_of_checkins_scale_param)


    def average_nums(df):
        df['average_nums_url1'] = df['whole_time_buy'] / (df['uniq_urls']+1)
        df['average_nums_checkin1'] = df['whole_time_buy'] / (df['num_checkins'] + 1)
        df['average_nums_url2'] = df['whole_time_visit'] / (df['uniq_urls'] + 1)
        df['average_nums_checkin2'] = df['whole_time_visit'] / (df['num_checkins'] + 1)
        return df
    data = average_nums(data)


    whole_time_buy_scale_param = scaler.fit(data['whole_time_buy'].values.reshape(-1,1))
    data['whole_time_buy_scaled'] = scaler.fit_transform(data['whole_time_buy'].values.reshape(-1,1),whole_time_buy_scale_param)

    whole_time_visit_scale_param = scaler.fit(data['whole_time_visit'].values.reshape(-1,1))
    data['whole_time_visit_buy_scaled'] = scaler.fit_transform(data['whole_time_visit'].values.reshape(-1,1),whole_time_visit_scale_param)


    # data.drop(columns=['expected_time_buy','expected_time_visit','uniq_urls','num_checkins'],inplace=True)
    print('大量特征构建。。。。。。。。。。。。。。。')


    def get_new_features(df):
        df['buy_freq_count'] = df['buy_freq'].map(df['buy_freq'].value_counts()).astype(int)
        df['buy_freq_count'] = (df['buy_freq_count'] - df['buy_freq_count'].min()) / (
        df['buy_freq_count'].max() - df['buy_freq_count'].min())

        df['visit_freq_count'] = df['visit_freq'].map(df['visit_freq'].value_counts()).astype(int)
        df['visit_freq_count'] = (df['visit_freq_count'] - df['visit_freq_count'].min()) / (
        df['visit_freq_count'].max() - df['visit_freq_count'].min())

        for i in ['buy_freq','visit_freq','isbuyer','multiple_buy','multiple_visit']:
            for j in ['buy_interval','sv_interval','last_buy','expected_time_buy','expected_time_visit','uniq_urls','num_checkins']:
                temp = df.groupby(i, as_index=False)[j].agg({i + '_' + j + '_mean': 'mean', i + '_' + j + '_max': 'max', i + '_' + j + '_min': 'min', i + '_' + j + '_var': 'var', i + '_' + j + '_median': 'median'})
                df = pd.merge(df, temp, on=i, how='left')

        for i in ['buy_interval','sv_interval','last_buy','expected_time_buy','expected_time_visit','uniq_urls','num_checkins']:
            df[i + '-mean_buy_freq'] = df[i] - df['buy_freq_' + i + '_mean']

        for i in ['buy_interval','sv_interval','last_buy','expected_time_buy','expected_time_visit','uniq_urls','num_checkins']:
            df[i + '-mean_visit_freq'] = df[i] - df['visit_freq_' + i + '_mean']

        for i in ['buy_interval','sv_interval','last_buy','expected_time_buy','expected_time_visit','uniq_urls','num_checkins']:
            df[i + '-mean_isbuyer'] = df[i] - df['isbuyer_' + i + '_mean']

        for i in ['buy_interval','sv_interval','last_buy','expected_time_buy','expected_time_visit','uniq_urls','num_checkins']:
            df[i + '-mean_multiple_buy'] = df[i] - df['multiple_buy_' + i + '_mean']

        for i in ['buy_interval','sv_interval','last_buy','expected_time_buy','expected_time_visit','uniq_urls','num_checkins']:
            df[i + '-mean_multiple_visit'] = df[i] - df['multiple_visit_' + i + '_mean']

        return df

    data = get_new_features(data)
    print(len(data.columns))

    def count_corr(df):
        '''
        输出相关系数dataframe:col_1,col_2,cor(不包含同一特征且已去重复)
        '''
        # 计算列之间的相关系数，abs求绝对值，unstack变成堆叠形式的，sort_values降序排列，然后用reset_index重新赋值索引
        x1 = df.corr().abs().unstack().sort_values(ascending=False).reset_index()
        # 找到除了对角线之外的行
        x1 = x1.loc[x1.level_0 != x1.level_1]
        x2 = pd.DataFrame([sorted(i) for i in x1[['level_0','level_1']].values])
        x2['cor'] = x1[0].values
        x2.columns = ['col_1','col_2','cor']
        # 删除重复的行
        return x2.drop_duplicates()


    corr_col = count_corr(data)
    corr1_col = corr_col[corr_col.cor>0.9].col_2.values.tolist()
    corr_col = list(set(corr1_col))
    # print(corr_col)
    # 删除掉相关性大于0.9的列
    # for i in corr_col:
    data.drop(columns=corr_col, inplace=True)
    print(len(data.columns))

    from sklearn.preprocessing import Imputer, StandardScaler

    imp = Imputer(missing_values='NaN', strategy='mean', axis=0, verbose=0, copy=True)
    X = imp.fit_transform(data)

    X = X[:train_length]
    print(np.isnan(X).any())
    y = true_label.as_matrix()
    clf = RandomForestClassifier()
    clf.fit(X,y)

    feat_labels = data.columns
    importance = clf.feature_importances_
    imp_result = np.argsort(importance)[::-1]

    final_features = []
    for i in range(66):
        final_features.append(feat_labels[imp_result[i]])
        print("%2d. %-*s %f" % (i + 1, 30, feat_labels[imp_result[i]], importance[imp_result[i]]))
    data = data.loc[:, final_features]

    data = data.fillna(0)
    '''
    scoring = ['precision_macro', 'recall_macro','f1_macro']
    scores = cross_validate(clf,X,y,cv=5,scoring=scoring)
    # predictions = clf.predict(survived_df.as_matrix()[train_length:])
    print(scores['test_precision_macro'])
    print(scores['test_recall_macro'])
    print(scores['test_f1_macro'])
    '''

'''模型训练'''
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
data = ss.fit_transform(data)
data_train_x = data[:60860]
X_test = data[60860:]
#model_1
clf = LGBMClassifier()
#clf.fit(data_train_x, true_label)
#precision, recall, thresholds = precision_recall_curve(y_test, y0_pred)
# scoring = ['precision_macro', 'recall_macro', 'roc_auc']
# tmp_y = pd.concat([true_label, y_test])
# scores = cross_validate(clf, data, tmp_y, cv=5, scoring=scoring)
# # predictions = clf.predict(survived_df.as_matrix()[train_length:])
# print(scores['test_precision_macro'])
# print(scores['test_recall_macro'])
# print(scores['test_roc_auc'])

clf.fit(data_train_x, true_label)
y_pred = clf.predict(X_test)
f1_score(y_test, y_pred)
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred)
from sklearn.metrics import precision_score
precision_score(y_test, y_pred)
from sklearn.metrics import recall_score
recall_score(y_test, y_pred)




