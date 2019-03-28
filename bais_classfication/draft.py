# -*- coding: utf-8 -*-
# __author__ = 'lgtcarol'


#isbuyer影响：isbuyer为1的数据y_buy概率大
y_buy_0 = data_train.isbuyer[data_train.y_buy == 0].value_counts()
y_buy_1 = data_train.isbuyer[data_train.y_buy == 1].value_counts()
df = pd.DataFrame({'y_buy==1':y_buy_1, 'y_buy==0':y_buy_0})
df.plot(kind='bar')
plt.title("y_buy == 0 or 1 of whether isbuyer")
plt.xlabel("isbuyer or not")
plt.ylabel("number")
plt.show()

# buy_freq影响：buy_freq>=1的数据y_buy概率大
data_train = data_train.fillna(0)
y_buy_0 = data_train.buy_freq[data_train.y_buy == 0].value_counts()
y_buy_1 = data_train.buy_freq[data_train.y_buy == 1].value_counts()
df = pd.DataFrame({'y_buy == 1':y_buy_1, 'y_buy == 0':y_buy_0})
df.plot(kind='bar')
plt.title("y_buy == 0 or 1 of different buy_freq")
plt.xlabel("buy_freq")
plt.ylabel("number")
plt.show()

# visit_freq影响：visit_freq==0和visit_freq>=3的数据y_buy概率大，0可能是异常值
# y_buy_0 = data_train.visit_freq[data_train.y_buy == 0].value_counts()
# y_buy_1 = data_train.visit_freq[data_train.y_buy == 1].value_counts()
# df = pd.DataFrame({'y_buy == 1':y_buy_1, 'y_buy == 0':y_buy_0})
# df.plot(kind='bar')
# plt.title("y_buy == 0 or 1 of different visit_freq")
# plt.xlabel("visit_freq")
# plt.ylabel("number")
# plt.show()

# buy_interval影响：该特征大部分为零，对此特征有点迷惑
# print(data_train.buy_interval.value_counts())
# y_buy_0 = data_train.buy_interval[data_train.y_buy == 0].value_counts()
# y_buy_1 = data_train.buy_interval[data_train.y_buy == 1].value_counts()
# df = pd.DataFrame({'y_buy == 1':y_buy_1, 'y_buy == 0':y_buy_0})
# df.plot()
# plt.title("y_buy == 0 or 1 of different buy_interval")
# plt.xlabel("buy_interval")
# plt.ylabel("number")
# plt.show()

# sv_interval影响：同上
# y_buy_0 = data_train.sv_interval[data_train.y_buy == 0].value_counts()
# y_buy_1 = data_train.sv_interval[data_train.y_buy == 1].value_counts()
# df = pd.DataFrame({'y_buy == 1':y_buy_1, 'y_buy == 0':y_buy_0})
# df.plot()
# plt.title("y_buy == 0 or 1 of different sv_interval")
# plt.xlabel("sv_interval")
# plt.ylabel("number")
# plt.show()

# expected_time_buy影响：匿名特征
# print(data_train.expected_time_buy.describe())
# y_buy_0 = data_train.expected_time_buy[data_train.y_buy == 0].value_counts()
# y_buy_1 = data_train.expected_time_buy[data_train.y_buy == 1].value_counts()
# df = pd.DataFrame({'y_buy == 1':y_buy_1, 'y_buy == 0':y_buy_0})
# df.plot()
# plt.title("y_buy == 0 or 1 of different expected_time_buy")
# plt.xlabel("expected_time_buy")
# plt.ylabel("number")
# plt.show()

# expected_time_visit影响：匿名特征。与xpected_time_visit数据分布相似-180~90左右
# print(data_train.expected_time_visit.describe())
# y_buy_0 = data_train.expected_time_visit[data_train.y_buy == 0].value_counts()
# y_buy_1 = data_train.expected_time_visit[data_train.y_buy == 1].value_counts()
# df = pd.DataFrame({'y_buy == 1':y_buy_1, 'y_buy == 0':y_buy_0})
# df.plot(kind='bar')
# plt.title("y_buy == 0 or 1 of different expected_time_visit")
# plt.xlabel("expected_time_visit")
# plt.ylabel("number")
# plt.show()

# print(data_train[data_train['last_buy']!=data_train['last_visit']].shape[0])
# last_buy和last_visit数据相同

# last_buy影响：
# y_buy_0 = data_train.last_buy[data_train.y_buy == 0].value_counts()
# y_buy_1 = data_train.last_buy[data_train.y_buy == 1].value_counts()
# df = pd.DataFrame({'y_buy == 1':y_buy_1, 'y_buy == 0':y_buy_0})
# df.plot()
# plt.title("y_buy == 0 or 1 of different last_buy")
# plt.xlabel("last_buy")
# plt.ylabel("number")
# plt.show()

# multiple_buy影响：multiple_buy为1时y_buy==1的概率大
# y_buy_0 = data_train.multiple_buy[data_train.y_buy == 0].value_counts()
# y_buy_1 = data_train.multiple_buy[data_train.y_buy == 1].value_counts()
# df = pd.DataFrame({'y_buy == 1': y_buy_1, 'y_buy == 0': y_buy_0})
# df.plot(kind='bar')
# plt.title("y_buy == 0 or 1 of whether multiple_buy")
# plt.xlabel("multiple_buy or not")
# plt.ylabel("number")
# plt.show()

# multiple_visit影响：multiple_visit为1时y_buy==1的概率大
# y_buy_0 = data_train.multiple_visit[data_train.y_buy == 0].value_counts()
# y_buy_1 = data_train.multiple_visit[data_train.y_buy == 1].value_counts()
# df = pd.DataFrame({'y_buy == 1': y_buy_1, 'y_buy == 0': y_buy_0})
# df.plot(kind='bar')
# plt.title("y_buy == 0 or 1 of whether multiple_visit")
# plt.xlabel("multiple_visit or not")
# plt.ylabel("number")
# plt.show()

# uniq_urls影响：
# y_buy_0 = data_train.uniq_urls[data_train.y_buy == 0].value_counts()
# y_buy_1 = data_train.uniq_urls[data_train.y_buy == 1].value_counts()
# df = pd.DataFrame({'y_buy == 1': y_buy_1, 'y_buy == 0': y_buy_0})
# df.plot()
# plt.title("y_buy == 0 or 1 of different uniq_urls")
# plt.xlabel("different uniq_urls")
# plt.ylabel("number")
# plt.show()

# num_checkins影响：
# y_buy_0 = data_train.num_checkins[data_train.y_buy == 0].value_counts()
# y_buy_1 = data_train.num_checkins[data_train.y_buy == 1].value_counts()
# df = pd.DataFrame({'y_buy == 1': y_buy_1, 'y_buy == 0': y_buy_0})
# df.plot()
# plt.title("y_buy == 0 or 1 of different num_checkins")
# plt.xlabel("different num_checkins")
# plt.ylabel("number")
# plt.show()

data_train = up_sample_data(data_train)
data_train_0 = data_train[data_train.y_buy == 0]
data_train_1 = data_train[data_train.y_buy == 1]
train_length = data_train.shape[0]
print('上采样后数据分布：', train_length, (data_train[data_train['y_buy'] == 1].shape[0]))  # 数据高度倾斜问题解决

