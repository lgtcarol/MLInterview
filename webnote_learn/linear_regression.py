# -*- coding: utf-8 -*-
# __author__ = 'lgtcarol'

# 下面有标注的问题没解决

""" 生成数据 """
import matplotlib.pyplot as plt # import matplotlib
import numpy as np
import tensorflow as tf
import numpy as np
trX = np.linspace(-1, 1, 101)
trY = 2*trX + np.random.randn(*trX.shape)*0.4 + 0.2   # ??
plt.figure()
plt.scatter(trX,trY)
plt.plot(trX, .2 + 2 * trX)
plt.show()  # pycharm中绘图必加

""" 单变量示例 """
# h(X)= b + wX
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def model(X, w, b):
    return tf.multiply(X, w) + b

trX = np.linspace(-1, 1, 101).astype(np.float32)
trY = 2 * trX + np.random.randn(*trX.shape) * 0.33 + 10
w = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
cost = tf.reduce_mean(tf.square(trY-model(trX, w, b)))
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    for i in range(1000):
        sess.run(train_op)
    print ("w should be something around [2]: %d" % sess.run(w))
    print ("b should be something around [10]: %d" % sess.run(b))

    plt.plot(trX, trY, "ro", label="Orinal data")
    plt.plot(trX, w.eval()*trX + b.eval(), label="Fitted line")
    plt.legend()
    plt.show()

""" 多变量示例 """
# h(X)= B + WX
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def model(X, w, b):
    return tf.multiply(w, X) + b

trX = np.mgrid[-1:1:0.01, -10:10:0.1].reshape(2, -1).T  # ??
trW = np.array([3, 5])
trY = trW*trX + np.random.randn(*trX.shape) + [20, 100]

w = tf.Variable(np.array([1., 1.]).astype(np.float32))
b = tf.Variable(np.array([[1., 1.]]).astype(np.float32))
cost = tf.reduce_mean(tf.square(trY-model(trX, w, b)))
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)

with tf.Session() as sess:
    # you need to initialize variables (in this case just variable W)
    tf.initialize_all_variables().run()

    for i in range(1000):
        if i % 99 == 0:
            print ("Cost at step %d is: %lf" % (i, cost.eval()))
        sess.run(train_op)

    print ("w should be something around [3, 5]: %lf" % sess.run(w))
    print ("b should be something around [20,100]: %lf" % sess.run(b))

""" tensorflow 示例 """
# ??前两个不就是tensorflow示例吗，此例何意

""" sklearn示例 """
# ？？调用包不对
import tensorflow.contrib.learn.python.learn as learn
from sklearn import datasets, metrics, preprocessing

boston = datasets.load_boston()
x = preprocessing.StandardScaler().fit_transform(boston.data)
feature_columns = learn.infer_real_valued_columns_from_input(x)  # ??
regressor = learn.LinearRegressor(feature_columns=feature_columns)
regressor.fit(x, boston.target, steps=200, batch_size=32)
boston_predictions = list(regressor.predict(x, as_iterable=True))
score = metrics.mean_squared_error(boston_predictions, boston.target)
print("MSE: %f" % score)






















