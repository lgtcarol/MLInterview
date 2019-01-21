# -*- coding: utf-8 -*-
# __author__ = 'lgtcarol'

# 因适用Tensorflow，框架问题还未解决故放弃
import os
import tensorflow as tf

# 类别不平衡之再缩放？？
# initialize variables/model parameters
W = tf.Variable(tf.zeros([5, 1]), name="weights")
b = tf.Variable(0., name="bias")