#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: r.li
@license: Apache Licence 
@contact: r.li@bmi-tech.com
@site: 
@software: PyCharm
@file: VGG.py
@time: 18-11-13 下午11:33
@brief： implementation for VGG16
"""

from datetime import datetime
import tensorflow as tf
import math
import time
import tensorflow.contrib.layers as tcl
from tensorflow.contrib.framework import arg_scope


"""
net structs 
-------------------------------------------------- 
layer          | kh x kw, out, s | out size 
-------------------------------------------------- 
conv1_1        | 3x3, 64, 1      | 224x224x64 
conv1_2        | 3x3, 64, 1      | 224x224x64 
-------------------------------------------------- 
max_pool       | 2x2, 64,2       | 112x112x64
-------------------------------------------------- 
conv2_1        | 3x3, 128, 1     | 112x112x128
conv2_2        | 3x3, 128, 1     | 112x112x128
-------------------------------------------------- 
max_pool       | 2x2, 2          | 56x56x128
-------------------------------------------------- 
conv3_1        | 3x3, 256, 1     | 56x56x256 
conv3_2        | 3x3, 256, 1     | 56x56x256 
conv3_3        | 3x3, 256, 1     | 56x56x256 
-------------------------------------------------- 
max_pool       | 2x2, 256,2      | 28x28x256
-------------------------------------------------- 
conv4_1        | 3x3, 512, 1     | 28x28x512 
conv4_2        | 3x3, 512, 1     | 28x28x512 
conv4_3        | 3x3, 512, 1     | 28x28x512 
-------------------------------------------------- 
max_pool       | 2x2, 512,2      | 14x14x512
-------------------------------------------------- 
conv5_1        | 3x3, 512, 1     | 14x14x512 
conv5_2        | 3x3, 512, 1     | 14x14x512 
conv5_3        | 3x3, 512, 1     | 14x14x512 
-------------------------------------------------- 
max_pool       | 2x2, 512,2      | 7x7x512
-------------------------------------------------- 
fc6            | 4096            | 1x1x4096 
fc7            | 4096            | 1x1x4096 
fc8            | 1000            | 1x1x1000
Softmax        | Classifier      | 1x1x1000
--------------------------------------------------

"""

batch_size = 16
num_batches = 100


# --------------------------Method 0 --------------------------------------------
# 用来创建卷积层并把本层的参数存入参数列表
def conv_op(input_op, name, kh, kw, n_out, dh, dw, p):
    """
    define conv operator with tf.nn 
    :param input_op: 输入的tensor
    :param name: 该层的名称
    :param kh: 卷积层的高
    :param kw: 卷积层的宽
    :param n_out: 输出通道数
    :param dh: 步长的高
    :param dw: 步长的宽
    :param p: 参数列表
    :return: 
    """
    # 输入的通道数
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + "w", shape=[kh, kw, n_in, n_out], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='SAME')
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = tf.Variable(bias_init_val, trainable=True, name='b')
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name=scope)
        p += [kernel, biases]
        return activation


# 定义全连接层
def fc_op(input_op, name, n_out, p):
    """
    define full connect opterator with tf.nn 
    :param input_op: 输入的tensor
    :param name: 该层的名称
    :param n_out: 输出通道数
    :param p: 参数列表
    :return: 
    """
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + 'w', shape=[n_in, n_out], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float32), name='b')
        # tf.nn.relu_layer()用来对输入变量input_op与kernel做乘法并且加上偏置b
        activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope)
        p += [kernel, biases]
        return activation


# 定义最大池化层
def mpool_op(input_op, name, kh, kw, dh, dw):
    return tf.nn.max_pool(input_op, ksize=[1, kh, kw, 1], strides=[1, dh, dw, 1], padding='SAME', name=name)


# 定义网络结构 Method 0
def vgg16_op(input_op, keep_prob):
    p = []
    conv1_1 = conv_op(input_op, name='conv1_1', kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
    conv1_2 = conv_op(conv1_1, name='conv1_2', kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
    pool1 = mpool_op(conv1_2, name='pool1', kh=2, kw=2, dw=2, dh=2)

    conv2_1 = conv_op(pool1, name='conv2_1', kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
    conv2_2 = conv_op(conv2_1, name='conv2_2', kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
    pool2 = mpool_op(conv2_2, name='pool2', kh=2, kw=2, dw=2, dh=2)

    conv3_1 = conv_op(pool2, name='conv3_1', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    conv3_2 = conv_op(conv3_1, name='conv3_2', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    conv3_3 = conv_op(conv3_2, name='conv3_3', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    pool3 = mpool_op(conv3_3, name='pool3', kh=2, kw=2, dw=2, dh=2)

    conv4_1 = conv_op(pool3, name='conv4_1', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv4_2 = conv_op(conv4_1, name='conv4_2', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv4_3 = conv_op(conv4_2, name='conv4_3', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    pool4 = mpool_op(conv4_3, name='pool4', kh=2, kw=2, dw=2, dh=2)

    conv5_1 = conv_op(pool4, name='conv5_1', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv5_2 = conv_op(conv5_1, name='conv5_2', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv5_3 = conv_op(conv5_2, name='conv5_3', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    pool5 = mpool_op(conv5_3, name='pool5', kh=2, kw=2, dw=2, dh=2)

    shp = pool5.get_shape()
    print("pool5 shape ", shp)

    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    resh1 = tf.reshape(pool5, [-1, flattened_shape], name="resh1")

    fc6 = fc_op(resh1, name="fc6", n_out=4096, p=p)
    fc6_drop = tf.nn.dropout(fc6, keep_prob, name='fc6_drop')
    fc7 = fc_op(fc6_drop, name="fc7", n_out=4096, p=p)
    fc7_drop = tf.nn.dropout(fc7, keep_prob, name="fc7_drop")
    fc8 = fc_op(fc7_drop, name="fc8", n_out=1000, p=p)
    softmax = tf.nn.softmax(fc8)
    predictions = tf.argmax(softmax, 1)
    return predictions, softmax, fc8, p


# --------------------------Method 1 --------------------------------------------
class VGG1:
    """
    define with tf.layers
    """
    def __init__(self, resolution_inp=224, channel=3, name='vgg'):
        self.name = name
        self.channel = channel
        self.resolution_inp = resolution_inp

    def __call__(self, x, dropout=0.5, is_training=True):
        with tf.variable_scope(self.name) as scope:
            size = 64

            # conv1 (64, 3, 1) x 2
            se = self.vgg_block(x, 2, size, is_training=is_training)

            # conv2 (128, 3, 1) x 2
            se = self.vgg_block(se, 2, size * 2, is_training=is_training)

            # conv3 (256, 3, 1) x 3
            se = self.vgg_block(se, 3, size * 4, is_training=is_training)

            # conv4 (512, 3, 1) x 3
            se = self.vgg_block(se, 3, size * 8, is_training=is_training)

            # conv5 (512, 3, 1) x 3
            pool5 = self.vgg_block(se, 3, size * 8, is_training=is_training)

            # full connect
            pool5_flat = tcl.flatten(pool5)

            fc6 = tf.layers.dense(pool5_flat, 4096)
            fc6_drop = tcl.dropout(fc6, dropout, is_training=is_training)

            fc7 = tf.layers.dense(fc6_drop, 4096)
            fc7_drop = tcl.dropout(fc7, dropout, is_training=is_training)
            self.fc_out = tf.layers.dense(fc7_drop, 1000)

            # predict for classify
            softmax = tf.nn.softmax(self.fc_out)
            self.predictions = tf.argmax(softmax, 1)
            return self.predictions

    def vgg_block(self, x, num_convs, num_channels, scope=None, is_training=True):
        """
        define the basic repeat unit in vgg: n x (conv-relu-batchnorm)-maxpool
        :param x: tensor or numpy.array, input
        :param num_convs: int, number of conv-relu-batchnorm 
        :param num_channels: int, number of conv filters
        :param scope: name space or scope
        :param is_training: bool, is training or not
        :return: 
        """
        with tf.variable_scope(scope, "conv"):
            se = x
            # conv-relu-batchnorm group
            for i in range(num_convs):
                se = tf.layers.conv2d(se,
                                      filters=num_channels,
                                      kernel_size=3,
                                      padding="same",
                                      strides=1,
                                      activation=tf.nn.relu)
                se = tf.layers.batch_normalization(se,
                                                   training=is_training,
                                                   scale=True)

            se = tf.layers.max_pooling2d(se, 2, 2, padding="same")

        return se

    @property
    def trainable_vars(self):
        return [var for var in tf.trainable_variables() if self.name in var.name]


# --------------------------Method 2 --------------------------------------------
class VGG2:
    """
    define with tf.contrib.layers
    """
    def __init__(self, resolution_inp=224, channel=3, name='vgg'):
        self.name = name
        self.channel = channel
        self.resolution_inp = resolution_inp

    def __call__(self, x, dropout=0.5, is_training=True):
        with tf.variable_scope(self.name) as scope:
            size = 64

            # conv1 (64, 3, 1) x 2
            se = self.vgg_block(x, 2, size, is_training=is_training)

            # conv2 (128, 3, 1) x 2
            se = self.vgg_block(se, 2, size * 2, is_training=is_training)

            # conv3 (256, 3, 1) x 3
            se = self.vgg_block(se, 3, size * 4, is_training=is_training)

            # conv4 (512, 3, 1) x 3
            se = self.vgg_block(se, 3, size * 8, is_training=is_training)

            # conv5 (512, 3, 1) x 3
            pool5 = self.vgg_block(se, 3, size * 8, is_training=is_training)

            pool5_flat = tcl.flatten(pool5)

            fc6 = tf.layers.dense(pool5_flat, 4096)
            fc6_drop = tcl.dropout(fc6, dropout, is_training=is_training)
            print("dropout ", fc6, fc6_drop)

            fc7 = tf.layers.dense(fc6_drop, 4096)
            fc7_drop = tcl.dropout(fc7, dropout, is_training=is_training)
            self.fc_out = tf.layers.dense(fc7_drop, 1000)

            # predict for classify
            softmax = tf.nn.softmax(self.fc_out)
            self.predictions = tf.argmax(softmax, 1)
            return self.predictions

    def vgg_block(self, x, num_convs, num_channels, scope=None, is_training=True):
        """
        define the basic repeat unit in vgg: n x (conv-relu-batchnorm)-maxpool
        :param x: 
        :param num_convs: 
        :param num_channels: 
        :param scope: 
        :param is_training: 
        :return: 
        """
        with tf.variable_scope(scope, "conv"):
            with arg_scope([tcl.batch_norm], is_training=is_training, scale=True):
                with arg_scope([tcl.conv2d],
                               padding="SAME",
                               normalizer_fn=tcl.batch_norm,
                               activation_fn=tf.nn.relu, ):
                    se = x
                    for i in range(num_convs):
                        se = tcl.conv2d(se, num_outputs=num_channels, kernel_size=3, stride=1)
                    se = tf.layers.max_pooling2d(se, 2, 2, padding="same")

        print("layer ", self.name, "in ", x, "out ", se)

        return se

    @property
    def trainable_vars(self):
        return [var for var in tf.trainable_variables() if self.name in var.name]

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


# -------------------------- Demo and Test --------------------------------------------
def time_tensorflow_run(session, target, feed, info_string):
    """
    calculate time for each session run
    :param session: tf.Session
    :param target: opterator or tensor need to run with session
    :param feed: feed dict for session
    :param info_string: info message for print
    :return: 
    """
    num_steps_burn_in = 10  # 预热轮数
    total_duration = 0.0  # 总时间
    total_duration_squared = 0.0  # 总时间的平方和用以计算方差
    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target, feed_dict=feed)
        duration = time.time() - start_time

        if i >= num_steps_burn_in:  # 只考虑预热轮数之后的时间
            if not i % 10:
                print('[%s] step %d, duration = %.3f' % (datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration

    mn = total_duration / num_batches  # 平均每个batch的时间
    vr = total_duration_squared / num_batches - mn * mn  # 方差
    sd = math.sqrt(vr)  # 标准差
    print('[%s] %s across %d steps, %.3f +/- %.3f sec/batch' % (datetime.now(), info_string, num_batches, mn, sd))


# test demo
def run_benchmark():
    """
    main function for test or demo
    :return: 
    """
    with tf.Graph().as_default():
        image_size = 224  # 输入图像尺寸
        images = tf.Variable(tf.random_normal([batch_size, image_size, image_size, 3], dtype=tf.float32, stddev=1e-1))
        keep_prob = tf.placeholder(tf.float32)

        # method 0
        # prediction, softmax, fc8, p = vgg16_op(images, keep_prob)

        # method 1 and method 2
        # vgg16 = VGG1(resolution_inp=image_size, name="vgg16")
        vgg16 = VGG2(resolution_inp=image_size, name="vgg16")
        prediction = vgg16(images, 0.5, True)
        fc8 = vgg16.fc_out
        p = vgg16.trainable_vars

        for v in p:
            print(v)
        init = tf.global_variables_initializer()

        # for var in tf.global_variables():
        #     print("param ", var.name)
        sess = tf.Session()
        print("init...")
        sess.run(init)

        print("predict..")
        writer = tf.summary.FileWriter("./logs")
        writer.add_graph(sess.graph)
        time_tensorflow_run(sess, prediction, {keep_prob: 1.0}, "Forward")

        # 用以模拟训练的过程
        objective = tf.nn.l2_loss(fc8)  # 给一个loss
        grad = tf.gradients(objective, p)  # 相对于loss的 所有模型参数的梯度

        print('grad backword')
        time_tensorflow_run(sess, grad, {keep_prob: 0.5}, "Forward-backward")
        writer.close()

if __name__ == '__main__':
    run_benchmark()
