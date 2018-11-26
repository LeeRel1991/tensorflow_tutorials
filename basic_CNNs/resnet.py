#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: r.li
@license: Apache Licence 
@contact: r.li@bmi-tech.com
@site: 
@software: PyCharm
@file: resnet.py
@time: 18-11-21 上午10:14
@brief： 
"""
from datetime import datetime
import math
import time
import tensorflow as tf
import tensorflow.contrib.layers as tcl
from tensorflow.contrib.framework import arg_scope

"""
net structs 
see https://blog.csdn.net/liangyihuai/article/details/79140481
"""


# --------------------------Method 0 --------------------------------------------
# 用来创建卷积层并把本层的参数存入参数列表
def conv_op(input_op, n_out, kh, kw, hstride, wstride, name="conv"):
    """
    define conv operator with tf.nn 
    :param input_op: 输入的tensor
    :param name: 该层的名称
    :param kh: 卷积层的高
    :param kw: 卷积层的宽
    :param n_out: 输出通道数
    :param hstride: 步长的高
    :param wstride: 步长的宽
    :return: 
    """
    # 输入的通道数
    n_in = input_op.get_shape()[-1].value
    with tf.variable_scope(name) as scope:
        kernel = tf.get_variable("weight", shape=[kh, kw, n_in, n_out], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(input_op, kernel, (1, hstride, wstride, 1), padding='SAME')
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = tf.Variable(bias_init_val, trainable=True, name='bias')
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z)
        return activation


def identity_block(x, out_filters, kernel_size, stride=1, training=True, name="idblock"):
    """
    Implementation of the identity block as defined in Figure 3
    :param x: input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    :param kernel_size: integer, specifying the shape of the middle CONV's window for the main path
    :param out_filters: 
    :param training: train or test
    :return: add_result, output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis

    with tf.variable_scope(name):
        shortcut = x

        # first
        y = conv_op(x, out_filters, 1, 1, stride, stride, name="conv1")
        y = tf.layers.batch_normalization(y, axis=3, training=training)
        y = tf.nn.relu(y)

        # second
        y = conv_op(y, out_filters, kernel_size, kernel_size, stride, stride, name="conv2")
        y = tf.layers.batch_normalization(y, axis=3, training=training)
        y = tf.nn.relu(y)

        # third
        y = conv_op(y, out_filters, 1, 1, stride, stride, name="conv3")
        y = tf.layers.batch_normalization(y, axis=3, training=training)
        y = tf.nn.relu(y)

        # final step
        add = tf.add(y, shortcut)
        add_result = tf.nn.relu(add)

    return add_result


def residual_block(x, out_filters, kernel_size, stride, training=True, name="convblock"):
    with tf.variable_scope(name):
        small_ch = out_filters // 4

        # first
        y = conv_op(x, small_ch, 1, 1, stride, stride, name="conv1")
        y = tf.layers.batch_normalization(y, axis=1, training=training)
        y = tf.nn.relu(y)

        # second
        y = conv_op(y, small_ch, kernel_size, kernel_size, 1, 1, name="conv2")
        y = tf.layers.batch_normalization(y, axis=1, training=training)
        y = tf.nn.relu(y)

        # third
        y = conv_op(y, out_filters, 1, 1, 1, 1, name="conv3")
        y = tf.layers.batch_normalization(y, axis=1, training=training)
        y = tf.nn.relu(y)

        shortcut = conv_op(x, out_filters, 1, 1, stride, stride, "shortcut")
        shortcut = tf.layers.batch_normalization(shortcut, axis=3, training=training)

        # final step
        add = tf.add(y, shortcut)
        add_result = tf.nn.relu(add)

        return add_result


def resnet50(input, training=True, name="resnet50"):
    with tf.variable_scope(name):
        conv1 = conv_op(input, 64, 7, 7, 2, 2)
        conv1 = tf.layers.batch_normalization(conv1, axis=1, training=training)
        conv1 = tf.nn.relu(conv1)  # [n, 112, 112, 64
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')  # [n, 55,55,64]

        size = 128
        conv2 = residual_block(conv1, 256, 3, stride=1, name="conv2_1")  # [n, 28, 28, 256]
        conv2 = identity_block(conv2, 256, 3, stride=1, name="conv2_2")
        conv2 = identity_block(conv2, 256, 3, stride=1, name="conv2_3")

        conv3 = residual_block(conv2, 512, 3, stride=2, name="conv3_1")
        conv3 = identity_block(conv3, 512, 3, stride=1, name="conv3_2")
        conv3 = identity_block(conv3, 512, 3, stride=1, name="conv3_3")
        conv3 = identity_block(conv3, 512, 3, stride=1, name="conv3_4")

        conv4 = residual_block(conv3, 1024, 3, stride=2, name="conv4_1")
        conv4 = identity_block(conv4, 1024, 3, stride=1, name="conv4_2")
        conv4 = identity_block(conv4, 1024, 3, stride=1, name="conv4_3")
        conv4 = identity_block(conv4, 1024, 3, stride=1, name="conv4_4")
        conv4 = identity_block(conv4, 1024, 3, stride=1, name="conv4_5")
        conv4 = identity_block(conv4, 1024, 3, stride=1, name="conv4_6")

        conv5 = residual_block(conv4, 2048, 3, stride=2, name="conv5_1")
        conv5 = identity_block(conv5, 2048, 3, stride=1, name="conv5_2")
        conv5 = identity_block(conv5, 2048, 3, stride=1, name="conv5_3")

        avg_pool = tf.nn.avg_pool(conv5, [1, 7, 7, 1], strides=[1, 1, 1, 1], padding="VALID")
        flatten = tf.layers.flatten(avg_pool)

        fc6 = tf.layers.dense(flatten, units=1000, activation=tf.nn.relu)
        # dropout = tf.nn.dropout(fc6, keep_prob=0.5)
        predictions = tf.nn.softmax(fc6)

        print("conv1 ", conv1)
        print("conv2 ", conv2)
        print("conv3 ", conv3)
        print("conv4 ", conv4)
        print("conv5 ", conv5)
        print("avg_pool ", avg_pool)
        print("flatten ", flatten)
        print("fc6 ", fc6)
        # print("dropout ", dropout)
        print("predictions ", predictions)
        return predictions, fc6


# --------------------------Method 1 --------------------------------------------

class ResNet50:
    def __init__(self, resolution_inp=224, channel=3, name='resnet50'):
        self.name = name
        self.channel = channel
        self.resolution_inp = resolution_inp

    def __call__(self, x, dropout=0.5, is_training=True):
        with tf.variable_scope(self.name) as scope:
            with arg_scope([tcl.batch_norm], is_training=is_training, scale=True):
                with arg_scope([tcl.conv2d],
                               activation_fn=tf.nn.relu,
                               normalizer_fn=tcl.batch_norm,
                               padding="SAME"):
                    conv1 = tcl.conv2d(x, 64, 7, stride=2)
                    conv1 = tcl.max_pool2d(conv1, kernel_size=3, stride=2)

                    conv2 = self._res_blk(conv1, 256, 3, stride=1)
                    conv2 = self._res_blk(conv2, 256, 3, stride=1)
                    conv2 = self._res_blk(conv2, 256, 3, stride=1)

                    conv3 = self._res_blk(conv2, 512, 3, stride=2)
                    conv3 = self._res_blk(conv3, 512, 3, stride=1)
                    conv3 = self._res_blk(conv3, 512, 3, stride=1)
                    conv3 = self._res_blk(conv3, 512, 3, stride=1)

                    conv4 = self._res_blk(conv3, 1024, 3, stride=2)
                    conv4 = self._res_blk(conv4, 1024, 3, stride=1)
                    conv4 = self._res_blk(conv4, 1024, 3, stride=1)
                    conv4 = self._res_blk(conv4, 1024, 3, stride=1)
                    conv4 = self._res_blk(conv4, 1024, 3, stride=1)
                    conv4 = self._res_blk(conv4, 1024, 3, stride=1)

                    conv5 = self._res_blk(conv4, 2048, 3, stride=2)
                    conv5 = self._res_blk(conv5, 2048, 3, stride=1)
                    conv5 = self._res_blk(conv5, 2048, 3, stride=1)

                    avg_pool = tf.nn.avg_pool(conv5, [1, 7, 7, 1], strides=[1, 1, 1, 1], padding="VALID")
                    flatten = tf.layers.flatten(avg_pool)

                    self.fc6 = tf.layers.dense(flatten, units=1000, activation=tf.nn.relu)
                    # dropout = tf.nn.dropout(fc6, keep_prob=0.5)
                    predictions = tf.nn.softmax(self.fc6)
                    return predictions

    def _res_blk(self, x, num_outputs, kernel_size, stride=1, scope=None):
        with tf.variable_scope(scope, "resBlk"):
            small_ch = num_outputs // 4

            conv1 = tcl.conv2d(x, small_ch, kernel_size=1, stride=stride, padding="SAME")
            conv2 = tcl.conv2d(conv1, small_ch, kernel_size=kernel_size, stride=1, padding="SAME")
            conv3 = tcl.conv2d(conv2, num_outputs, kernel_size=1, stride=1, padding="SAME")

            shortcut = x
            if stride != 1 or x.get_shape()[-1] != num_outputs:
                shortcut = tcl.conv2d(x, num_outputs, kernel_size=1, stride=stride, padding="SAME",scope="shortcut")

            out = tf.add(conv3, shortcut)
            out = tf.nn.relu(out)
            return out


# -------------------------- Demo and Test --------------------------------------------
batch_size = 16
num_batches = 100


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

        # method 0
        # prediction, fc = resnet50(images, training=True)
        model = ResNet50(224, 3)
        prediction = model(images, is_training=True)
        fc = model.fc6

        params = tf.trainable_variables()

        for v in params:
            print(v)
        init = tf.global_variables_initializer()

        print("out shape ", prediction)
        sess = tf.Session()
        print("init...")
        sess.run(init)

        print("predict..")
        writer = tf.summary.FileWriter("./logs")
        writer.add_graph(sess.graph)
        time_tensorflow_run(sess, prediction, {}, "Forward")

        # 用以模拟训练的过程
        objective = tf.nn.l2_loss(fc)  # 给一个loss
        grad = tf.gradients(objective, params)  # 相对于loss的 所有模型参数的梯度

        print('grad backword')
        time_tensorflow_run(sess, grad, {}, "Forward-backward")
        writer.close()

if __name__ == '__main__':
    run_benchmark()
