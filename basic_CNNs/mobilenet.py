#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: r.li
@license: Apache Licence 
@contact: r.li@bmi-tech.com
@site: 
@software: PyCharm
@file: mobilenet.py
@time: 18-11-13 下午11:02
@brief： 
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

"""
mobile net structs 
-------------------------------------------------- 
layer        | kh x kw, out, s | out size 
--------------------------------------------------
         input image (224 x 224 x3)
-------------------------------------------------- 
conv         | 3x3, 32, 2      | 112x112x32
-------------------------------------------------- 
conv_dw      | 3x3, 32dw, 1    | 112x112x32 
conv1x1      | 1x1, 64, 1      | 112x112x64
-------------------------------------------------- 
conv_dw      | 3x3, 64dw, 2    | 56x56x64 
conv1x1      | 1x1, 128, 1     | 56x56x128
-------------------------------------------------- 
conv_dw      | 3x3, 128dw, 1   | 56x56x128 
conv1x1      | 1x1, 128, 1     | 56x56x128 
-------------------------------------------------- 
conv_dw      | 3x3, 128dw, 2   | 28x28x128 
conv1x1      | 1x1, 256, 1     | 28x28x128
-------------------------------------------------- 
conv_dw      | 3x3, 256dw, 1   | 28x28x256 
conv1x1      | 1x1, 256, 1     | 28x28x256 
-------------------------------------------------- 
conv_dw      | 3x3, 256dw, 2   | 14x14x256 
conv1x1      | 1x1, 512, 1     | 14x14x512
-------------------------------------------------- 
5x 
conv_dw      | 3x3, 512dw, 1   | 14x14x512 
conv1x1      | 1x1, 512, 1     | 14x14x512 
-------------------------------------------------- 
conv_dw      | 3x3, 512dw, 2   | 7x7x512 
conv1x1      | 1x1, 1024, 1    | 7x7x1024 
-------------------------------------------------- 
conv_dw      | 3x3, 1024dw, 1  | 7x7x1024 
conv1x1      | 1x1, 1024, 1    | 7x7x1024 
-------------------------------------------------- 
Avg Pool      | 7x7, 1          | 1x1x1024 
FC           | 1024, 1000      | 1x1x1000 
Softmax      | Classifier      | 1x1x1000
--------------------------------------------------

"""


def mobilenet(inputs,
              num_classes=1000,
              is_training=True,
              width_multiplier=1,
              scope='MobileNet'):
    """ MobileNet
  More detail, please refer to Google's paper(https://arxiv.org/abs/1704.04861).
  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether or not the model is being trained.
    scope: Optional scope for the variables.
  Returns:
    logits: the pre-softmax activations, a tensor of size
      [batch_size, `num_classes`]
    end_points: a dictionary from components of the network to the corresponding
      activation.
  """

    def _depthwise_separable_conv(inputs,
                                  num_pwc_filters,
                                  width_multiplier,
                                  sc,
                                  downsample=False):
        """ Helper function to build the depth-wise separable convolution layer.
    """
        num_pwc_filters = round(num_pwc_filters * width_multiplier)
        _stride = 2 if downsample else 1

        # skip pointwise by setting num_outputs=None
        depthwise_conv = slim.separable_convolution2d(inputs,
                                                      num_outputs=None,
                                                      stride=_stride,
                                                      depth_multiplier=1,
                                                      kernel_size=[3, 3],
                                                      scope=sc + '/depthwise_conv')

        bn = slim.batch_norm(depthwise_conv, scope=sc + '/dw_batch_norm')
        pointwise_conv = slim.convolution2d(bn,
                                            num_pwc_filters,
                                            kernel_size=[1, 1],
                                            scope=sc + '/pointwise_conv')
        bn = slim.batch_norm(pointwise_conv, scope=sc + '/pw_batch_norm')
        return bn

    with tf.variable_scope(scope) as sc:
        end_points_collection = sc.name + '_end_points'
        with slim.arg_scope([slim.convolution2d, slim.separable_convolution2d],
                            activation_fn=None,
                            outputs_collections=[end_points_collection]):
            with slim.arg_scope([slim.batch_norm],
                                is_training=is_training,
                                activation_fn=tf.nn.relu,
                                fused=True):
                net = slim.convolution2d(inputs, round(32 * width_multiplier), [3, 3], stride=2, padding='SAME',
                                         scope='conv_1')
                net = slim.batch_norm(net, scope='conv_1/batch_norm')
                net = _depthwise_separable_conv(net, 64, width_multiplier, sc='conv_ds_2')
                net = _depthwise_separable_conv(net, 128, width_multiplier, downsample=True, sc='conv_ds_3')
                net = _depthwise_separable_conv(net, 128, width_multiplier, sc='conv_ds_4')
                net = _depthwise_separable_conv(net, 256, width_multiplier, downsample=True, sc='conv_ds_5')
                net = _depthwise_separable_conv(net, 256, width_multiplier, sc='conv_ds_6')
                net = _depthwise_separable_conv(net, 512, width_multiplier, downsample=True, sc='conv_ds_7')

                net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_8')
                net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_9')
                net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_10')
                net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_11')
                net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_12')

                net = _depthwise_separable_conv(net, 1024, width_multiplier, downsample=True, sc='conv_ds_13')
                net = _depthwise_separable_conv(net, 1024, width_multiplier, sc='conv_ds_14')
                net = slim.avg_pool2d(net, [7, 7], scope='avg_pool_15')

        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
        end_points['squeeze'] = net
        logits = slim.fully_connected(net, num_classes, activation_fn=None, scope='fc_16')
        predictions = slim.softmax(logits, scope='Predictions')

        end_points['Logits'] = logits
        end_points['Predictions'] = predictions

    return logits, end_points


mobilenet.default_image_size = 224


def mobilenet_arg_scope(weight_decay=0.0):
    """Defines the default mobilenet argument scope.
  Args:
    weight_decay: The weight decay to use for regularizing the model.
  Returns:
    An `arg_scope` to use for the MobileNet model.
  """
    with slim.arg_scope(
            [slim.convolution2d, slim.separable_convolution2d],
            weights_initializer=slim.initializers.xavier_initializer(),
            biases_initializer=slim.init_ops.zeros_initializer(),
            weights_regularizer=slim.l2_regularizer(weight_decay)) as sc:
        return sc


# --------------------------Method 1 --------------------------------------------
import tensorflow.contrib.layers as tcl
from tensorflow.contrib.framework import arg_scope


class Mobilenet:
    def __init__(self, resolution_inp=224, channel=3, name='resnet50'):
        self.name = name
        self.channel = channel
        self.resolution_inp = resolution_inp

    def _depthwise_separable_conv(self, x, num_outputs, kernel_size=3, stride=1, scope=None):
        with tf.variable_scope(scope, "dw_blk"):
            dw_conv = tcl.separable_conv2d(x, num_outputs=None,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           depth_multiplier=1)
            conv_1x1 = tcl.conv2d(dw_conv, num_outputs=num_outputs, kernel_size=1, stride=1)
            return conv_1x1

    def __call__(self, x, dropout=0.5, is_training=True):
        with tf.variable_scope(self.name) as scope:
            with arg_scope([tcl.batch_norm], is_training=is_training, scale=True):
                with arg_scope([tcl.conv2d, tcl.separable_conv2d],
                               activation_fn=tf.nn.relu,
                               normalizer_fn=tcl.batch_norm,
                               padding="SAME"):
                    conv1 = tcl.conv2d(x, 32, kernel_size=3, stride=2)

                    y = self._depthwise_separable_conv(conv1, 64, 3, stride=1)
                    y = self._depthwise_separable_conv(y, 128, 3, stride=2)

                    y = self._depthwise_separable_conv(y, 128, 3, stride=1)
                    y = self._depthwise_separable_conv(y, 256, 3, stride=2)

                    y = self._depthwise_separable_conv(y, 256, 3, stride=1)
                    y = self._depthwise_separable_conv(y, 512, 3, stride=2)

                    y = self._depthwise_separable_conv(y, 512, 3, stride=1)
                    y = self._depthwise_separable_conv(y, 512, 3, stride=1)
                    y = self._depthwise_separable_conv(y, 512, 3, stride=1)
                    y = self._depthwise_separable_conv(y, 512, 3, stride=1)
                    y = self._depthwise_separable_conv(y, 512, 3, stride=1)

                    print("y", y)
                    y = self._depthwise_separable_conv(y, 512, 3, stride=2)
                    y = self._depthwise_separable_conv(y, 512, 3, stride=1)

                    avg_pool = tcl.avg_pool2d(y, 7, stride=1)
                    flatten = tf.layers.flatten(avg_pool)

                    self.fc6 = tf.layers.dense(flatten, units=1000, activation=tf.nn.relu)
                    # dropout = tf.nn.dropout(fc6, keep_prob=0.5)
                    predictions = tf.nn.softmax(self.fc6)

                    return predictions


# -------------------------- Demo and Test --------------------------------------------
batch_size = 16
num_batches = 100
import time
import math
from datetime import datetime


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
        model = Mobilenet(224, 3)
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
