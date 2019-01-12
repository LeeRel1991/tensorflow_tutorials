#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: r.li
@license: Apache Licence 
@contact: r.li@bmi-tech.com
@site: 
@software: PyCharm
@file: activation_functions.py
@time: 2018/12/6 22:22
@brief： 
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def activation(in_data):
    x = tf.constant(in_data, dtype=tf.float32)

    # y = 1 / (1 + exp(-x))
    # sigmoid, 取值范围为(0,1)
    # 激活函数计算量大，反向传播求误差梯度时，求导涉及除法
    # 反向传播时，很容易就会出现梯度消失的情况，从而无法完成深层网络的训练
    # Sigmoids函数饱和且kill掉梯度。
    # Sigmoids函数收敛缓慢。
    sigmoid = tf.nn.sigmoid(x)

    # y = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    # 也称为双切正切函数，取值范围为[-1,1]。
    # tanh在特征相差明显时的效果会很好，在循环过程中会不断扩大特征效果。
    # 与 sigmoid 的区别是，tanh 是 0 均值的，因此实际应用中 tanh 会比 sigmoid 更好。
    tanh = tf.nn.tanh(x)

    # y = max(x, 0)
    # 使用 ReLU 得到的 SGD 的收敛速度会比 sigmoid/tanh 快很多.
    # sigmoid 的梯度消失问题，ReLU 的导数就不存在这样的问题，它的导数表达式如下：
    # ReLU 的缺点：
    # 训练的时候很”脆弱”，很容易就”die”了
    # 例如，一个非常大的梯度流过一个 ReLU 神经元，更新过参数之后，这个神经元再也不会对任何数据有激活现象了，那么这个神经元的梯度就永远都会是 0.
    # 如果 learning rate 很大，那么很有可能网络中的 40% 的神经元都”dead”了。

    relu = tf.nn.relu(x)

    # y = max(0, x) + leak * min(0, x)
    # 针对ReLU函数中的硬饱和，出现了Leaky-ReLU与P-ReLU，它们在形式上相似，不同的是在Leaky-ReLU中，α是一个常数，
    # 在P-ReLU中，a也可以作为一个参数来学习。
    lrelu = tf.nn.leaky_relu(x, alpha=0.2)


    with tf.Session() as sess:
        sigmoid_out, tanh_out , relu_out, lrelu_out = sess.run([sigmoid, tanh, relu, lrelu])

    return sigmoid_out, tanh_out, relu_out, lrelu_out

if __name__ == '__main__':
    x = np.arange(-10, 10, 0.01)
    sigmoid_out, tanh_out, relu_out, lrelu_out = activation(x)

    # plot
    plt.subplot(1, 2, 1)

    line4, = plt.plot(x, sigmoid_out, 'r-', linewidth=2, label="sigmoid")
    line5, = plt.plot(x, tanh_out, 'g-', linewidth=2, label="tanh")
    ax = plt.gca()
    ax.legend(loc='best')
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))
    plt.title('sigmoid vs tanh')


    plt.subplot(1, 2, 2)
    line6, = plt.plot(x, relu_out, 'b-', linewidth=2, label="relu")
    line7, = plt.plot(x, lrelu_out, 'b--', linewidth=2, label="lrelu")
    ax = plt.gca()
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))

    plt.title('relu vs leaky relu')
    ax.legend(loc='best')
    plt.show()
