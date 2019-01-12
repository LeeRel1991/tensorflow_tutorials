#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: r.li
@license: Apache Licence 
@contact: r.li@bmi-tech.com
@site: 
@software: PyCharm
@file: learning_rate.py
@time: 2018/12/6 22:08
@brief： 
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

"""
分段衰减
多项式衰减
指数衰减
自然指数衰减
反时限衰减（not implementation）
余弦衰减（not implementation）
see https://blog.csdn.net/dcrmg/article/details/80017200
https://www.cnblogs.com/cloud-ken/p/8452689.html
"""
max_step = 5000


def generate_lrs():
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # 分段常数衰减 创建分段变化的迭代次数和相应的值.
    # 注：len(step_lr) = len(step_values)+1。n个点分成n+1个区间（值）
    step_values = list(map(int, range(0,max_step, 500)))
    step_lr = [x*0.5/max_step for x in step_values] + [0.5]
    step_lr.reverse()

    lr_piece = tf.train.piecewise_constant(global_step, boundaries=step_values, values=step_lr)

    # 多项式衰减,  cycle为True，学习率在到达最低学习率后往复升高降低
    lr_poly1 = tf.train.polynomial_decay(0.5, global_step, 200, end_learning_rate=0.1, power=2, cycle=True,name=None)
    lr_poly2 = tf.train.polynomial_decay(0.5, global_step, 200, end_learning_rate=0.1, power=2, cycle=False, name=None)

    # 最后一个参数 staircase默认是False表示衰减的学习率是连续的，如果是True代表衰减的学习率是一个离散的间隔。
    lr_exp1 = tf.train.exponential_decay(0.5, global_step, 200, 0.9, True)
    lr_exp2 = tf.train.exponential_decay(0.5, global_step, 200, 0.9, False)

    lr_nexp1 = tf.train.natural_exp_decay(0.5, global_step, 200, 0.9, True)
    lr_nexp2 = tf.train.natural_exp_decay(0.5, global_step, 200, 0.9, False)

    lrs = []
    # tensorboard 可视化
    tf.summary.scalar("lr_piece1", lr_piece)
    tf.summary.scalar("lr_poly1(cycle=True)", lr_poly1)
    tf.summary.scalar("lr_poly2(cycle=False)", lr_poly2)
    tf.summary.scalar("lr_exp1(staircase=True)", lr_exp1)
    tf.summary.scalar("lr_exp2(staircase=False)", lr_exp2)
    tf.summary.scalar("lr_nexp1(staircase=True)", lr_nexp1)
    tf.summary.scalar("lr_nexp2(staircase=False)", lr_nexp2)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('logs/')
        for i in range(max_step):
            # 一定要记得feed_dict 的设置，不然global_step 不会更新
            lr = sess.run([lr_piece, lr_poly1, lr_poly2, lr_exp1, lr_exp2, lr_nexp1, lr_nexp2], feed_dict={global_step: i})
            lrs.append(lr)
            result = sess.run(merged, feed_dict={global_step: i})
            writer.add_summary(result, i)
        return np.array(lrs)


if __name__ == '__main__':
    y = generate_lrs()
    print(y.shape)
    x = range(max_step)

    # plot with matplot
    # plt.subplot(1, 2, 1)
    line1, = plt.plot(x, y[:, 0], 'b-', linewidth=2, label='lr_piece')
    line2, = plt.plot(x, y[:, 1], 'k-', linewidth=2, label="lr_poly1(cycle=True)")
    line3, = plt.plot(x, y[:, 2], 'k--', linewidth=2, label="lr_poly2(cycle=False)")

    line4, = plt.plot(x, y[:, 3], 'm-', linewidth=2, label="lr_exp1(staircase=True)")
    line5, = plt.plot(x, y[:, 4], 'g-', linewidth=2, label="lr_exp2(staircase=False)")
    line6, = plt.plot(x, y[:, 5], 'c-', linewidth=2, label="lr_nexp1(staircase=True)")
    line7, = plt.plot(x, y[:, 6], 'r-', linewidth=2, label="lr_nexp2(staircase=False)")
    ax = plt.gca()
    ax.set_ylim([0, 0.55])
    ax.set_xlim([0, 5500])

    plt.title('learning rate')
    ax.legend(loc='best')
    plt.show()
