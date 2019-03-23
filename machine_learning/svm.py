#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: r.li
@license: Apache Licence 
@contact: r.li@bmi-tech.com
@site: 
@software: PyCharm
@file: svm.py
@time: 18-12-18 上午11:06
@brief： 
"""

from sklearn import datasets, svm, metrics
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
import numpy as np


def get_mnist(test_ratio=0.4):
    """
    get mnist with sklearn
    :param test_ratio: ratio that test samples account for
    :return: train data and test data
    """
    from sklearn.datasets import fetch_mldata
    # fetch_mldata 返回的是一个字典，有"target"(与data对应，样本的标签), "data"([N_sample, N_feature]训练样本), "COL_NAMES"等key
    # 其中MNIST original 包含70000张图片，每个图片有784个特征(图片的像素值，28x28)
    dataset = fetch_mldata("MNIST original")
    samples = dataset["data"]
    label = dataset["target"]
    # 避免过拟合，采用交叉验证，验证集占训练集20%，固定随机种子（random_state)
    train_X, test_X, train_y, test_y = train_test_split(samples,
                                                        label,
                                                        test_size=test_ratio,
                                                        random_state=0)
    return train_X, test_X, train_y, test_y


# --------------------------Eval --------------------------------------------

def eval_svm():
    train_x, test_x, train_y, test_y = get_mnist(0.4)

    classifier = svm.SVC(gamma=0.001, C=1, kernel="poly", degree=2, verbose=False)
    classifier.fit(train_x[:5000], train_y[:5000])

    right = 0
    for s, gt_y in zip(test_x[:500], test_y[:500]):
        pred_y = classifier.predict(np.expand_dims(s, 0))
        print("predict: %d, groundtruth: %d" % (pred_y, gt_y))

        if pred_y == gt_y:
            right += 1

    print("total test samples: %d, right: %d " % (test_x.shape[0], right))


if __name__ == '__main__':
    eval_svm()
