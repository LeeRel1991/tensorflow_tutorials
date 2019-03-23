#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: r.li
@license: Apache Licence 
@contact: r.li@bmi-tech.com
@site: 
@software: PyCharm
@file: knn_classifier.py
@time: 18-12-10 下午6:10
@brief： 
"""
from sklearn.model_selection import train_test_split

import cv2
import numpy as np
from collections import Counter

"""
KNN 是通过计算不同特征值之间的距离进行分类,
整体的思路是：如果一个样本在特征空间中的k个最相似(即特征空间中最邻近)的样本中的大多数属于某一个类别，则该样本也属于这个类别。
K通常是不大于20的整数。KNN算法中，所选择的邻居都是已经正确分类的对象。该方法在定类决策上只依据最邻近的一个或者几个样本的类别来决定待分样本所属的类别。
KNN算法要解决的核心问题是K值选择，它会直接影响分类结果。"""


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



def vis_dataset(data, label):
    """
    visualize the image of samples
    :param data: np.array, [n_samples, n_features] 
    :param label: np.array, [n_sampples, 1]
    :return: 
    """
    n_samples = data.shape[0]

    num_per_cls = Counter(label)
    print(num_per_cls)
    # for img, cls in zip(data, label):
    #     img = img.reshape([28, 28])
    #     img = cv2.resize(img, (300, 300))
    #
    #     cv2.putText(img, "label: %d" % cls, (5, 20), cv2.FONT_HERSHEY_PLAIN, 1.2, (255))
    #     cv2.imshow("im", img)
    #     cv2.waitKey(0)

    num_box = 20
    img_size = 28
    img_box = np.zeros((img_size * num_box, img_size * num_box))
    index = np.random.randint(0, n_samples, num_box * num_box)
    for idx, i in enumerate(index):
        img = data[i, :].reshape([28, 28])
        # cv2.imshow("s", img)
        # cv2.waitKey(0)
        # img = cv2.resize(img, (img_size, img_size))
        # img[:, (0,-1)] = 255
        # img[(0, -1), :] = 255
        row, col = idx // num_box, idx % num_box
        print(row, col, idx, i)
        row, col = row * img_size, col * img_size
        img_box[row:row + img_size, col:col + img_size] = img

    cv2.imshow("box", img_box)
    cv2.waitKey(0)


# --------------------------Method 0 --------------------------------------------

def knn_classify(new_sample, train_data, train_label, k=10):
    """    
    :param new_sample: [1, n_features]
    :param train_data:[n_samples, n_features] 
    :param train_label: [n_samples]
    :param k: number of neighbors
    :return: 
    """
    n_samples = train_data.shape[0]
    new_sample = new_sample.astype(np.float32)
    train_data = train_data.astype(np.float32)

    # np.tile 与np.broadcast 类似，默认np.array 运算时会进行broadcast，因此改行并非必须，
    # 仅是为了显示展现运算过程
    new_sample = np.tile(new_sample, (n_samples, 1))

    # np.linalg.norm 默认l2范数，即欧式距离，两行等价
    dis = np.linalg.norm(new_sample - train_data, ord=2, axis=1)
    # dis = np.sqrt(np.sum((new_sample - train_data)**2, axis=1))

    # np.argsort 返回排序后的索引号， 默认从小到大
    sorted_index = np.argsort(dis)

    # 取排序后前k的索引号，及其对应的label标签
    top_k_index = sorted_index[:k]
    top_k_labels = train_label[top_k_index]

    top_k_label_cnts = Counter(top_k_labels)
    max_label = max(top_k_label_cnts.items(), key=lambda x: x[-1])[0]

    return max_label


# --------------------------Method 1 --------------------------------------------
from sklearn.neighbors import KNeighborsClassifier
def knn_classify_sk(new_sample, train_data, train_label, k=10):
    knn = KNeighborsClassifier(n_neighbors=k, p=2)
    knn.fit(train_data, train_label)
    return knn.predict(np.expand_dims(new_sample, 0))


# --------------------------Method 2 --------------------------------------------
def knn_classify_tf(new_sample, train_data, train_label, k=10):
    pass


# --------------------------Eval --------------------------------------------

def eval_knn():
    train_x, test_x, train_y, test_y = get_mnist(0.4)
    # vis_dataset(train_x, train_y)

    # classifier = knn_classify
    classifier = knn_classify_sk
    right = 0
    for s, gt_y in zip(test_x[:500], test_y[:500]):
        pred_y = classifier(s, train_data=train_x[:5000], train_label=train_y[:5000], k=5)
        print("predict: %d, groundtruth: %d" % (pred_y, gt_y))

        if pred_y == gt_y:
            right += 1

    print("total test samples: %d, right: %d " % (test_x.shape[0], right))


if __name__ == '__main__':
    # vis_dataset()
    eval_knn()
