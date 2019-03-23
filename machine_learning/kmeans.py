#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: r.li
@license: Apache Licence 
@contact: r.li@bmi-tech.com
@site: 
@software: PyCharm
@file: kmeans.py
@time: 18-12-14 上午9:29
@brief： 
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split


def eclud_distance(sample1, sample2):
    """
    
    :param sample1: [n_samples, n_features]
    :param sample2: [1, n_features]
    :return: 
    """

    return np.sqrt(np.sum((sample1 - sample2)**2, axis=1))


def show(dataset, centroids, cluster_assment):
    k = centroids.shape[0]
    # cs = np.zeros((k, 4), dtype=np.int8)
    for i in range(k):
        cs = np.zeros(4, dtype=np.int8)
        cs[i] = 1
        cs[3] = 1
        idx = np.where(cluster_assment[:, 0] == i)[0]
        samples = dataset[idx]
        plt.scatter(samples[:, 0], samples[:, 1], c=cs)
        plt.scatter(centroids[i, 0], centroids[i, 1], c=cs, linewidths=20)
    plt.show()


# 随机生成K个质心
def random_center(pointers, k):
    """

    :param pointers: 
    :param k: 
    :return: 
    """
    idx = np.random.random_integers(0, len(pointers) - 1, k)
    centers = pointers[idx, :]

    return centers


def KMeans(train_data, init_centers, max_iter=200):
    n_samples = train_data.shape[0]

    # 用于存放该样本属于哪类及质心距离
    cluster_assment = np.zeros((n_samples, 2))

    goon = True  # 用来判断聚类是否已经收敛

    n_cluster = init_centers.shape[0]
    centers = init_centers.copy()
    n_iter = 0
    while goon and n_iter < max_iter:
        goon = False
        n_iter += 1
        for i in range(n_samples):  # 把每一个数据点划分到离它最近的中心点
            # 计算样本i与每个center的距离，将其归到距离最小的center对应的那一类
            dis = eclud_distance(centers, train_data[i, :])
            min_index = np.argsort(dis)[0]
            min_dist = dis[min_index]

            # 如果样本i的聚类归属发生变化，则需要继续迭代
            if cluster_assment[i, 0] != min_index:
                goon = True

            # 更新样本i的聚类归属及距离
            cluster_assment[i, :] = min_index, min_dist ** 2

        for k in range(n_cluster):  # 重新计算中心点
            samples_in_clust = train_data[np.where(cluster_assment[:, 0] == k)[0]]
            centers[k, :] = np.mean(samples_in_clust, axis=0)  # 算出这些数据的中心点

    return centers, cluster_assment


def random_samples():
    ponters = [np.random.random_integers(0, 10, 2) for a in range(50)]
    np.save("data", ponters)
    print(ponters)


def demo():

    pointers = np.load("data.npy")
    centers = random_center(pointers, 3)

    centers, cluster_assment = KMeans(pointers, centers)
    show(pointers, centers, cluster_assment)


def divide_group(data, label, test_ratio=0.4):
    # 避免过拟合，采用交叉验证，验证集占训练集20%，固定随机种子（random_state)
    train_X, test_X, train_y, test_y = train_test_split(data,
                                                        label,
                                                        test_size=test_ratio,
                                                        random_state=0)

    return train_X, test_X, train_y, test_y


def demo_mnist():
    dataset = fetch_mldata("MNIST original")
    samples = dataset["data"]
    label = dataset["target"]

    train_x, test_x, train_y, test_y = divide_group(samples, label, 0.4)
    train_x = train_x[:500, :].astype(np.float32)
    train_y = train_y[:500].astype(np.float32)
    label_list = np.unique(train_y).astype(np.uint8)

    n_center = len(label_list)
    init_centers = np.zeros((n_center, train_x.shape[1]))
    center_label = np.zeros(n_center)
    for i in label_list:
        idx = np.where(train_y == i)[0]

        init_centers[i, :] = np.mean(train_x[idx, :], 0)
        center_label[i] = i

    centers, cluster_assment = KMeans(train_x, init_centers, max_iter=2000)

    for i in range(centers.shape[0]):
        idx = np.where(cluster_assment[:, 0] == i)[0]
        samples = train_x[idx]

        img = samples.reshape((-1, 28, 28))
        img = np.concatenate(img, 1)

        cv2.imshow("im", img)
        cv2.waitKey(0)



if __name__ == '__main__':
    # random_samples()
    # demo()
    demo_mnist()