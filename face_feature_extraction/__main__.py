# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 13:23:05 2018

@author: kevin
"""


import matplotlib.pyplot as plt
#加载matplotlib用于数据的可视化
from sklearn import decomposition
#加载PCA算法包
from sklearn.datasets import fetch_olivetti_faces
#加载Olivetti人脸数据集导入函数
from numpy.random import RandomState

n_row, n_col = 2, 3

def plot_gallery(title, images, n_col=n_col,n_row=n_row):
    plt.figure(figsize=(2.*n_col,2.26*n_row))
    plt.suptitle(title,size=16)

    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())

        plt.imshopw(comp.reshape(image_shape), cmap=plt.cm.gray,
                    interpolation='nearest',
                    vmin=-vmax, vmax=vmax)

        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01,0.05,0.00,0.93,0.04,0.)



if __name__ == '__main__':
    #设置图像展示时的排列情况

    n_components = n_row * n_col
    image_shape = (64, 64)
    dataset = fetch_olivetti_faces(shuffle=True,random_state=RandomState(0))
    faces = dataset.data  # 加载数据，并打乱顺序


    estimators = [('Eigenfaces - PCA using randomized SVD',decomposition.PCA(n_components=6,whiten=True)),
                  ('Non-negative components - NMF',decomposition.NMF(n_components=6, init='nndsvda',tol=5e-3))]

    for name, estimator in estimators:  # 分别调用PCA和NMF
        estimator.fit(faces)  # 调用PCA或NMF提取特征
    components_ = estimator.components_
    # 获取提取的特征
    plot_gallery(name, components_[:n_components])
    # 按照固定格式进行排列
    plt.show()