# -*- coding: utf-8 -*-
"""
Created on 2019-11-13 20:23:05

@author: kevin
"""


from sklearn.datasets import make_blobs

from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


data = make_blobs(n_samples=200,centers=2,random_state=8)

X, y = data

plt.scatter(X[:,0],X[:,1],c=y, cmap=plt.cm.spring,edgecolor = 'k')

plt.show()
