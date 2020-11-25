#!/usr/bin/env python3

"""
@author: Aatmun Baxi
@created: Tue Nov 24 10:23:42 2020
@description: Implementation of LDA on fashion MNIST
"""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn import svm

import random

import numpy as np

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

import pandas as pd
from pandas.plotting import scatter_matrix

import time

def reduce_and_classify(post_redux_dim,
                          train_data_a,train_data_lab,
                                 test_data_a,test_data_lab):
    """
    Keyword Arguments:
    post_redux_dim -- 
    train_data_a   -- 
    train_data_lab -- 
    test_data_a    -- 
    test_data_lab  -- 
    """
    model = LDA(n_components=post_redux_dim)
    reduced_train = model.fit_transform(train_data_a, train_data_lab)
    reduced_test = model.transform(test_data_a)
    clf = RFC()
    clf.fit(reduced_train,train_data_lab)
    return clf.score(reduced_test,test_data_lab)

data_train = pd.read_csv("data/fashion-mnist_train.csv")
data_test = pd.read_csv("data/fashion-mnist_test.csv")

data_train_labels = data_train.iloc[:,0]
data_train_attr = data_train.iloc[:,1:]
print("Initial dimensionality of data: ", data_train_attr.shape)

NUM_CLASSES = max(data_train_labels)

data_test_labels = data_test.iloc[:,0]
data_test_attr = data_test.iloc[:,1:]

# for i in range(1,NUM_CLASSES+1):
#     print(f'Random Forest on dimension {i} reduced data:', reduce_and_classify(
#         i,data_train_attr, data_train_labels, data_test_attr, data_test_labels
#     ))


# lda_start = time.time()
# model = LDA(n_components=9)
# red_data_train = model.fit_transform(data_train_attr,data_train_labels)
# red_data_test = model.transform(data_test_attr)
# rfc = RFC()
# rfc.fit(red_data_train,data_train_labels)
# print("LDA dimension 9 random forest classifier score:", rfc.score(red_data_test,data_test_labels) )
# lda_end = time.time()

# rfc_start = time.time()
# model = RFC()
# model.fit(data_train_attr,data_train_labels)
# print("Just random forest classifier score:", model.score(data_test_attr,data_test,labels) )
# rfc_end = time.time()

# print("LDA + RFC time:")
# print("--- %s seconds" % (lda_end - lda_start))
# print("RFC alone time:")
# print("--- %s seconds" % (rfc_end - rfc_start))


RED_DIM = 9

model = LDA(n_components=RED_DIM)

model.fit(data_train_attr,data_train_labels)

red_data_train = model.transform(data_train_attr)
print("Post-reduction dimensionality of data: ", red_data_train.shape)

# colors = ['aqua', 'black', 'darkblue', 'darkgreen', 'gold', 'red', 'plum', 'sienna', 'violet',
#           'olive']


# # Number of points to plot
# NUM_POINTS = 30

# if RED_DIM == 3:
#     fig = plt.figure(figsize=(10,7))
#     ax = plt.axes(projection="3d")
#     for i in random.sample(range(len(red_data_train)),NUM_POINTS):
#         X = red_data_train[i,0]
#         Y = red_data_train[i,1]
#         Z = red_data_train[i,2]
#         ax.scatter3D(X,Y,Z, color=colors[data_train_labels[i]])

#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_zlabel('z')
#     plt.title("Projected train data")
#     plt.show()

# if RED_DIM == 2:
#     plt.figure()
#     for i in random.sample(range(len(red_data_train)),NUM_POINTS):
#         X = red_data_train[i,0]
#         Y = red_data_train[i,1]
#         #Z = red_data_train[i,2]
#         plt.scatter(X,Y, color=colors[data_train_labels[i]])

#     plt.title("Projected train data")
#     plt.show()


red_test_data = model.transform(data_test_attr)

#rand_sample = random.sample(range(len(data_train_attr)),30)

# reduced_df_train = pd.DataFrame(random.sample(list(red_data_train), 100),
#                                 columns=['x1','x2','x3','x4','x5','x6','x7','x8','x9'])
# scatter_matrix(reduced_df_train, figsize=(10,10),alpha=0.3, diagonal='kde')
# plt.show()

predictor = svm.SVC(kernel="sigmoid")
predictor.fit(red_data_train, data_train_labels)

guesses = predictor.score(red_test_data, data_test_labels)

print("Classifier on reduced data score: ",guesses)
