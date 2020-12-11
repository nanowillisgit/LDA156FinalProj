#!/usr/bin/env python3

"""
@author: Aatmun Baxi
@created: Sun Dec  6 17:52:34 2020
@description: LDA application on synthetic data
"""
import numpy as np
from numpy import linalg as linalg
from sklearn.datasets import make_blobs as mb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

features, labels = mb(n_samples=1000,
                      n_features=100,
                      centers=5, # Number of clusters to generate
                      cluster_std=[8.0, 8.0, 8.0, 8.0, 8.0], # Adjust these to change intraclass scatter
                      shuffle=True,
                      random_state=2)

# colors = ['blue', 'green', 'red', 'magenta', 'purple']

# lda = LDA(n_components=2,solver='eigen')
# cov = np.cov(features)

# evals, evecs = linalg.eig(cov)

# evals = evals.astype(float)
# eigvecs = evecs.astype(float)

# e_pairs = [(np.abs(evals[i]), eigvecs[:,i]) for i in range(len(evals))]
# e_pairs.sort(key=lambda x:x[0], reverse=True)

# x = np.arange(0,len(evals))
# dists = [ evals[M:len(evals)].sum() for M in range(len(evals))]

# plt.scatter( x,dists ,marker=".")

# plt.xlabel("M")
# plt.ylabel("$\sum$")
# plt.title("$Distortion=\sum_{n>M}\lambda_n$ for $\{\lambda_n\}_n ^{100}$ sorted eignvalues")

# lda.fit(features,labels)

# reduced = lda.transform(features)

# plt.figure()

# for i in range(5):
#     plt.scatter(reduced[labels==i][:,0],reduced[labels==i][:,1],
#                 color=colors[i],marker='.')
# plt.title("LDA-projected data")
# plt.show()



## Classifcation ##

train_attr, test_attr, train_lab, test_lab = train_test_split(features,labels,
    train_size = 0.4,
    random_state=0,
    shuffle=True)

classifier = LDA(n_components=2,solver='eigen')
classifier.fit_transform(train_attr,train_lab)
#print(classifier.score(test_attr,test_lab))

reduced_test = classifier.transform(test_attr)

print(classifier.score(reduced_test, test_attr))
