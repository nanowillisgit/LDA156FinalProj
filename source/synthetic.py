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
import matplotlib.pyplot as plt

features, labels = mb(n_samples=1000,
                      n_features=100,
                      centers=5, # Number of clusters to generate
                      cluster_std=[9.2, 16.4, 15.3, 11.5, 7.0 ], # Adjust these to change intraclass scatter
                      shuffle=True,
                      random_state=0)

colors = ['blue', 'green', 'red', 'magenta', 'purple']

lda = LDA(n_components=2,solver='eigen')
cov = np.cov(features)

evals, evecs = linalg.eig(cov)

evals = evals.astype(float)
eigvecs = evecs.astype(float)

e_pairs = [(np.abs(evals[i]), eigvecs[:,i]) for i in range(len(evals))]
e_pairs.sort(key=lambda x:x[0], reverse=True)

x = np.arange(0,len(evals))
dists = [ evals[M:len(evals)].sum() for M in range(len(evals))]

plt.scatter( x,dists ,marker=".")

plt.xlabel("M")
plt.ylabel("Distortion")
plt.title("Distortion")

lda.fit(features,labels)

reduced = lda.transform(features)

plt.figure()

for i in range(5):
    plt.scatter(reduced[labels==i][:,0],reduced[labels==i][:,1],
                color=colors[i],marker='.')
plt.title("LDA-projected data")
plt.show()

