#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""
@author: Abhijith Vemulapati
@created: Sat Nov 21 20:30:00 2020
@description: Implementation of LDA dimension reduction and various binary classifiers on X-ray data
"""


import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from PIL import Image
import time

dim = 220
n_samples = 750

# Change this to the local path of chest_xray directory
datapath = '/Users/abhiv/Downloads/chest_xray/'
train_path = datapath + 'train/'
test_path = datapath + 'test/'

np.random.seed(np.int64(time.time()))

#%%

def applySobel(im):
    im = im.astype('int32')
    dx = ndimage.sobel(im, 1)  # horizontal derivative
    dy = ndimage.sobel(im, 0)  # vertical derivative
    mag = np.hypot(dx, dy)  # magnitude
    mag *= 255.0 / np.max(mag)
    return mag

#%%

# Open all images in a directory and generate feature matrix X with shape (n_samples, dim**2)
def getFeatureMatrix(dirname, filenames, path, sample=True):
    filenames_sample = np.array(filenames)[np.random.choice(len(filenames), n_samples, replace='False')]
    images = list(map(lambda f: Image.open(path + dirname + '/' + f).copy().resize((dim,dim)), filenames_sample))
    imageArr = map(np.array, images)
    # Convert any RGB images to grayscale
    imageArr = [0.2126*i[:,:,0] + 0.7152*i[:,:,1] + 0.0722*i[:,:,2] if len(i.shape) == 3 else i for i in imageArr]
    return np.array(imageArr).reshape(len(imageArr), dim**2)
#%%

normal_files = os.listdir(train_path + 'NORMAL/')
pneumonia_files = os.listdir(train_path + 'PNEUMONIA/')

# Split pneumonia images into bacterial and viral
bacterial_p_files = list(filter(lambda f: 'bacteria' in f, pneumonia_files))
virus_p_files = list(filter(lambda f: 'virus' in f, pneumonia_files))

#%%

# Generate feature matrix and class tags for each class
normal_features = getFeatureMatrix('NORMAL', normal_files, train_path)
normal_tags = np.zeros((normal_features.shape[0],))

bac_features = getFeatureMatrix('PNEUMONIA', bacterial_p_files, train_path)
bac_tags = np.ones((bac_features.shape[0],))

virus_features = getFeatureMatrix('PNEUMONIA', virus_p_files, train_path)
virus_tags = np.ones((virus_features.shape[0],)) + 1
#%%

# virus_tags should be treated as a separate class in the first step of the model
X = np.concatenate((normal_features, bac_features, virus_features))
y = np.concatenate((normal_tags, bac_tags, virus_tags))

# Run LDA to project data on two dimensional subspace
LDA_model = LDA(n_components=2)
LDA_model.fit(X,y)

# Transform training data to run logistic regression on dimension-reducted data
transformed_normal = LDA_model.transform(normal_features)
transformed_pneumonia = LDA_model.transform(np.concatenate((bac_features, virus_features)))
transformed_pneumonia = transformed_pneumonia[np.random.choice(transformed_pneumonia.shape[0], n_samples, replace=False)]
transformed_X = np.concatenate((transformed_normal, transformed_pneumonia))

# Tag transformed data without distinction within pneumonia data
y_bin = np.concatenate((normal_tags, np.ones((transformed_pneumonia.shape[0]))))

#%%
plt.figure(figsize=(11,8))
transformed = [transformed_normal, transformed_pneumonia]
cols = [('Normal','red'),('Pneumonia','blue')]
for cls, col in zip(transformed, cols):
    plt.scatter(cls[:,0],cls[:,1],color=col[1], label=col[0])
plt.title('X-Ray Data 2D Projection - Training Set')
plt.legend(loc = 'lower left')
#%%

normal_test_files = os.listdir(test_path + 'NORMAL/')
pneumonia_test_files = os.listdir(test_path + 'PNEUMONIA/')

normal_test_features = getFeatureMatrix('NORMAL', normal_test_files, test_path)
pneumonia_test_features = getFeatureMatrix('PNEUMONIA', pneumonia_test_files, test_path)

transformed_normal_test = LDA_model.transform(normal_test_features)
transformed_pneumonia_test = LDA_model.transform(pneumonia_test_features)
transformed_X_test = np.concatenate((transformed_normal_test, transformed_pneumonia_test))

normal_tags_test = np.zeros((normal_test_features.shape[0],))
pneumonia_tags_test = np.ones((pneumonia_test_features.shape[0],))
tags_test = np.concatenate((normal_tags_test, pneumonia_tags_test))

#%%

logistic_regress_model = LogisticRegression()
logistic_regress_model.fit(transformed_X, y_bin)
accuracy = logistic_regress_model.score(transformed_X_test, tags_test)

#%%
plt.figure(figsize=(11,8))
transformed_test = [transformed_normal_test, transformed_pneumonia_test]
cols = ['red', 'blue']
cols = [('Normal','red'),('Pneumonia','blue')]
for cls, col in zip(transformed_test, cols):
    plt.scatter(cls[:,0],cls[:,1],color=col[1], label=col[0])
plt.title('X-Ray Data 2D Projection - Testing Set')
plt.legend(loc = 'lower left')
#%%
fig, axes = plt.subplots(2,3)
imgs = [normal_features[0,:],normal_features[1,:],bac_features[0,:],bac_features[1,:],virus_features[0,:],virus_features[1,:]]
plt.axis('off')

for i in np.arange(2):
    for j in np.arange(3):
        axes[i,j].axis('off')
        axes[i,j].imshow(np.reshape(imgs[3*i+j], (220,220), order = 'C'), cmap = 'gray')
