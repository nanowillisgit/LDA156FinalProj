#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""
@author: Aatmun Baxi
@created: Sat Nov 21 16:10:00 2020
@description: Implementation of dimensionality reduction on x-ray image set
"""



####### To recreate the directory structure for the data, just download
####### the dataset .zip and extract it in the same directory as this file

import os
import glob

from random import sample

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split

import cv2

train_path_normal = 'data/chest_xray/chest_xray/train/NORMAL/'
train_path_pneum = 'data/chest_xray/chest_xray/train/PNEUMONIA/'

normal_ims = [ (im, 0) for im in glob.glob(train_path_normal +  '*.jpeg') ]
pneum_ims = [ (im, 1) for im in glob.glob(train_path_pneum + '*.jpeg') ]

normal_imvecs = []
pneum_imvecs = []

# Set desired number of images from train set
N = 300

normal_choices = sample(range(0,len(normal_ims)), N)
pneum_choices = sample(range(0,len(pneum_ims)), N)

# Set image resize dimensions
dim = 130

for i in normal_choices:
    pix_vec = cv2.imread( normal_ims[i][0])

    pix_vec.resize((dim,dim))
    pix_vec = np.reshape(pix_vec, dim*dim)

    normal_imvecs.append(np.concatenate((pix_vec, np.zeros(1))))


for i in pneum_choices:
    pix_vec = cv2.imread( pneum_ims[i][0])

    pix_vec.resize((dim,dim))
    pix_vec = np.reshape(pix_vec, dim*dim)

    pneum_imvecs.append(np.concatenate((pix_vec, np.ones(1))))

immat = np.concatenate( (np.array(normal_imvecs), np.array(pneum_imvecs)) )

# Set numpy see for reproducibility
np.random.seed(20)
np.random.shuffle(immat)

data = immat[:,:(dim**2)]
classes = immat[:,(dim**2)]

# Set number of desired dimensions after reduction
# NOTE: To display images after reduction, this must be a perfect square number

final_dim = 36

model = LDA(n_components=None)

reduced = model.fit_transform(data,classes)


colors = [ 'red', 'blue' ]

zeros = reduced[classes == 0]
ones = reduced[classes == 1]

zeros = np.reshape(zeros, len(zeros))
ones = np.reshape(ones, len(ones))

plt.scatter(zeros, np.zeros(len(zeros)), color='red')
plt.scatter(ones, np.zeros(len(ones)), color='cyan')

plt.title("Post-reduction data")
plt.show()

test_path_normal = 'data/chest_xray/chest_xray/test/NORMAL/'
test_path_pneum = 'data/chest_xray/chest_xray/test/PNEUMONIA/'

normal_ims_test = [ (im, 0) for im in glob.glob(test_path_normal +  '*.jpeg') ]
pneum_ims_test = [ (im, 1) for im in glob.glob(test_path_pneum + '*.jpeg') ]

normal_imvecs_test = []
pneum_imvecs_test = []

# Set desired number of images from train set
test_N = 30

normal_choices_test = sample(range(0,len(normal_ims_test)), test_N)
pneum_choices_test = sample(range(0,len(pneum_ims_test)), test_N)

for i in normal_choices_test:
    pix_vec = cv2.imread( normal_ims_test[i][0])

    pix_vec.resize((dim,dim))
    pix_vec = np.reshape(pix_vec, dim*dim)

    normal_imvecs_test.append(np.concatenate((pix_vec, np.zeros(1))))


for i in pneum_choices_test:
    pix_vec = cv2.imread( pneum_ims_test[i][0])

    pix_vec.resize((dim,dim))
    pix_vec = np.reshape(pix_vec, dim*dim)

    pneum_imvecs_test.append(np.concatenate((pix_vec, np.ones(1))))

immat_test = np.concatenate( (np.array(normal_imvecs_test), np.array(pneum_imvecs_test)) )
np.random.shuffle(immat_test)
test_attr = immat_test[:,:(dim**2)]
test_classes = immat_test[:,(dim**2)]


model.fit(data,classes)

test_reduced = model.transform(test_attr)

colors = [ 'red', 'blue' ]

zeros = test_reduced[test_classes == 0]
ones = test_reduced[test_classes == 1]

test_zeros = np.reshape(zeros, len(zeros))
test_ones = np.reshape(ones, len(ones))

plt.scatter(zeros, np.zeros(len(zeros)), color='red')
plt.scatter(ones, np.zeros(len(ones)), color='cyan')

plt.title("Post-reduction test data to train-fitted model")
plt.show()
