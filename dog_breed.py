from __future__ import absolute_import, division, print_function

import argparse
import os
import random
from datetime import datetime

import cv2
import keras
import numpy as np
import pandas as pd
from keras import backend as K
from keras.applications import *
from keras.applications.inception_v3 import preprocess_input
from keras.callbacks import *
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import *
from keras.utils.vis_utils import model_to_dot
from tqdm import tqdm


# Loading Datasets
df = pd.read_csv('../dog_breed_datasets/labels.csv')
df.head()
n = len(df)
breed = set(df['breed'])
n_class = len(breed)
class_to_num = dict(zip(breed, range(n_class)))
num_to_class = dict(zip(range(n_class), breed))

width = 299
X = np.zeros((n, width, width, 3), dtype=np.uint8)
y = np.zeros((n, n_class), dtype=np.uint8)
for i in tqdm(range(n)):
    X[i] = cv2.resize(
        cv2.imread('../dog_breed_datasets/train/%s.jpg' % df['id'][i]),
        (width, width))
    y[i][class_to_num[df['breed'][i]]] = 1

# Data visualization
#
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'
#
# plt.figure(figsize=(12, 6))
# for i in range(8):
#     random_index = random.randint(0, n-1)
#     plt.subplot(2, 4, i+1)
#     plt.imshow(X[random_index][:,:,::-1])
#     plt.title(num_to_class[y[random_index].argmax()])
#

# Computing feature


def get_features(MODEL, data=X):
    cnn_model = MODEL(
        include_top=False, input_shape=(width, width, 3), weights=None)

    inputs = Input((width, width, 3))
    x = inputs
    x = Lambda(preprocess_input, name='preprocessing')(x)
    x = cnn_model(x)
    x = GlobalAveragePooling2D()(x)
    cnn_model = Model(inputs, x)

    features = cnn_model.predict(data, batch_size=64, verbose=1)
    return features


print('Start computing inception_features: ')
inception_features = get_features(InceptionV3, X)
print('Start computing xception_features: ')
xception_features = get_features(Xception, X)
features = np.concatenate([inception_features, xception_features], axis=-1)

# Training models
inputs = Input(features.shape[1:])
x = inputs
x = Dropout(0.5)(x)
x = Dense(n_class, activation='softmax')(x)
model = Model(inputs, x)
model.compile(
    optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
h = model.fit(features, y, batch_size=128, epochs=10, validation_split=0.1)

# Models visualization

# SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))

# Training preprocess visualization

# import matplotlib.pyplot as plt
#
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'
#
# plt.figure(figsize=(10, 4))
# plt.subplot(1, 2, 1)
# plt.plot(h.history['loss'])
# plt.plot(h.history['val_loss'])
# plt.legend(['loss', 'val_loss'])
# plt.ylabel('loss')
# plt.xlabel('epoch')
#
# plt.subplot(1, 2, 2)
# plt.plot(h.history['acc'])
# plt.plot(h.history['val_acc'])
# plt.legend(['acc', 'val_acc'])
# plt.ylabel('acc')
# plt.xlabel('epoch')

# Loading test Datasets
df2 = pd.read_csv('../dog_breed_datasets/sample_submission.csv')

n_test = len(df2)
X_test = np.zeros((n_test, width, width, 3), dtype=np.uint8)
for i in tqdm(range(n_test)):
    X_test[i] = cv2.resize(
        cv2.imread('../dog_breed_datasets/test/%s.jpg' % df2['id'][i]),
        (width, width))

# Compute test dataset features
print('Start computing inception_features: ')
inception_features = get_features(InceptionV3, X_test)
print('Start computing xception_features: ')
xception_features = get_features(Xception, X_test)

features_test = np.concatenate(
    [inception_features, xception_features], axis=-1)

# predict
y_pred = model.predict(features_test, batch_size=128)

for b in breed:
    df2[b] = y_pred[:, class_to_num[b]]

df2.to_csv('pred.csv', index=None)
