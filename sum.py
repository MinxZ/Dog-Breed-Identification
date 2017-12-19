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

# Loading test Datasets
df = pd.read_csv('../dog_breed_datasets/labels.csv')
n = len(df)
breed = set(df['breed'])
n_class = len(breed)
class_to_num = dict(zip(breed, range(n_class)))
num_to_class = dict(zip(range(n_class), breed))

df2 = pd.read_csv('../dog_breed_datasets/sample_submission.csv')
width = 299
n_test = len(df2)
X_test = np.zeros((n_test, width, width, 3), dtype=np.float16)
for i in tqdm(range(n_test)):
    X_test[i] = cv2.resize(
        cv2.imread('../dog_breed_datasets/test/%s.jpg' % df2['id'][i]),
        (width, width))/127.5 - 1

list_model = {
    "Xception": Xception,
    "InceptionV3": InceptionV3,
    "InceptionResNetV2": InceptionResNetV2
}

model_name = "InceptionResNetV2"

# # Loading models
# model = load_model('../dog_breed_datasets/'+model_name + '.h5')
#
# y_pred = model.predict(X_test, verbose=1)

def get_features(MODEL, data=X_test):
    cnn_model = MODEL(
        include_top=False, input_shape=(width, width, 3), weights=None)

    inputs = Input((width, width, 3))
    x = inputs
    x = cnn_model(x)
    x = GlobalAveragePooling2D()(x)
    cnn_model = Model(inputs, x)

    features = cnn_model.predict(data, batch_size=64, verbose=1)
    return features

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


for b in breed:
    df2[b] = y_pred[:, class_to_num[b]]

df2.to_csv('pred.csv', index=None)
