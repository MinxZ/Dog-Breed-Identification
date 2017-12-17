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
df2 = pd.read_csv('../dog_breed_datasets/sample_submission.csv')

n_test = len(df2)
X_test = np.zeros((n_test, width, width, 3), dtype=np.uint8)
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
# Loading models
model = load_model('../dog_breed_datasets/'+model_name + '.h5')
y_pred = model.predict(X_test)

for b in breed:
    df2[b] = y_pred[:, class_to_num[b]]

df2.to_csv('pred.csv', index=None)
