from __future__ import absolute_import, division, print_function

import argparse
import os
import random
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from keras import backend as K
from keras.applications import *
from keras.applications.inception_v3 import preprocess_input
from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import *
from keras.utils.vis_utils import model_to_dot
from tqdm import tqdm


def run(lr, compiler):
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
    # Loading Datasets
    print('\n\n Loading Datasets. \n')
    for i in tqdm(range(n)):
        X[i] = cv2.resize(
            cv2.imread('../dog_breed_datasets/train/%s.jpg' % df['id'][i]),
            (width, width))
        y[i][class_to_num[df['breed'][i]]] = 1

    dvi = int(X.shape[0] * 0.9)
    x_train = X[:dvi, :, :, :]
    y_train = y[:dvi, :]
    x_val = X[dvi:, :, :, :]
    y_val = y[dvi:, :]

    # Compute the bottleneck feature

    def get_features(MODEL, data=X):
        cnn_model = MODEL(
            include_top=False,
            input_shape=(width, width, 3),
            weights='imagenet')

        inputs = Input((width, width, 3))
        x = inputs
        x = Lambda(preprocess_input, name='preprocessing')(x)
        x = cnn_model(x)
        x = GlobalAveragePooling2D()(x)
        cnn_model = Model(inputs, x)

        features = cnn_model.predict(data, batch_size=32, verbose=1)
        return features

    def fine_tune(MODEL, model_name, compiler, lr, X=X):
        # Fine-tune the model
        print("\n\n Fine tune " + model_name + " : \n")

        datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        #     rotation_range=20,
        #     width_shift_range=0.2,
        #     height_shift_range=0.2)

        val_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input)

        inputs = Input((width, width, 3))
        x = inputs
        cnn_model = MODEL(
            include_top=False,
            input_shape=(width, width, 3),
            weights='imagenet')
        x = cnn_model(x)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)
        x = Dense(n_class, activation='softmax', name='predictions')(x)
        model = Model(inputs=inputs, outputs=x)

        # for layer in model.layers[:114]:
        #     layer.trainable = False

        try:
            model.load_weights('Xception.h5')
            print('Load Xception.h5 successfully.')
        except:
            print('Start computing ' + model_name + ' bottleneck feature: ')
            features = get_features(MODEL, X)

            # Training models
            inputs = Input(features.shape[1:])
            x = inputs
            x = Dropout(0.5)(x)
            x = Dense(n_class, activation='softmax', name='predictions')(x)
            model_fc = Model(inputs, x)
            model_fc.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
            h = model_fc.fit(
                features, y, batch_size=128, epochs=5, validation_split=0.1)

            model_fc.save('fc_' + model_name + '.h5', 'w')
            model.load_weights('fc_' + model_name + '.h5', by_name=True)
            print('Fail to load Xception1.h5, load fc.h5 instead.')

        print("\n "+"Compiler="+compiler+"lr=" + str(lr) + " \n")

        if compiler=="Nadam":
            model.compile(
                optimizer=Nadam(lr=lr),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
        elif compiler=="SGD":
            model.compile(
                loss='categorical_crossentropy',
                optimizer=SGD(lr=0.005, momentum=0.9, nesterov=True, decay=0.014),
                metrics=['accuracy'])

        batch_size = 16
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=3, verbose=1, mode='auto')
        h2 = model.fit_generator(
            datagen.flow(x_train, y_train, batch_size=batch_size),
            steps_per_epoch=len(x_train) / batch_size,
            validation_data=val_datagen.flow(
                x_val, y_val, batch_size=batch_size),
            validation_steps=len(x_val) / batch_size,
            epochs=10000,
            callbacks=[early_stopping])

        model.save(model_name + '.h5', 'w')

    list_model = [Xception, InceptionV3, InceptionResNetV2]
    list_name_model = ["Xception", "InceptionV3", "InceptionResNetV2"]

    for x in range(3):
        fine_tune(list_model[x], list_name_model[x], compiler, lr, X)


def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Learnign Rate changer")
    parser.add_argument(
        "--lr", help="learning rate", default=0.0005, type=float)
    parser.add_argument(
        "--compiler", help="Compiler", default="SGD", type=str)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.lr, args.compiler)

