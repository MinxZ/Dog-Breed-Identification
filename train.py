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


def run(model_name, lr, optimizer, epoch, patience, batch_size, test=None):
    # Loading Datasets
    def load_data():

        df = pd.read_csv('../dog_breed_datasets/labels.csv')

        n = len(df)
        breed = set(df['breed'])
        n_class = len(breed)
        class_to_num = dict(zip(breed, range(n_class)))
        num_to_class = dict(zip(range(n_class), breed))

        width = 299
        X = np.zeros((n, width, width, 3), dtype=np.float16)
        y = np.zeros((n, n_class), dtype=np.uint8)
        # Loading Datasets
        print('\n\n Loading Datasets. \n')
        for i in tqdm(range(n)):
            X[i] = cv2.resize(
                cv2.imread('../dog_breed_datasets/train/%s.jpg' % df['id'][i]),
                (width, width))/127.5 - 1
            y[i][class_to_num[df['breed'][i]]] = 1

        dvi = int(X.shape[0] * 0.9)
        x_train = X[:dvi, :, :, :]
        y_train = y[:dvi, :]
        x_val = X[dvi:, :, :, :]
        y_val = y[dvi:, :]
        return x_train, y_train, x_val, y_val
    # x_train, y_train, x_val, y_val = load_data()
    x_train = np.load('../dog_breed_datasets/x_train.npy')
    y_train = np.load('../dog_breed_datasets/y_train.npy')
    x_val = np.load('../dog_breed_datasets/x_val.npy')
    y_val = np.load('../dog_breed_datasets/y_val.npy')

    width = x_train.shape[1]
    n_class = y_train.shape[1]
    # Compute the bottleneck feature
    def get_features(MODEL, data=x_train):
        cnn_model = MODEL(
            include_top=False,
            input_shape=(width, width, 3),
            weights='imagenet')
        inputs = Input((width, width, 3))
        x = inputs
        # x = Lambda(preprocess_input, name='preprocessing')(x)
        x = cnn_model(x)
        x = GlobalAveragePooling2D()(x)
        cnn_model = Model(inputs, x)

        features = cnn_model.predict(data, batch_size=32, verbose=1)
        return features

    def fine_tune(MODEL,
                  model_name,
                  optimizer,
                  lr,
                  epoch,
                  patience,
                  batch_size,
                  X=x_train,
                  test=None):
        # Fine-tune the model
        print("\n\n Fine tune " + model_name + ": \n")

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

#         Loading weights
        try:
            model.load_weights('../dog_breed_datasets/'+model_name + '.h5')
            print('Load ' + model_name + '.h5 successfully.')
        except:
            try:
                model.load_weights('../dog_breed_datasets/'+'fc_' + model_name + '.h5', by_name=True)
                print('Fail to load ' + model_name + '.h5, load fc_' +
                      model_name + '.h5 instead.')
            except:
                print(
                    'Start computing ' + model_name + ' bottleneck feature: ')
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
                    features,
                    y_train,
                    batch_size=128,
                    epochs=5,
                    validation_split=0.1)

                model_fc.save('../dog_breed_datasets/'+'fc_' + model_name + '.h5', 'w')
                model.load_weights('../dog_breed_datasets/'+'fc_' + model_name + '.h5', by_name=True)


        print("\n " + "Optimizer=" + optimizer + " lr=" + str(lr) + " \n")
        if optimizer == "Adam":
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
        elif optimizer == "SGD":
            model.compile(
                loss='categorical_crossentropy',
                optimizer=SGD(lr=lr, momentum=0.9, nesterov=True),
                metrics=['accuracy'])

        from random_eraser import get_random_eraser

        datagen = ImageDataGenerator(
            preprocessing_function=get_random_eraser(p=0.2, s_l=0.02, s_h=0.2, r_1=0.3, r_2=1/0.3,
                  v_l=0, v_h=60, pixel_level=False),
            horizontal_flip=True,
            zoom_range=0.1,
            rotation_range=10)

        val_datagen = ImageDataGenerator()

        if not test:
            class LossHistory(keras.callbacks.Callback):
                def on_train_begin(self, logs={}):
                    self.losses = []
                def on_epoch_end(self, batch, logs={}):
                    self.losses.append((logs.get('loss'), logs.get("val_loss")))
            history = LossHistory()
            early_stopping = EarlyStopping(
                monitor='val_loss', patience=patience, verbose=1, mode='auto')
            checkpointer = ModelCheckpoint(
                filepath='../dog_breed_datasets/'+model_name + '.h5', verbose=0, save_best_only=True)
            reduce_lr = ReduceLROnPlateau(factor=0.5, patience=0, verbose=1)

            model.fit_generator(
                datagen.flow(x_train, y_train, batch_size=batch_size),
                steps_per_epoch=len(x_train) / batch_size,
                validation_data=val_datagen.flow(
                    x_val, y_val, batch_size=batch_size),
                validation_steps=len(x_val) / batch_size,
                epochs=epoch,
                callbacks=[history, early_stopping, checkpointer, reduce_lr])
            with open(model_name + ".csv", 'a') as f_handle:
                np.savetxt(f_handle, history.losses)
        else:
            print('Evalute on test set')
            val_datagen.fit(x_test)
            score = model.evaluate_generator(
                val_datagen.flow(x_test, y_test, batch_size=batch_size),
                len(x_test) / batch_size)
            print(score)
            return score

    list_model = {
        "Xception": Xception,
        "InceptionV3": InceptionV3,
        "InceptionResNetV2": InceptionResNetV2
    }
    fine_tune(list_model[model_name], model_name, optimizer, lr, epoch,
              patience, batch_size, x_train, test)


def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Hyper parameter")
    parser.add_argument(
        "--model", help="Model to use", default="Xception", type=str)
    parser.add_argument(
        "--lr", help="learning rate", default=0.0005, type=float)
    parser.add_argument(
        "--optimizer", help="optimizer", default="Adam", type=str)
    parser.add_argument(
        "--epoch", help="Number of epochs", default=1e4, type=int)
    parser.add_argument(
        "--patience", help="Patience to wait", default=4, type=int)
    parser.add_argument(
        "--batch_size", help="Batch size", default=16, type=int)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.model, args.lr, args.optimizer, args.epoch, args.patience,
        args.batch_size)
