import pickle
import numpy as np
from source.rel_path import rootDir
import os
from scipy.stats import describe
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix
from source.keras_function import *
from source.data_standardization import DataStandardizer
import pandas
import keras
from keras import models, layers, regularizers

with open(os.path.join(rootDir, 'data', 'train_merged.pickle'), 'rb') as inFile:
    xs, ys, ids = pickle.load(inFile)

# assign greater weights to positive samples to fix the balance
num0s = np.count_nonzero(ys == 0)
num1s = np.count_nonzero(ys == 1)
oneWeight = num0s / num1s
weightArray = np.zeros(ys.shape, np.float)
weightArray[ys == 0] = 1.0
weightArray[ys == 1] = oneWeight

dataStandardizer = DataStandardizer()
dataStandardizer.fit(xs)
newXs = dataStandardizer.transform(xs)

onehot = OneHotEncoder()
newYs = np.asarray(onehot.fit_transform(ys.reshape(-1, 1)).todense())

dropoutRate = 0.3
regRate = 0.0

def get_model():
    model = models.Sequential()
    model.add(layers.Conv3D(20, (2, 2, 2), data_format='channels_last', input_shape=newXs.shape[1:], activity_regularizer=regularizers.l2(regRate)))
    model.add(layers.Dropout(dropoutRate))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv3D(32, (2, 2, 2), data_format='channels_last', activity_regularizer=regularizers.l2(regRate)))
    model.add(layers.Dropout(dropoutRate))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv3D(64, (3, 3, 3), data_format='channels_last', activity_regularizer=regularizers.l2(regRate)))
    model.add(layers.Dropout(dropoutRate))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling3D())
    model.add(layers.Flatten())
    model.add(layers.Dropout(dropoutRate))
    model.add(layers.Dense(2, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=8.0e-4),
                  metrics=[auc_roc])

    return model



def train_kfold():

    histories = []

    kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=0x12345678)

    for train_idx, test_idx in kfold.split(newXs, ys):
        x_train, y_train = newXs[train_idx], newYs[train_idx]
        x_train_weight = weightArray[train_idx]
        x_test, y_test = newXs[test_idx], newYs[test_idx]

        aucCallback = PreciseAUC(x_train, y_train, x_test, y_test)

        model = get_model()
        history = model.fit(x_train, y_train, batch_size=20, epochs=300,
                  validation_data=(x_test, y_test), sample_weight=x_train_weight,
                  callbacks=[aucCallback])

        y_test_true = np.argmax(y_test, axis=1)
        y_test_pred = model.predict_proba(x_test)[:, 1]
        historyDict = history.history
        historyDict['y_test_true'] = y_test_true
        historyDict['y_test_pred'] = y_test_pred
        histories.append(history.history)

    with open(os.path.join(rootDir, 'data', 'train_history.pickle'), 'wb') as outFile:
        pickle.dump(histories, outFile)


def get_final_model_dataset(splitIds = False):
    if not splitIds:
        splits = train_test_split(newXs, weightArray, newYs, test_size=0.3, random_state=0x12345678, stratify=ys)
    else:
        splits = train_test_split(newXs, weightArray, newYs, ids, test_size=0.3, random_state=0x12345678, stratify=ys)
    return splits

def train_final_model():
    x_train, x_test, x_train_weights, _, y_train, y_test = get_final_model_dataset()
    aucCallback = PreciseAUC(x_train, y_train, x_test, y_test)
    model = get_model()
    history = model.fit(x_train, y_train, batch_size=20, epochs=300,
                  validation_data=(x_test, y_test),sample_weight=x_train_weights,
              callbacks=[aucCallback])
    y_test_true = np.argmax(y_test, axis=1)
    y_test_pred = model.predict_proba(x_test)[:, 1]
    historyDict = history.history
    historyDict['y_test_true'] = y_test_true
    historyDict['y_test_pred'] = y_test_pred

    model.save(os.path.join(rootDir, 'data', 'model.bin'))
    model.save_weights(os.path.join(rootDir, 'data', 'model_weights.bin'))

    with open(os.path.join(rootDir, 'data', 'final_model_history.pickle'), 'wb') as outFile:
        pickle.dump(historyDict, outFile)


if __name__ == '__main__':
    train_final_model()