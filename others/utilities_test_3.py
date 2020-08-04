import os
import sys
from typing import Tuple
import scipy.io.wavfile as wav
from speechpy.feature import mfcc
import pandas as pd
from tqdm import tqdm
from keras.utils import np_utils
from typing import Tuple
import numpy
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import sys
import numpy as np
from keras import Sequential
from keras.layers import LSTM as KERAS_LSTM, Dense, Dropout

mean_signal_length = 32000
data_path_emodb = r'F:\EMODB\wav/'
emovo_path_cleaned = r'F:\EMOVO\clean_5_vad_silence/'
db = 'EMOVO.CSV'
_DATA_PATH = emovo_path_cleaned
selected_emotions = ['Anger','Joy','Neutral']
number_of_labels = len(selected_emotions)


def get_feature_vector_from_mfcc(fs, signal,
                                 mfcc_len: int = 39) -> np.ndarray:
    s_len = len(signal)
    # padding and slicing signal to the mean
    if s_len < mean_signal_length:
        pad_len = mean_signal_length - s_len
        pad_rem = pad_len % 2
        pad_len //= 2
        signal = np.pad(signal, (pad_len, pad_len + pad_rem),
                        'constant', constant_values=0)
    else:
        pad_len = s_len - mean_signal_length
        pad_len //= 2
        signal = signal[pad_len:pad_len + mean_signal_length]
    mel_coefficients = mfcc(signal, fs, num_cepstral=mfcc_len)
    return mel_coefficients


def get_data(data_path, selected_emotions, mfcc_len):
    data = []
    labels = []
    names = []
    df = pd.read_csv(db)
    classes = list(np.unique(df.emotion))
    for f in tqdm(df.fname):
        label = df.loc[df.fname == f, 'emotion'].values[0]
        if label in selected_emotions:
            labels.append(classes.index(label))
            filepath = (data_path + f)
            fs, signal = wav.read(filepath)
            feature_vector = get_feature_vector_from_mfcc(fs, signal, mfcc_len=mfcc_len)
            data.append(feature_vector)
            names.append(f)
    return np.array(data), np.array(labels)


class Model(object):

    def __init__(self, save_path: str = 'F:\EMOVO\saves', name: str = 'test'):
        self.model = None
        self.save_path = save_path
        self.name = name
        self.trained = False

    def train(self, x_train: numpy.ndarray, y_train: numpy.ndarray,
              x_val: numpy.ndarray = None,
              y_val: numpy.ndarray = None) -> None:
        raise NotImplementedError()


def predict(model, samples: numpy.ndarray) -> Tuple:
    results = []
    for _, sample in enumerate(samples):
        results.append(np.argmax(model.predict(np.array([sample]))))
    return tuple(results)


def evaluate(model, x_test: numpy.ndarray, y_test: numpy.ndarray) -> None:
    predictions = predict(model, x_test)
    print(y_test)
    print(predictions)
    print('Accuracy:%.3f\n' % accuracy_score(y_pred=predictions,
                                             y_true=y_test))
    print('Confusion matrix:', confusion_matrix(y_pred=predictions,
                                                y_true=y_test))


def train(x_train, y_train, model, n_repetitions=1):
    best_acc = 0
    for i in range(n_repetitions):
        p = np.random.permutation(len(x_train))
        x_train = x_train[p]
        y_train = y_train[p]
        model.fit(x_train, y_train, batch_size=32, epochs=50)

def extract_data(selected_emotions):
    data, labels = get_data(_DATA_PATH,selected_emotions,mfcc_len=39)
    x_train, x_test, y_train, y_test = train_test_split(
        data,
        labels,
        test_size=0.2,
        random_state=42, stratify=labels)
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(
        y_test), number_of_labels


def get_feature_vector(file_path, flatten):
    return get_feature_vector_from_mfcc(file_path, flatten, mfcc_len=39)


def lstm(num_classes,selected_emotions):
    x_train, x_test, y_train, y_test, num_labels = extract_data(selected_emotions)
    y_train = np_utils.to_categorical(y_train)
    y_test_train = np_utils.to_categorical(y_test)
    print('Starting LSTM')
    model = Sequential()
    input_shape = x_train[0].shape
    model.add(
        KERAS_LSTM(128,
                   input_shape=(input_shape[0], input_shape[1])))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='tanh'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary(), file=sys.stderr)
    train(x_train, y_train, model, n_repetitions=1)
    print('\n\nmodel evaluation')
    evaluate(model,x_test,y_test)


lstm(number_of_labels,selected_emotions)
