import os
import random
import sys


## Package
import glob
import keras
import IPython.display as ipd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as py
import plotly.tools as tls
import seaborn as sns
import scipy.io.wavfile
import tensorflow as tf



## Keras
from keras import regularizers
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras.callbacks import  History, ReduceLROnPlateau, CSVLogger
from keras.models import Model, Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from keras.utils import to_categorical


## Sklearn
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit


## Rest
from scipy.fftpack import fft
from scipy import signal
from scipy.io import wavfile
from tqdm import tqdm

def cnn_model(db,path):
    df = pd.read_csv(db)
    df.set_index('fname', inplace=True)
    classes = list(np.unique(df.emotion))
    df.reset_index(inplace=True)

    # 5 class: angry, calm, sad, happy & fearful
    label5_list = []
    if db == 'EMOVO.csv':
        for i in range(len(df)):
            if df.emotion[i] == 'joy':
                lb = "joy"
            elif df.emotion[i] == 'neutrality':
                lb = "neutrality"
            elif df.emotion[i] == 'fear':
                lb = "fear"
            elif df.emotion[i] == 'anger':
                lb = "anger"
            elif df.emotion[i] == 'sadness':
                lb = "sadness"

            label5_list.append(lb)

    if db == 'TESS.csv':
        for i in range(len(df)):
            if df.emotion[i] == 'happy':
                lb = "joy"
            elif df.emotion[i] == 'neutral':
                lb = "neutrality"
            elif df.emotion[i] == 'fear':
                lb = "fear"
            elif df.emotion[i] == 'angry':
                lb = "anger"
            elif df.emotion[i] == 'sad':
                lb = "sadness"

            label5_list.append(lb)

    len(label5_list)
    df['label'] = label5_list
    df.head()
    print(df.label.value_counts().keys())

    # feature extraction
    data = pd.DataFrame(columns=['feature'])
    for i in tqdm(range(len(df))):
        X, sample_rate = librosa.load(path + df.fname[i])
        sample_rate = np.array(sample_rate)
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
        feature = mfccs
        data.loc[i] = [feature]
        data.head()
    df5 = pd.DataFrame(data['feature'].values.tolist())
    labels = df.label
    df5.head()
    newdf = pd.concat([df5, labels], axis=1)
    rnewdf = newdf.rename(index=str, columns={"0": "label"})
    rnewdf.isnull().sum().sum()
    rnewdf = rnewdf.fillna(0)
    rnewdf.head()

    def plot_time_series(data):
        """
        Plot the Audio Frequency.
        """
        fig = plt.figure(figsize=(14, 8))
        plt.title('Raw wave ')
        plt.ylabel('Amplitude')
        plt.plot(np.linspace(0, 1, len(data)), data)
        plt.show()

    def noise(data):
        """
        Adding White Noise.
        """
        # can take any distribution from https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.random.html
        noise_amp = 0.005 * np.random.uniform() * np.amax(data)
        data = data.astype('float64') + noise_amp * np.random.normal(size=data.shape[0])
        return data

    def shift(data):
        """
        Random Shifting.
        """
        s_range = int(np.random.uniform(low=-5, high=5) * 500)
        return np.roll(data, s_range)

    def stretch(data, rate=0.8):
        """
        Streching the Sound.
        """
        data = librosa.effects.time_stretch(data, rate)
        return data

    def pitch(data, sample_rate):
        """
        Pitch Tuning.
        """
        bins_per_octave = 12
        pitch_pm = 2
        pitch_change = pitch_pm * 2 * (np.random.uniform())
        data = librosa.effects.pitch_shift(data.astype('float64'),
                                           sample_rate, n_steps=pitch_change,
                                           bins_per_octave=bins_per_octave)
        return data

    def dyn_change(data):
        """
        Random Value Change.
        """
        dyn_change = np.random.uniform(low=1.5, high=3)
        return (data * dyn_change)

    def speedNpitch(data):
        """
        peed and Pitch Tuning.
        """
        # you can change low and high here
        length_change = np.random.uniform(low=0.8, high=1)
        speed_fac = 1.0 / length_change
        tmp = np.interp(np.arange(0, len(data), speed_fac), np.arange(0, len(data)), data)
        minlen = min(data.shape[0], tmp.shape[0])
        data *= 0
        data[0:minlen] = tmp[0:minlen]
        return data

    # Augmentation Method 1

    syn_data1 = pd.DataFrame(columns=['feature', 'label'])
    for i in tqdm(range(len(df))):
        X, sample_rate = librosa.load(path + df.fname[i])
        if df.label[i]:
            #     if data2_df.label[i] == "male_positive":
            X = noise(X)
            sample_rate = np.array(sample_rate)
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
            feature = mfccs
            a = random.uniform(0, 1)
            syn_data1.loc[i] = [feature, df.label[i]]

    # Augmentation Method 2

    syn_data2 = pd.DataFrame(columns=['feature', 'label'])
    for i in tqdm(range(len(df))):
        X, sample_rate = librosa.load(path + df.fname[i])
        if df.label[i]:
            X = pitch(X, sample_rate)
            sample_rate = np.array(sample_rate)
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
            feature = mfccs
            a = random.uniform(0, 1)
            syn_data2.loc[i] = [feature, df.label[i]]

    print(len(syn_data1), len(syn_data2))
    syn_data1 = syn_data1.reset_index(drop=True)
    syn_data2 = syn_data2.reset_index(drop=True)

    df4 = pd.DataFrame(syn_data1['feature'].values.tolist())
    labels4 = syn_data1.label
    syndf1 = pd.concat([df4, labels4], axis=1)
    syndf1 = syndf1.rename(index=str, columns={"0": "label"})
    syndf1 = syndf1.fillna(0)

    df4 = pd.DataFrame(syn_data2['feature'].values.tolist())
    labels4 = syn_data2.label
    syndf2 = pd.concat([df4, labels4], axis=1)
    syndf2 = syndf2.rename(index=str, columns={"0": "label"})
    syndf2 = syndf2.fillna(0)
    len(syndf2)

    combined_df = pd.concat([rnewdf, syndf1, syndf2], ignore_index=True)
    combined_df = combined_df.fillna(0)

    X = combined_df.drop(['label'], axis=1)
    y = combined_df.label
    xxx = StratifiedShuffleSplit(1, test_size=0.2, random_state=12)

    for train_index, test_index in xxx.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    X_train.isna().sum().sum()
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    lb = LabelEncoder()
    y_train = np_utils.to_categorical(lb.fit_transform(y_train))
    y_test = np_utils.to_categorical(lb.fit_transform(y_test))

    x_traincnn = np.expand_dims(X_train, axis=2)
    x_testcnn = np.expand_dims(X_test, axis=2)

    from keras import backend as K

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def get_lr_metric(optimizer):
        def lr(y_true, y_pred):
            return optimizer.lr

        return lr

    model = Sequential()
    model.add(Conv1D(256, 8, padding='same', input_shape=(X_train.shape[1], 1)))
    model.add(Activation('relu'))
    model.add(Conv1D(256, 8, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(MaxPooling1D(pool_size=(8)))
    model.add(Conv1D(128, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 8, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(MaxPooling1D(pool_size=(8)))
    model.add(Conv1D(64, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(64, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    # Edit according to target class no.
    model.add(Dense(5))
    model.add(Activation('softmax'))
    opt = keras.optimizers.SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=20, min_lr=0.000001)
    if lr_reduce == 0.000001:
        print('\n ############ hit minimum lr of 0.000001 ############ \n')
    # Please change the model name accordingly.
    mcp_save = ModelCheckpoint('aug_noiseNshift_5class_np.h5', save_best_only=True, monitor='val_loss', mode='min')
    cnnhistory = model.fit(x_traincnn, y_train, batch_size=16, epochs=10, validation_data=(x_testcnn, y_test),
                           callbacks=[mcp_save, lr_reduce])
