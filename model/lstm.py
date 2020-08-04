import os
import scipy.io.wavfile as wav
from speechpy.feature import mfcc
from keras.utils import np_utils
from typing import Tuple
import numpy
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import sys
import numpy as np
from keras import Sequential
from keras.layers import LSTM as KERAS_LSTM, Dense, Dropout
from preprocessing.data_augmentation import pitch_augmentation,noise,shift
from keras.callbacks import EarlyStopping
import librosa
mean_signal_length = 32000

data_path_emodb = r'F:\EMODB\wav/'
emovo_path_cleaned = r'C:\Users\mp95\Desktop\EMOVO/'
_DATA_PATH = emovo_path_cleaned
_CLASS_LABELS = ("Sad","Happy","Angry","Neutral")


def get_feature_vector_from_mfcc(signal,fs,mfcc_len: int = 39) -> np.ndarray:
    s_len = len(signal)
    # pad the signals to have same size if lesser than required
    # else slice them
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
    mel_coefficients = concatenate_test_feature(mel_coefficients)
    return mel_coefficients


def concatenate_test_feature(data):
    delta1 = librosa.feature.delta(data)
    concatenated=np.concatenate((data, delta1), axis=1)
    permutation = []
    #reorder mfcc1,delta1,...,mfccN,deltaN
    for i in range(0, int(concatenated.shape[1] / 2)):
        permutation.append(i)
        permutation.append(39 + i)
    permutation = np.array(permutation)
    feature_vector = concatenated[:, permutation]
    return feature_vector


def get_data(data_path: str,class_labels: Tuple, augment: bool, mfcc_len: int = 39) -> \
        Tuple[np.ndarray, np.ndarray]:
    data = []
    labels = []
    names = []
    cur_dir = os.getcwd()
    sys.stderr.write('curdir: %s\n' % cur_dir)
    os.chdir(data_path)
    for i, directory in enumerate(class_labels):
        sys.stderr.write("started reading folder %s\n" % directory)
        os.chdir(directory)
        for filename in os.listdir('..'):
            filepath = os.getcwd() + '/' + filename
            fs, signal = wav.read(filepath)
            feature_vector = get_feature_vector_from_mfcc(signal,fs,
                                                          mfcc_len=mfcc_len)
            data.append(feature_vector)
            labels.append(i)
            names.append(filename)
            if augment:
                signal_pitch = pitch_augmentation(signal, fs)
                signal_noise_pitch = noise(signal_pitch)
                feature_vector = get_feature_vector_from_mfcc(signal_noise_pitch, fs,
                                                              mfcc_len=mfcc_len)
                data.append(feature_vector)
                labels.append(i)
                signal_pitch = pitch_augmentation(signal,fs)
                signal_shift_pitch = shift(signal_pitch)
                feature_vector = get_feature_vector_from_mfcc(signal_shift_pitch, fs,
                                                              mfcc_len=mfcc_len)
                data.append(feature_vector)
                labels.append(i)
        sys.stderr.write("ended reading folder %s\n" % directory)
        os.chdir('../..')
    os.chdir(cur_dir)
    return np.array(data), np.array(labels)


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
    print('Confusion matrix:\n', confusion_matrix(y_pred=predictions,
                                                y_true=y_test))


def train(x_train, y_train,x_test, y_test_train,model, n_repetitions=5):
    acc = []
    loss = []
    for i in range(n_repetitions):
        model.load_weights('model.u5')
        p = np.random.permutation(len(x_train))
        x_train = x_train[p]
        y_train = y_train[p]
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        model.fit(x_train, y_train, batch_size=32, epochs=30,callbacks=[es])
        loss, acc_l = model.evaluate(x_test, y_test_train,batch_size = 32)
        print("## TEST ACCURACY: "+str(acc_l)+" ## TEST LOSS: "+str(loss))
        acc.append(acc_l)
    print("######### MEAN ACCURACY OVER "+str(n_repetitions)+" REPETITIONS: "+str(np.mean(acc))+"###########")


def extract_data(augment):
    data, labels = get_data(_DATA_PATH,class_labels=_CLASS_LABELS,
                            augment=augment)
    x_train, x_test, y_train, y_test = train_test_split(
        data,
        labels,
        test_size=0.2,
        random_state=42)
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(
        y_test), len(_CLASS_LABELS)

def lstm():
    augment = False
    x_train, x_test, y_train, y_test, num_labels = extract_data(
        augment=augment)
    y_train = np_utils.to_categorical(y_train)
    y_test_train = np_utils.to_categorical(y_test)
    print('Starting LSTM')
    model = Sequential()
    input_shape = x_train[0].shape
    model.add(
        KERAS_LSTM(128,
                   input_shape=(input_shape[0], input_shape[1])))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='tanh'))
    model.add(Dense(num_labels, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary(), file=sys.stderr)
    model.save_weights('model.u5')
    train(x_train, y_train,x_test, y_test_train, model, n_repetitions=5)
    evaluate(model, x_test, y_test)
    print('\n\nmodel evaluation')

lstm()