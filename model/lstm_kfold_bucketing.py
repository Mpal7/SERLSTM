
from sklearn.utils import shuffle
import os
import librosa
import scipy.io.wavfile as wav
from speechpy.feature import mfcc
from keras.utils import np_utils
from typing import Tuple
import numpy
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import sys
import numpy as np
from keras import Sequential
from keras.layers import LSTM as KERAS_LSTM, Dense, Dropout,Masking
from preprocessing.data_augmentation import pitch_augmentation,noise,shift
from tqdm import tqdm
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.models import load_model

Kfold = 5
special_value = 100
emovo_path_cleaned = r'F:\EMOVO/'
class_labels = ("Sad", "Happy", "Angry", "Neutral")
bucketing = False
augment = True

def get_feature_vector_from_mfcc(signal,fs,mfcc_len: int = 39) -> np.ndarray:
    s_len = len(signal)
    # pad the signals to have same size if lesser than required
    # else slice them
    mel_coefficients = mfcc(signal, fs, num_cepstral=mfcc_len)
    mel_coefficients = concatenate_test_feature(mel_coefficients)
    return mel_coefficients

def concatenate_test_feature(data):
    delta1 = librosa.feature.delta(data)
    concatenated=np.concatenate((data, delta1), axis=1)
    permutation = []
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
        os.chdir(data_path+directory)
        for filename in tqdm(os.listdir()):
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
def padding(X):
    # Padding
    max_seq_len = 0
    for e in X:
        if e.shape[0] > max_seq_len:
            max_seq_len = e.shape[0]
    X_pad = np.full((len(X), max_seq_len, X[0].shape[1]), fill_value=special_value)
    for s, x in enumerate(X):
        seq_len = x.shape[0]
        X_pad[s, 0:seq_len, :] = x
    return X_pad


def bucketing_padding(X, y):
    # sorting and bucketing X, to keep correct y permutation i use a lenx[X,y] array
    curr = 0
    # sorting descending for bucketing
    Xy_paired = [list(pair) for pair in zip(X, y)]

    def takeSecond(elem):
        return len(elem[0])

    Xy_paired.sort(reverse=True, key=takeSecond)

    X_sorted = []
    y_sorted = []

    for i in Xy_paired:
        X_sorted.append(i[0])
        y_sorted.append(i[1])

    Xpad = []
    # bucketing and padding for X
    while (curr + batch_size < len(X)):
        max_seq_len = 0
        for e in X_sorted[curr:curr + batch_size]:
            if e.shape[0] > max_seq_len:
                max_seq_len = e.shape[0]
        Xpad_local = np.full((batch_size, max_seq_len, X[0].shape[1]), fill_value=special_value)
        for s, x in enumerate(X_sorted[curr:batch_size]):
            seq_len = x.shape[0]
            Xpad_local[s, 0:seq_len, :] = x
        curr += batch_size
        Xpad.append(Xpad_local)
    if curr != len(X_sorted):
        max_seq_len = 0
        for e in X_sorted[curr:len(X)]:
            if e.shape[0] > max_seq_len:
                max_seq_len = e.shape[0]
        Xpad_local = np.full(((len(X) - curr), max_seq_len, X[0].shape[1]), fill_value=special_value)
        for s, x in enumerate(X_sorted[curr:len(X)]):
            seq_len = x.shape[0]
            Xpad_local[s, 0:seq_len, :] = x
        Xpad.append(Xpad_local)

    # reshape y for batches
    curr = 0
    y_batches = []
    while (curr + batch_size < len(y_sorted)):
        y_local = []
        [y_local.append(i) for i in y_sorted[curr:curr + batch_size]]
        curr += batch_size
        y_batches.append(y_local)
    if curr != len(y_sorted):
        y_local = []
        [y_local.append(i) for i in y_sorted[curr:len(y_sorted)]]
        y_batches.append(y_local)

    return Xpad, y_batches


def k_fold_cross_validation(model, batch_size, x_size, class_number):
    kf = StratifiedKFold(n_splits=Kfold,shuffle=True)
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    for i, (train_index, test_index) in enumerate(kf.split(X, y)):
        print("\n##### FOLDER NUMBER ", i + 1, " #######")
        X_train = []
        X_test = []
        [X_train.append(X[i]) for i in train_index]
        [X_test.append(X[i]) for i in test_index]
        y_train, y_test = y[train_index], y[test_index]
        y_train = np_utils.to_categorical(y_train)
        y_test_train = np_utils.to_categorical(y_test)
        if batch_size == int(x_size - (x_size / Kfold)) or bucketing == False:
            X_train = padding(X_train)
            X_test = padding(X_test)
            es = EarlyStopping(monitor='val_loss', mode='min',verbose=1,patience=15)
            mc = ModelCheckpoint(emovo_path_cleaned + 'best_epoch_pad.h5', monitor='val_loss', mode='min',
                                 save_best_only=True, verbose=1)
            history = model.fit(X_train, y_train, epochs=50, batch_size=batch_size, validation_data=(X_test, y_test_train),callbacks=[es,mc])
            best_epoch = np.argmin(history.history['val_loss']) + 1
            train_loss.append(history.history['loss'][best_epoch - 1])
            train_acc.append(history.history['accuracy'][best_epoch - 1])
            test_loss.append(history.history['val_loss'][best_epoch - 1])
            test_acc.append(history.history['val_accuracy'][best_epoch - 1])
            print('best epoch:', best_epoch, ' loss:', test_loss[-1], ' acc:', test_acc[-1])
            print(test_loss, test_acc)
        else:
            if batch_size != 1 and bucketing:
                X_train, y_train = bucketing_padding(X_train, y_train)
            model_train_task, report_loss_train, report_acc_train, report_loss_test, report_acc_test = train_test_task(
                model, X_train, y_train, X_test, y_test, 50, batch_size, class_number)
            train_loss.append(report_loss_train)
            train_acc.append(report_acc_train)
            test_loss.append(report_loss_test)
            test_acc.append(report_acc_test)

        model.load_weights('F:\EMOVO/model_pad.u5')  # reset weights for next folder
        best_epoch = load_model(emovo_path_cleaned + 'best_epoch_pad.h5')
        evaluate(best_epoch,X_test,y_test)
    avg_loss_train = np.mean(train_loss)
    avg_acc_train = np.mean(train_acc)
    avg_loss_test = np.mean(test_loss)
    avg_acc_test = np.mean(test_acc)
    print(f'acc_train: {avg_acc_train} loss_train: {avg_loss_train}')
    print(f'acc_test: {avg_acc_test} loss_test: {avg_loss_test}')

    return avg_loss_train,avg_acc_train,avg_loss_test,avg_acc_test


def train_test_task(model, X_train, y_train, X_test, y_test, EPOCH, batch_size, class_number):
    # single zero padding for NULL datas
    report_acc = 0.0
    for epoch in range(EPOCH):
        print(f'Training epoch {epoch} ...')
        avg_train_loss = 0.0
        avg_train_acc = 0.0
        avg_test_loss = 0.0
        avg_test_acc = 0.0

        print("Training")
        for sample_i in tqdm(range(int(len(X_train))), position=0, leave=True):
            if batch_size != 1:
                [train_loss, train_acc] = model.train_on_batch(X_train[sample_i], np.array(y_train[sample_i]).reshape(
                    len(y_train[sample_i]), class_number))
            else:
                [train_loss, train_acc] = model.train_on_batch(
                    X_train[sample_i].reshape(1, X_train[sample_i].shape[0], X_train[sample_i].shape[1]),
                    y_train[sample_i].reshape(1, class_number))
            avg_train_loss += train_loss / (len(X_train))
            avg_train_acc += train_acc / (len(X_train))
        shuffle(X_train, y_train)
        report_train_acc = avg_train_acc
        report_train_loss = avg_train_loss

        print("Testing")
        for sample_i in tqdm(range(int(len(X_test))), position=0, leave=True):
            [test_loss, test_acc] = model.test_on_batch(
                X_test[sample_i].reshape(1, X_test[sample_i].shape[0], X_test[sample_i].shape[1]),
                y_test[sample_i].reshape(1, class_number))
            avg_test_loss += test_loss / (len(X_test))
            avg_test_acc += test_acc / (len(X_test))
        report_test_acc = avg_test_acc
        report_test_loss = avg_test_loss

        print(
            f'acc_train: {avg_train_acc} loss_train: {avg_train_loss} acc_test: {avg_test_acc} loss_test: {avg_test_loss}')

    return model, report_train_loss, report_train_acc, report_test_loss, report_test_acc


Multiple_it_acc_te = []
Multiple_it_loss_te = []

for i in range(0,5):
    print("####ITERATION NUMBER:",i+1," ######")
    X, y = get_data(emovo_path_cleaned, class_labels, augment, mfcc_len=39)
    n_features = X[0].shape[1]
    full_batch = int(len(X) - (len(X) / Kfold))
    batch_size = 32
    model = Sequential()
    model.add(Masking(mask_value=special_value, input_shape=(None, n_features)))
    model.add(KERAS_LSTM(128, input_shape=(None, n_features)))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(len(class_labels), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.save_weights('F:\EMOVO/model_pad.u5')
    avg_l_tr,avg_a_tr,avg_l_te,avg_a_te=k_fold_cross_validation(model, batch_size, len(X), len(class_labels))
    Multiple_it_loss_te.append(avg_l_te)
    Multiple_it_acc_te.append(avg_a_te)

print("####LOSS OVER 10 ITERATIONS: ",np.mean(Multiple_it_loss_te)," ##### ACC OVER 10 ITERATIONS:",np.mean(Multiple_it_acc_te))
