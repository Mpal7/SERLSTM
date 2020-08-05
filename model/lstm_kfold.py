
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


mean_signal_length = 32000
emovo_path_cleaned = r'F:\EMOVO/'
#emovo_path_cleaned = r'C:\Users\mp95\Desktop\EMOVO/'
db = 'EMOVO.CSV'
data_path = emovo_path_cleaned
class_labels = ("Sad", "Happy", "Angry", "Neutral")
#fp o sp
#"mfcc","deltas","formants","pitch","intensity"
features = ("mfcc")
splits = 5
augment = False
signal_mode = 'fp'
special_value = 100
routine_it = 5
epochs_n = 50

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

def signal_slicing_padding(signal):
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
    return signal

def get_feature_vector_from_mfcc(signal,fs,mfcc_len: int = 39) -> np.ndarray:
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
            if signal_mode == 'sp':
                signal = signal_slicing_padding(signal)
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


def train(x_train, y_train,x_test,y_test_train,model,acc,loss):
    es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=20)
    mc=ModelCheckpoint(emovo_path_cleaned+'best_epoch.h5', monitor='val_accuracy', mode='max', save_best_only=True,verbose=1)
    history=model.fit(x_train, y_train, batch_size=96, epochs=epochs_n,callbacks=[es,mc],validation_data=[x_test,y_test_train])
    #retrieve from history best acc and relative loss from early stopping
    best_epoch = np.argmax(history.history['val_accuracy']) + 1
    acc.append(history.history['val_accuracy'][best_epoch-1])
    loss.append(history.history['val_loss'][best_epoch-1])
    print('best epoch:',best_epoch, ' loss:',loss[-1],' acc:',acc[-1])
    print(loss,acc)


def lstm():
    Multiple_it_acc_te = []
    Multiple_it_loss_te = []
    counter = 0
    data, labels = get_data(data_path, class_labels=class_labels,
                            augment=augment)

    if signal_mode == 'fp':
        data = padding(data)

    print("\nEXECUTION PARAMETERS: {NUMBER OF FOLDERS: ",splits,"}-{NUMBER OF EPOCHS: ",epochs_n,"}-{NUMBER OF ROUTINE ITERATIONS: ",routine_it,"}-{BATCH SIZE : ",epochs_n,"}-{SIGNAL MODE: ",signal_mode,"}-{AUGMENT:",augment,"}-{FEATURES: ",features,"}-{EMOTIONS:",class_labels,"}")

    for i in range(0,5):
        it = 0
        # even if class are balanced we do not have many datapoints therefore i use stratifiedKFold
        kf = StratifiedKFold(n_splits=splits, shuffle=True)
        acc = []
        loss = []
        for i, (train_index, test_index) in enumerate(kf.split(data, labels)):
            print('#####FOLDER NUMBER: ' + str(it + 1))
            x_train, x_test = data[train_index], data[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            x_train = np.array(x_train)
            x_test = np.array(x_test)
            y_train = np.array(y_train)
            y_test = np.array(y_test)
            if it > 0:
                model.load_weights(emovo_path_cleaned + 'model.u5')
            y_train = np_utils.to_categorical(y_train)
            y_test_train = np_utils.to_categorical(y_test)
            if it == 0:
                print('Starting LSTM')
                model = Sequential()
                input_shape = x_train[0].shape
                if signal_mode == 'fp':
                    model.add(Masking(mask_value=special_value, input_shape=(None, input_shape[1])))
                model.add(KERAS_LSTM(128,input_shape=(input_shape[0], input_shape[1])))
                model.add(Dropout(0.5))
                model.add(Dense(32, activation='tanh'))
                model.add(Dense(len(class_labels), activation='softmax'))
                model.compile(loss='categorical_crossentropy', optimizer='adam',
                              metrics=['accuracy'])
                print(model.summary(), file=sys.stderr)
                model.save_weights(emovo_path_cleaned + 'model.u5')
            it += 1
            train(x_train, y_train, x_test, y_test_train, model, acc, loss)
            best_epoch = load_model(emovo_path_cleaned + 'best_epoch.h5')
            evaluate(best_epoch, x_test, y_test)
        print('\n\n ############# AVERAGE EVALUATION ############')
        print("######### MEAN LOSS OVER " + str(splits) + " FOLDERS: " + str(np.mean(loss)) + "  ###########")
        print("######### MEAN ACCURACY OVER " + str(splits) + " FOLDERS: " + str(np.mean(acc)) + "  ###########")
        Multiple_it_acc_te.append(np.mean(loss))
        Multiple_it_loss_te.append(np.mean(acc))
        counter = counter+1
    print("####LOSS OVER ",counter," ITERATIONS: ", np.mean(Multiple_it_loss_te), " ##### ACC OVER ",counter," ITERATIONS:",
          np.mean(Multiple_it_acc_te))


lstm()