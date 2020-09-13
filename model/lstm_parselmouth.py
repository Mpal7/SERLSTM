import parselmouth
import os
import librosa
import scipy.io.wavfile as wav
from speechpy.feature import mfcc
from keras.utils import np_utils
from typing import Tuple
import numpy
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout,Masking
from tqdm import tqdm
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.models import load_model
from parselmouth.praat import call
import sys
from spafe.features.lpc import  lpc, lpcc


#parameters
mean_signal_length = 32000
data_path = r'F:\EMOVO/'
class_labels = ("Sad", "Happy", "Angry", "Neutral")
#parselmouth can be used only with full padding without altering original files
#fp o sp, beware in sp only mfcc are functioning
#"mfcc","deltas","formants","pitch","intensity"
features = ("mfcc")
splits = 5
signal_mode = 'fp'
special_value = 100
routine_it = 10
epochs_n = 80


def get_feature_vector_from_formants(filepath,feature_vector):
    path = (filepath)
    sound = parselmouth.Sound(path)  # read the sound
    pitch = sound.to_pitch_ac()
    meanF0 = call(pitch, "Get mean", 0, 0, "Hertz")  # get mean pitch
    if meanF0 > 150:
        maxFormant = 5500  # women
    else:
        maxFormant = 5000  # men
    formant = call(sound, "To Formant (burg)", 0.02, 5, maxFormant, 0.01, 50)    # even if i need 3 formants i calculate 5, it behaves better apparently
    local_formant = []
    it = 0
    for x in formant.xs():
        for f in range(1, 4):
            local_formant.append(formant.get_value_at_time(f, x))
    formant_array = np.reshape(local_formant, (formant.get_number_of_frames(), 3))
    #padding array to match first feature in features array length with special masking value
    if features.index('formants') is not 0:
        if formant_array.shape[0] < feature_vector.shape[0]:
            formant_array = np.pad(formant_array, ((0, feature_vector.shape[0] - formant_array.shape[0]), (0, 0)),
                                   'constant', constant_values=(0, 100))
        elif formant_array.shape[0] > feature_vector.shape[0]:
            pad_len = formant_array.shape[0] - feature_vector.shape[0]
            pad_len //= 2
            formant_array = formant_array[pad_len:pad_len + feature_vector.shape[0]]
    #standardization
    for i in range(0, 3):
        formant_array[:, i] = formant_array[:, i] / np.linalg.norm(formant_array[:, i])
    if features.index('formants') is not 0:
        formant_array = np.concatenate((feature_vector, formant_array), axis=1)
    return formant_array

def get_feature_vector_from_pitch(filepath,feature_vector):
    path = (filepath)
    signal = parselmouth.Sound(path)
    #compare with pitch = sound.to_pitch_ac(time_step = 0.01,pitch_floor=150,very_accurate=True)
    pitch = signal.to_pitch_ac()
    x_sample = pitch.selected_array['frequency']
    x_sample = x_sample/np.linalg.norm(x_sample)
    if features.index('pitch') is not 0:
        if len(x_sample) < feature_vector.shape[0]:
            x_sample = np.pad(x_sample, ((0, feature_vector.shape[0] - len(x_sample))), 'constant', constant_values=100)
        elif len(x_sample) > feature_vector.shape[0]:
            pad_len = len(x_sample) - feature_vector.shape[0]
            pad_len //= 2
            x_sample = x_sample[pad_len:pad_len + feature_vector.shape[0]]
    x_sample = np.reshape(x_sample, (len(x_sample), 1))
    if features.index('pitch') is not 0:
        x_sample = np.concatenate((feature_vector,x_sample),axis=1)
    return x_sample

def get_feature_vector_from_intensity(filepath,feature_vector):
    path = (filepath)
    signal = parselmouth.Sound(path)
    x_intensity_local =[]
    intensity = signal.to_intensity(time_step=0.01, minimum_pitch=150)
    for x in intensity.xs():
        x_intensity_local.append(intensity.get_value(time=x))
    x_sample = np.array(x_intensity_local)
    x_sample = x_sample / np.linalg.norm(x_sample)
    if features.index('intensity') is not 0:
        if len(x_sample) < feature_vector.shape[0]:
            x_sample = np.pad(x_sample, ((0, feature_vector.shape[0] - len(x_sample))), 'constant', constant_values=100)
        if len(x_sample) > feature_vector.shape[0]:
            pad_len = len(x_sample) - feature_vector.shape[0]
            pad_len //= 2
            x_sample = x_sample[pad_len:pad_len + feature_vector.shape[0]]
    x_sample = np.reshape(x_sample, (len(x_sample), 1))
    if features.index('intensity') is not 0:
        x_sample = np.concatenate((feature_vector, x_sample), axis=1)
    return x_sample

def get_feature_vector_from_mfcc(signal,fs):
    #window 0.2 , stride 0.1
    mel_coefficients = mfcc(signal, fs, frame_stride=0.1,num_cepstral=13)
    mel_coefficients_for_deltas = mel_coefficients
    return mel_coefficients,mel_coefficients_for_deltas

def get_feature_vector_from_deltas(data):
    delta1 = librosa.feature.delta(data)
    concatenated=np.concatenate((data, delta1), axis=1)
    permutation = []
    for i in range(0, int(concatenated.shape[1] / 2)):
        permutation.append(i)
        permutation.append(13 + i)
    permutation = np.array(permutation)
    feature_vector = concatenated[:, permutation]
    return feature_vector

def get_feature_vector_from_lpcc(signal,fs,feature_vector):
    # compute lpccs
    x_sample = lpcc(sig=signal, fs=fs, win_len=0.02, win_hop=0.01, num_ceps=13, lifter=0, normalize=True)
    if features.index('lpcc') is not 0:
        if x_sample.shape[0] < feature_vector.shape[0]:
            x_sample = np.pad(x_sample, ((0, feature_vector.shape[0] - x_sample.shape[0]), (0, 0)),
                                   'constant', constant_values=(0, 100))
        if x_sample.shape[0] > feature_vector.shape[0]:
            pad_len = x_sample.shape[0] - feature_vector.shape[0]
            pad_len //= 2
            x_sample = x_sample[pad_len:pad_len + feature_vector.shape[0]]
    if features.index('lpcc') is not 0:
        x_sample = np.concatenate((feature_vector, x_sample), axis=1)
    return x_sample


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
                        'constant', constant_values=100)
    else:
        pad_len = s_len - mean_signal_length
        pad_len //= 2
        signal = signal[pad_len:pad_len + mean_signal_length]
    return signal


def get_data(data_path: str,class_labels: Tuple) -> \
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
            feature_vector=[]
            if signal_mode == 'sp':
                signal = signal_slicing_padding(signal)
            if 'mfcc' in features:
                if features.index('mfcc') is not 0:
                    sys.exit("\n###### please put mfcc as the first element of the feature array, aborting execution... ######")
                else:
                    feature_vector, mel_coefficients = get_feature_vector_from_mfcc(signal, fs)
            if 'lpcc' in features:
                feature_vector = get_feature_vector_from_lpcc(signal,fs,feature_vector)
            if 'deltas' in features:
                if 'mfcc' in features:
                    feature_vector = get_feature_vector_from_deltas(mel_coefficients)
                else:
                    sys.exit("\n ###### can't compute deltas without mfcc, aborting execution ######")
            if 'formants' in features:
                feature_vector = get_feature_vector_from_formants(filepath,feature_vector)
            if 'pitch' in features:
                feature_vector = get_feature_vector_from_pitch(filepath, feature_vector)
            if 'intensity' in features:
                feature_vector = get_feature_vector_from_intensity(filepath,feature_vector)
            data.append(feature_vector)
            labels.append(i)
            names.append(filename)
            names.append(filename)
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
    print('Accuracy:%.3f\n' % accuracy_score(y_pred=predictions,
                                             y_true=y_test))
    print('Confusion matrix:\n', confusion_matrix(y_pred=predictions,
                                                y_true=y_test))


def train(x_train, y_train,x_test,y_test_train,model,acc,loss):
    es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=20)
    mc=ModelCheckpoint('best_epoch.h5', monitor='val_accuracy', mode='max', save_best_only=True,verbose=1)
    history=model.fit(x_train, y_train, batch_size=96, epochs=epochs_n,callbacks=[es,mc],validation_data=(x_test,y_test_train))
    #retrieve from history best acc and relative loss from early stopping
    best_epoch = np.argmax(history.history['val_accuracy']) + 1
    acc.append(history.history['val_accuracy'][best_epoch-1])
    loss.append(history.history['val_loss'][best_epoch-1])
    print('best epoch:',best_epoch, ' loss:',loss[-1],' acc:',acc[-1])
    print(loss,acc)


def lstm():
    Multiple_it_acc_te = []
    Multiple_it_loss_te = []
    counter = 1
    data, labels = get_data(data_path, class_labels=class_labels)

    if signal_mode == 'fp':
        data = padding(data)

    print("\nEXECUTION PARAMETERS: {NUMBER OF FOLDERS: ",splits,"}-{NUMBER OF EPOCHS: ",epochs_n,"}-{NUMBER OF ROUTINE ITERATIONS: ",routine_it,"}-{BATCH SIZE : ",96,"}-{SIGNAL MODE: ",signal_mode,"}-{AUGMENT:",class_labels[0],"}-{FEATURES: ",features,"}-{EMOTIONS:",class_labels,"}")

    for i in range(0,routine_it):
        print("\n####ITERATION NUMBER: ",i+1)
        it = 0
        # even if class are balanced we do not have many datapoints therefore i use stratifiedKFold
        kf = StratifiedKFold(n_splits=splits, shuffle=True)
        acc = []
        loss = []
        for i, (train_index, test_index) in enumerate(kf.split(data, labels)):
            print('\n#####FOLDER NUMBER: ' + str(it + 1))
            x_train, x_test = data[train_index], data[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            x_train = np.array(x_train)
            x_test = np.array(x_test)
            y_train = np.array(y_train)
            y_test = np.array(y_test)
            if it > 0:
                model.load_weights('model.u5')
            y_train = np_utils.to_categorical(y_train)
            y_test_train = np_utils.to_categorical(y_test)
            if it == 0:
                print('Starting LSTM')
                model = Sequential()
                input_shape = x_train[0].shape
                model.add(Masking(mask_value=special_value, input_shape=(input_shape[0], input_shape[1])))
                model.add(LSTM(128,input_shape=(input_shape[0], input_shape[1]),return_sequences=True))
                model.add(Dropout(0.5))
                model.add(LSTM(64))
                model.add(Dropout(0.5))
                model.add(Dense(32, activation='tanh'))
                model.add(Dense(len(class_labels), activation='softmax'))
                model.compile(loss='categorical_crossentropy', optimizer='adam',
                              metrics=['accuracy'])
                print(model.summary(), file=sys.stderr)
                model.save_weights('model.u5')
            it += 1
            train(x_train, y_train, x_test, y_test_train, model, acc, loss)
            best_epoch = load_model('best_epoch.h5')
            evaluate(best_epoch, x_test, y_test)
        print('\n\n ############# AVERAGE EVALUATIONS ############')
        print("\n######### MEAN LOSS OVER THE " + str(splits) + " FOLDERS: " + str(np.mean(loss)) + "  ###########")
        print("######### MEAN ACCURACY OVER THE " + str(splits) + " FOLDERS: " + str(np.mean(acc)) + "  ###########")
        print(
            "######### STANDARD DEVIATION OVER THE " + str(splits) + " FOLDERS: " + str(np.std(loss)) + "  ###########")
        print(
            "######### STANDARD DEVIATION OVER THE " + str(splits) + " FOLDERS: " + str(np.std(acc)) + "  ###########")
        Multiple_it_acc_te.append(np.mean(acc))
        Multiple_it_loss_te.append(np.mean(loss))
        print("\n####MEAN LOSSES OF THE FIRST ", counter+1, " ITERATIONS: ", Multiple_it_loss_te,
              " MEAN ACC OF THE FIRST ", counter+1, " ITERATIONS: ",
              Multiple_it_acc_te, " #####")
        acc_std = np.std(Multiple_it_acc_te)
        loss_std = np.std(Multiple_it_loss_te)
        print("####LOSSES STANDARD DEVIATIONS OF THE FIRST ", counter+1, " ITERATIONS: ", acc_std,
              " ACC STANDARD DEVIATIONS OF THE FIRST ", counter+1,
              " ITERATIONS: ", loss_std, " #####")
        counter = counter + 1
    print("\n####LOSS OVER ", counter, " ITERATIONS: ", np.mean(Multiple_it_loss_te), " ##### ACC OVER ", counter,
          " ITERATIONS:",
          np.mean(Multiple_it_acc_te), " #####")
    print("####LOSS STANDARD DEVIATION OVER ", counter, " ITERATIONS: ", np.std(Multiple_it_loss_te),
          " ACC STANDARD DEVIATION OVER ", counter, " ITERATIONS:",
          np.std(Multiple_it_acc_te), "#####")

lstm()