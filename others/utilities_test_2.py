import os
from typing import Tuple
import numpy as np
import scipy.io.wavfile as wav
from speechpy.feature import mfcc



mean_signal_length = 32000  # Empirically calculated for the given data set

data_path_emovo = r'F:\EMODB/'

def get_feature_vector_from_mfcc(file_path: str, flatten: bool,
                                 mfcc_len: int = 39) -> np.ndarray:
    """
    Make feature vector from MFCC for the given wav file.
    Args:
        file_path (str): path to the .wav file that needs to be read.
        flatten (bool) : Boolean indicating whether to flatten mfcc obtained.
        mfcc_len (int): Number of cepestral co efficients to be consider.
    Returns:
        numpy.ndarray: feature vector of the wav file made from mfcc.
    """
    fs, signal = wav.read(file_path)
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
    if flatten:
        # Flatten the data
        mel_coefficients = np.ravel(mel_coefficients)
    return mel_coefficients


def get_data(data_path: str, flatten: bool = True, mfcc_len: int = 39,
             class_labels: Tuple = ("Neutral", "Angry", "Happy", "Sad")) -> \
        Tuple[np.ndarray, np.ndarray]:
    """Extract data for training and testing.
    1. Iterate through all the folders.
    2. Read the audio files in each folder.
    3. Extract Mel frequency cepestral coefficients for each file.
    4. Generate feature vector for the audio files as required.
    Args:
        data_path (str): path to the data set folder
        flatten (bool): Boolean specifying whether to flatten the data or not.
        mfcc_len (int): Number of mfcc features to take for each frame.
        class_labels (tuple): class labels that we care about.
    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: Two numpy arrays, one with mfcc and
        other with labels.
    """
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
            feature_vector = get_feature_vector_from_mfcc(file_path=filepath,
                                                          mfcc_len=mfcc_len,
                                                          flatten=flatten)
            data.append(feature_vector)
            labels.append(i)
            names.append(filename)
        sys.stderr.write("ended reading folder %s\n" % directory)
        os.chdir('../..')
    os.chdir(cur_dir)
    return np.array(data), np.array(labels)

from typing import Tuple

import numpy
from sklearn.metrics import accuracy_score, confusion_matrix

__author__ = 'harry-7'
__version__ = '1.1'


class Model(object):
    """
    Model is the abstract class which determines how a model should be.
    Any model inheriting this class should do the following.
    1.  Set the model instance variable to the corresponding model class which
        which will provide methods `fit` and `predict`.
    2.  Should implement the following abstract methods `load_model`,
        `save_model` `train` and `evaluate`. These methods provide the
        functionality to save the model to the disk, load the model from the
        disk and train the model and evaluate the model to return appropriate
        measure like accuracy, f1 score, etc.
    Attributes:
        model (Any): instance variable that holds the model.
        save_path (str): path to save the model.
        name (str): name of the model.
        trained (bool): True if model has been trained, false otherwise.
    """

    def __init__(self, save_path: str = 'F:\EMOVO\saves', name: str = 'test'):
        """
        Default constructor for abstract class Model.
        Args:
            save_path(str): path to save the model to.
            name(str): name of the model given as string.
        """
        # Place holder for model
        self.model = None
        # Place holder on where to save the model
        self.save_path = save_path
        # Place holder for name of the model
        self.name = name
        # Model has been trained or not
        self.trained = False

    def train(self, x_train: numpy.ndarray, y_train: numpy.ndarray,
              x_val: numpy.ndarray = None,
              y_val: numpy.ndarray = None) -> None:
        """
        Trains the model with the given training data.
        Args:
            x_train (numpy.ndarray): samples of training data.
            y_train (numpy.ndarray): labels for training data.
            x_val (numpy.ndarray): Optional, samples in the validation data.
            y_val (numpy.ndarray): Optional, labels of the validation data.
        """
        # This will be specific to model so should be implemented by
        # child classes
        raise NotImplementedError()

    def predict(self, samples: numpy.ndarray) -> Tuple:
        """
        Predict labels for given data.
        Args:
            samples (numpy.ndarray): data for which labels need to be predicted
        Returns:
            list: list of labels predicted for the data.
        """
        results = []
        for _, sample in enumerate(samples):
            results.append(self.predict_one(sample))
        return tuple(results)

    def predict_one(self, sample) -> int:
        """
        Predict label of a single sample. The reason this method exists is
        because often we might want to predict label for a single sample.
        Args:
            sample (numpy.ndarray): Feature vector of the sample that we want to
                                    predict the label for.
        Returns:
            int: returns the label for the sample.
        """
        # This need to be implemented for the child models. The reason is that
        # ML models and DL models predict the labels differently.
        raise NotImplementedError()

    def restore_model(self, load_path: str = None) -> None:
        """
        Restore the weights from a saved model and load them to the model.
        Args:
            load_path (str): Optional, path to load the weights from a given path.
        """
        to_load = load_path or self.save_path
        if to_load is None:
            sys.stderr.write(
                "Provide a path to load from or save_path of the model\n")
            sys.exit(-1)
        self.load_model(to_load)
        self.trained = True

    def load_model(self, to_load: str) -> None:
        """
        Load the weights from the given saved model.
        Args:
            to_load: path containing the saved model.
        """
        # This will be specific to model so should be implemented by
        # child classes
        raise NotImplementedError()

    def save_model(self) -> None:
        """
        Save the model to path denoted by `save_path` instance variable.
        """
        # This will be specific to model so should be implemented by
        # child classes
        raise NotImplementedError()

    def evaluate(self, x_test: numpy.ndarray, y_test: numpy.ndarray) -> None:
        """
        Evaluate the current model on the given test data.
        Predict the labels for test data using the model and print the relevant
        metrics like accuracy and the confusion matrix.
        Args:
            x_test (numpy.ndarray): Numpy nD array or a list like object
                                    containing the samples.
            y_test (numpy.ndarray): Numpy 1D array or list like object
                                    containing the labels for test samples.
        """
        predictions = self.predict(x_test)
        print(y_test)
        print(predictions)
        print('Accuracy:%.3f\n' % accuracy_score(y_pred=predictions,
                                                 y_true=y_test))
        print('Confusion matrix:', confusion_matrix(y_pred=predictions,
                                                    y_true=y_test))

import sys

import numpy as np
from keras import Sequential
from keras.layers import LSTM as KERAS_LSTM, Dense, Dropout, Conv2D, Flatten, \
    BatchNormalization, Activation, MaxPooling2D

class DNN(Model):
    """
    This class is parent class for all Deep neural network models. Any class
    inheriting this class should implement `make_default_model` method which
    creates a model with a set of hyper parameters.
    """

    def __init__(self, input_shape, num_classes, **params):
        """
        Constructor to initialize the deep neural network model. Takes the input
        shape and number of classes and other parameters required for the
        abstract class `Model` as parameters.
        Args:
            input_shape (tuple): shape of the input
            num_classes (int): number of different classes ( labels ) in the data.
            **params: Additional parameters required by the underlying abstract
                class `Model`.
        """
        super(DNN, self).__init__(**params)
        self.input_shape = input_shape
        self.model = Sequential()
        self.make_default_model()
        self.model.add(Dense(num_classes, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam',
                           metrics=['accuracy'])
        print(self.model.summary(), file=sys.stderr)
        self.save_path = self.save_path or self.name + '_best_model.h5'

    def load_model(self, to_load):
        """
        Load the model weights from the given path.
        Args:
            to_load (str): path to the saved model file in h5 format.
        """
        try:
            self.model.load_weights(to_load)
        except:
            sys.stderr.write("Invalid saved file provided")
            sys.exit(-1)

    def save_model(self):
        """
        Save the model weights to `save_path` provided while creating the model.
        """
        self.model.save_weights(self.save_path)

    def train(self, x_train, y_train, x_val=None, y_val=None, n_epochs=1):
        """
        Train the model on the given training data.
        Args:
            x_train (numpy.ndarray): samples of training data.
            y_train (numpy.ndarray): labels for training data.
            x_val (numpy.ndarray): Optional, samples in the validation data.
            y_val (numpy.ndarray): Optional, labels of the validation data.
            n_epochs (int): Number of epochs to be trained.
        """
        best_acc = 0
        if x_val is None or y_val is None:
            x_val, y_val = x_train, y_train
        for i in range(n_epochs):
            # Shuffle the data for each epoch in unison inspired
            # from https://stackoverflow.com/a/4602224
            p = np.random.permutation(len(x_train))
            x_train = x_train[p]
            y_train = y_train[p]
            self.model.fit(x_train, y_train, batch_size=32, epochs=50)
            loss, acc = self.model.evaluate(x_val, y_val)
            if acc > best_acc:
                best_acc = acc
        self.trained = True

    def predict_one(self, sample):
        if not self.trained:
            sys.stderr.write(
                "Model should be trained or loaded before doing predict\n")
            sys.exit(-1)
        return np.argmax(self.model.predict(np.array([sample])))

    def make_default_model(self) -> None:
        """
        Make the model with default hyper parameters
        """
        # This has to be implemented by child classes. The reason is that the
        # hyper parameters depends on the model.
        raise NotImplementedError()

class LSTM(DNN):
    """
    This class handles CNN for speech emotion recognitions
    """

    def __init__(self, **params):
        params['name'] = 'LSTM'
        super(LSTM, self).__init__(**params)

    def make_default_model(self):
        """
        Makes the LSTM model with keras with the default hyper parameters.
        """
        self.model.add(
            KERAS_LSTM(128,
                       input_shape=(self.input_shape[0], self.input_shape[1])))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(16, activation='tanh'))

import numpy as np
from sklearn.model_selection import train_test_split

_DATA_PATH = data_path_emovo
_CLASS_LABELS = ("Neutral", "Angry", "Happy", "Sad")


def extract_data(flatten):
    data, labels = get_data(_DATA_PATH, class_labels=_CLASS_LABELS,
                            flatten=flatten)
    x_train, x_test, y_train, y_test = train_test_split(
        data,
        labels,
        test_size=0.2,
        random_state=42,stratify=labels)
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(
        y_test), len(_CLASS_LABELS)


def get_feature_vector(file_path, flatten):
    return get_feature_vector_from_mfcc(file_path, flatten, mfcc_len=39)

from keras.utils import np_utils

def lstm_example():
    to_flatten = False
    x_train, x_test, y_train, y_test, num_labels = extract_data(
        flatten=to_flatten)
    print(y_train.shape,type(y_train))
    y_train = np_utils.to_categorical(y_train)
    y_test_train = np_utils.to_categorical(y_test)
    print('Starting LSTM')
    model = LSTM(input_shape=x_train[0].shape,
                 num_classes=num_labels)
    model.train(x_train, y_train, x_test, y_test_train, n_epochs=1)
    print('\n\nmodel evaluation')
    model.evaluate(x_test, y_test)

lstm_example()