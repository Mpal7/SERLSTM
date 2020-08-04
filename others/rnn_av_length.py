import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,LSTM,Masking
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

def rnn_avg(X, y,batch_size,Kfold):
    input_shape_x=X[0].shape

    def k_fold_cross_validation(model,batch_size,x_size):
        kf = StratifiedKFold(n_splits=Kfold)
        train_loss= []
        train_acc = []
        test_loss = []
        test_acc = []


        for i, (train_index,test_index) in enumerate(kf.split(X,y.argmax(1))):
            print("\n##### FOLDER NUMBER ",i+1," #######")
            X_train =[]
            X_test = []
            [X_train.append(X[i]) for i in train_index]
            [X_test.append(X[i]) for i in test_index]
            y_train, y_test = y[train_index],y[test_index]
            X_train = np.array(X_train)
            X_test = np.array(X_test)
            y_train = np.array(y_train)
            y_test = np.array (y_test)
            history = model.fit(X_train, y_train, epochs=25, batch_size=batch_size, validation_data=(X_test, y_test))
            loss, acc = model.evaluate(X_test, y_test, batch_size=batch_size,verbose=False)
            train_loss.append(history.history['loss'])
            train_acc.append(history.history['accuracy'])
            test_loss.append(loss)
            test_acc.append(acc)
            model.load_weights('model.h5') #reset weights for next folder
        avg_loss_train = np.mean(train_loss)
        avg_acc_train = np.mean(train_acc)
        avg_loss_test = np.mean(test_loss)
        avg_acc_test = np.mean(test_acc)
        print(f'f1: {avg_acc_train} loss: {avg_loss_train}')
        print(f'f1: {avg_acc_test} loss: {avg_loss_test}')


    model = Sequential()
    model.add(LSTM(128, input_shape=(input_shape_x[0], input_shape_x[1])))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='tanh'))
    model.add(Dense(4, activation='softmax'))
    opt = tf.keras.optimizers.Adam()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.save_weights('model.h5')
    k_fold_cross_validation(model,batch_size,len(X))

