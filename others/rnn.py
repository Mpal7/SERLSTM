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

def rnn(X, y,batch_size,Kfold,class_number):
    special_value = 100
    n_features = X[0].shape[1]

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


    def bucketing_padding(X,y):
        #sorting and bucketing X, to keep correct y permutation i use a lenx[X,y] array
        curr = 0
        #sorting descending for bucketing
        Xy_paired=[list(pair) for pair in zip(X, y)]

        def takeSecond(elem):
            return len(elem[0])
        Xy_paired.sort(reverse=True, key=takeSecond)

        X_sorted = []
        y_sorted = []

        for i in Xy_paired:
            X_sorted.append(i[0])
            y_sorted.append(i[1])

        Xpad = []
        #bucketing and padding for X
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
            Xpad_local = np.full(((len(X)-curr), max_seq_len, X[0].shape[1]), fill_value=special_value)
            for s, x in enumerate(X_sorted[curr:len(X)]):
                seq_len = x.shape[0]
                Xpad_local[s, 0:seq_len, :] = x
            Xpad.append(Xpad_local)

        #reshape y for batches
        curr=0
        y_batches = []
        while (curr + batch_size < len(y_sorted)):
            y_local = []
            [y_local.append(i) for i in y_sorted[curr:curr+batch_size]]
            curr += batch_size
            y_batches.append(y_local)
        if curr!= len(y_sorted):
            y_local=[]
            [y_local.append(i) for i in y_sorted[curr:len(y_sorted)]]
            y_batches.append(y_local)

        return Xpad,y_batches

    def k_fold_cross_validation(model,batch_size,x_size,class_number):
        kf = StratifiedKFold(n_splits=Kfold,random_state=42,shuffle=True)
        train_loss= []
        train_acc = []
        test_loss = []
        test_acc = []


        for i, (train_index,test_index) in enumerate(kf.split(X,y.argmax(1))):
            print("\n##### FOLDER NUMBER ",i+1," #######")
            epoch_counter = 0
            X_train =[]
            X_test = []
            [X_train.append(X[i]) for i in train_index]
            [X_test.append(X[i]) for i in test_index]
            y_train, y_test = y[train_index],y[test_index]
            if batch_size == int(x_size-(x_size/Kfold)):
                X_train=padding(X_train)
                X_test=padding(X_test)
                history = model.fit(X_train, y_train, epochs=1, batch_size=batch_size,validation_data=(X_test,y_test))
                loss, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
                train_loss.append(history.history['loss'])
                train_acc.append(history.history['accuracy'])
                test_loss.append(loss)
                test_acc.append(acc)
            else:
                if batch_size != 1:
                    X_train, y_train = bucketing_padding(X_train, y_train)
                model_train_task, report_loss_train, report_acc_train, report_loss_test, report_acc_test = train_test_task(model, X_train, y_train,X_test,y_test, 30,batch_size,class_number)
                train_loss.append(report_loss_train)
                train_acc.append(report_acc_train)
                test_loss.append(report_loss_test)
                test_acc.append(report_acc_test)

            model.load_weights('model.h5') #reset weights for next folder
        avg_loss_train = np.mean(train_loss)
        avg_acc_train = np.mean(train_acc)
        avg_loss_test = np.mean(test_loss)
        avg_acc_test = np.mean(test_acc)
        print(f'f1: {avg_acc_train} loss: {avg_loss_train}')
        print(f'f1: {avg_acc_test} loss: {avg_loss_test}')




    def train_test_task(model, X_train, y_train,X_test,y_test, EPOCH,batch_size,class_number):
        # single zero padding for NULL datas
        report_acc = 0.0
        for epoch in range(EPOCH):
            print(f'Training epoch {epoch} ...')
            avg_train_loss = 0.0
            avg_train_acc = 0.0
            avg_test_loss = 0.0
            avg_test_acc = 0.0

            print("Training")
            for sample_i in tqdm(range(int(len(X_train))),position=0,leave=True):
                if batch_size != 1:
                    [train_loss, train_acc] = model.train_on_batch(X_train[sample_i],np.array(y_train[sample_i]).reshape(len(y_train[sample_i]),class_number))
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
               [test_loss, test_acc] = model.test_on_batch(X_test[sample_i].reshape(1, X_test[sample_i].shape[0], X_test[sample_i].shape[1]),y_test[sample_i].reshape(1, class_number))
               avg_test_loss += test_loss / (len(X_test))
               avg_test_acc += test_acc / (len(X_test))
            report_test_acc = avg_test_acc
            report_test_loss = avg_test_loss

            print(f'acc_train: {avg_train_acc} loss_train: {avg_train_loss} acc_test: {avg_test_acc} loss_test: {avg_test_loss}')

        return model,report_train_loss, report_train_acc,report_test_loss, report_test_acc

    def confusion_matrix(y_test, y_pred):
        mat=confusion_matrix(y_test,y_pred)
        plot_confusion_matrix(conf_mat=mat)


    model = Sequential()
    model.add(Masking(mask_value=special_value, input_shape=(None, n_features)))
    model.add(LSTM(128, input_shape=(None, n_features)))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(class_number, activation='softmax'))
    opt = tf.keras.optimizers.Adam()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    print(model.summary())
    model.save_weights('model.h5')
    k_fold_cross_validation(model,batch_size,len(X),class_number)

