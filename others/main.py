from preprocessing.preproc import pre_proc_start
from others.rnn import rnn
from sklearn.model_selection import train_test_split
from others.decisiontree_svm_randomf import dectree_svm_rf
import numpy as np
from others.feature_ext_scipy import feature_extraction_scipy
from others.cnn import cnn_model

toronto_path_cleaned = r'F:\UNI_MAGISTRAL_GESTIONALE\Tesi_magistrale\Emotional_ behavior_ audio_recognition\Dataset\TorontoEmotionalSpeechSet\TESSTorontoemotionalspeechsetdata\Merged_cleaned_VAD/'
toronto_path = r'F:\UNI_MAGISTRAL_GESTIONALE\Tesi_magistrale\Emotional_ behavior_ audio_recognition\Dataset\TorontoEmotionalSpeechSet\TESSTorontoemotionalspeechsetdata\Merged/'
emovo_path = r'F:\EMOVO\merged_5/'
emovo_path_cleaned = r'F:\EMOVO\clean_5_vad_silence/'
emodb_path = r'F:\EMODB\wav/'
db = 'EMOVO.csv'
pre_proc_start(emovo_path,emovo_path_cleaned)
#['MFCC','pitch','intensity','formants'], returns a (420,Tx,n_features) array
valid = False
it=0
augment=True

while valid == False:
    if it == 0:
        mode = input("please enter selected algorithm {RNN},{CNN},{OTHERS}:\n")
        it+=1
    else:
        mode = input()

    if mode == 'RNN':
        slicing = True
        #X,y = short_term_features(emodb_path,'EMODB.csv',mode)
        #X,y= feature_extraction(['MFCC','formants','intensity'],emodb_path,db,mode)
        X,y= feature_extraction_scipy(['MFCC'],emovo_path_cleaned,db,mode,slicing,augment)
        Kfold = 10
        full_batch = int(len(X) - (len(X) / Kfold))
        #rnn(X,y,batch_size=full_batch,Kfold=Kfold,class_number=5)
        rnn(X,y,batch_size=32,Kfold=Kfold,class_number=5)

        valid = True


    elif mode == 'OTHERS':
        #X, y = short_term_features(emovo_path_cleaned, 'EMOVO.csv', mode)
        X,y=feature_extraction_scipy(['MFCC'],emovo_path_cleaned,db,mode)
        #X,y= feature_extraction_librosa(['MFCC'],emovo_path_cleaned,db,mode)
        X = np.array(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1,stratify=y)
        dectree_svm_rf(X_train, y_train, X_test, y_test)
        valid = True

    elif mode == 'CNN':
        cnn_model(db,emovo_path_cleaned)
        valid = True


    else:
        print("\nIncorrect entry, please type {RNN} or {OTHERS} or {CNN}")

