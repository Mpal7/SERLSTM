from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
import pandas as pd
import numpy as np
from keras.utils import to_categorical
from tqdm import tqdm

def short_term_features(path,dframe,mode):

    df = pd.read_csv(dframe)
    df.set_index('fname', inplace=True)
    classes = list(np.unique(df.emotion))
    df.reset_index(inplace=True)
    Y = []
    print("#### CREATING Y SET ####")
    for f in tqdm(df.fname):
        label = df.loc[df.fname == f, 'emotion'].values[0]
        Y.append(classes.index(label))
    Y = to_categorical(Y)
    result = []
    print("\n#### CREATING X SET ####")
    for f in tqdm(df.fname):
        [Fs, x] = audioBasicIO.read_audio_file(path + f)
        F, f_names = ShortTermFeatures.feature_extraction(x, Fs, 0.025 * Fs, 0.010 * Fs,deltas=False)
        if mode == 'RNN':
            F=np.transpose(F)
            result.append(F)
        if mode == 'OTHERS':
            local_results = np.array([])
            for row in F:
                feature_mean = np.nanmean(row)
                local_results = np.hstack((local_results,feature_mean))
            result.append(local_results)

    return result,Y
