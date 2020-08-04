import librosa
import numpy as np
import pandas as pd
import parselmouth
from keras.utils import to_categorical
from python_speech_features import delta
from python_speech_features import mfcc
from tqdm import tqdm
from parselmouth.praat import call


def extraction(fmode,filepath,df,mode,dictionary):
    if mode == 'RNN':
        if 'MFCC' in fmode:
            dictionary_updater = {'mfcc1': [], 'mfcc2': [], 'mfcc3': [], 'mfcc4': [], 'mfcc5': [], 'mfcc6': [],
                                 'mfcc7': [],
                                 'mfcc8': [],
                                 'mfcc9': [], 'mfcc10': [], 'mfcc11': [], 'mfcc12': []}
            dictionary.update(dictionary_updater)
            print('\n#### MFCC EXTRACTION START ######')
            for f in tqdm(df.fname):
                wav = parselmouth.Sound(filepath + f)
                x_sample_mfcc = wav.to_mfcc(number_of_coefficients=12, window_length=0.025, time_step=0.0125).to_array()
                for row_id in range(1, 13):
                    dictionary['mfcc' + str(row_id)].append(x_sample_mfcc[row_id][:])

        if 'pitch' in fmode:
            dictionary['pitch'] = []
            print('\n#### PITCH EXTRACTION START ######')
            for f in tqdm(df.fname):
                path = (filepath + f)
                signal = parselmouth.Sound(path)
                pitch = signal.to_pitch_ac(time_step=0.0125, pitch_floor=120)  # pitch floor 3/0.025ms
                x_sample = pitch.selected_array['frequency']
                dictionary['pitch'].append(x_sample)

        if 'formants' in fmode:
            dictionary_updater = {'F1': [], 'F2': [], 'F3': []}
            dictionary.update(dictionary_updater)
            print('\n#### FORMANTS EXTRACTION START ######')

            for file in tqdm(df.fname):
                local_formant = {'F1': [], 'F2': [], 'F3': [], 'F4': [], 'F5': []}
                path = (filepath + file)
                sound = parselmouth.Sound(path)  # read the sound
                pitch = sound.to_pitch_ac(time_step=0.0125, pitch_floor=120)
                meanF0 = call(pitch, "Get mean", 0, 0, "Hertz")  # get mean pitch
                if meanF0 > 150:
                    maxFormant = 5500  # women
                else:
                    maxFormant = 5000  # men
                formant = call(sound, "To Formant (burg)", 0.0125, 5, maxFormant, 0.025,
                               50)  # even if i need 3 formants i calculate 5, it behaves better apparently
                for x in formant.xs():
                    for f in range(1, 6):
                        local_formant['F' + str(f)].append(formant.get_value_at_time(f, x))
                [dictionary['F' + str(f)].append(local_formant['F' + str(f)]) for f in
                 range(1, 4)]  # string only F1,F2,F3
                # alternative get formants at at glottal pulses

        if 'intensity' in fmode:
            dictionary['intensity'] = []
            print('\n#### INTENSITY EXTRACTION START ######')
            for f in tqdm(df.fname):
                x_intensity_local = []
                path = (filepath + f)
                signal = parselmouth.Sound(path)
                intensity = signal.to_intensity(minimum_pitch=128,
                                                time_step=0.0125)  # window length is 3.2/minimum_pitch
                for x in intensity.xs():
                    x_intensity_local.append(intensity.get_value(time=x))
                dictionary['intensity'].append(x_intensity_local)

    if mode == 'OTHERS':
        dictionary['X'] = []
        for f in tqdm(df.fname):
            wav = parselmouth.Sound(filepath + f)
            result = np.array([])
            if 'MFCC' in fmode:
                x_sample_mfcc = np.mean(wav.to_mfcc(number_of_coefficients=12, window_length=0.025, time_step=0.0125).to_array().T, axis = 0)
                result = np.hstack((result,x_sample_mfcc))
            if 'intensity' in fmode:
                intensity = wav.to_intensity(minimum_pitch=128,time_step=0.0125).get_average()  # window length is 3.2/minimum_pitch
                result = np.hstack((result,intensity))
            if 'formants' in fmode:
                pitch = wav.to_pitch_ac(time_step=0.0125, pitch_floor=120)
                meanF0 = call(pitch, "Get mean", 0, 0, "Hertz")  # get mean pitch
                if meanF0 > 150:
                    maxFormant = 5500  # women
                else:
                    maxFormant = 5000  # men
                formant = call(wav, "To Formant (burg)", 0.0125, 5, maxFormant, 0.025,
                               50)  # even if i need 3 formants i calculate 5, it behaves better apparently
                meanformant1 = call(formant, "Get mean", 1, 0, 0, 'Hertz')
                result = np.hstack((result,meanformant1))
                meanformant2 = call(formant, "Get mean", 2, 0, 0, 'Hertz')
                result = np.hstack((result, meanformant2))
                meanformant3 = call(formant,"Get mean",3, 0, 0,'Hertz')
                result = np.hstack((result, meanformant3))
            if 'pitch' in fmode:
                pitch = wav.to_pitch_ac(time_step=0.0125, pitch_floor=120)  # pitch floor 3/0.025ms
                pitch_array = pitch.selected_array['frequency']
                pitch_mean=np.mean(pitch_array)
                result = np.hstack((result,pitch_mean))
            dictionary['X'].append(result)


def build_y_set(df, classes):
    print('\n#### CREATING Y SET ######')
    Y=[]
    for f in tqdm(df.fname):
        label = df.loc[df.fname == f, 'emotion'].values[0]
        Y.append(classes.index(label))
    Y = to_categorical(Y)
    return Y

def feature_extraction(fmode, filepath, df_name,mode):
    file_path = filepath
    df = pd.read_csv(df_name)
    df.set_index('fname', inplace=True)
    classes = list(np.unique(df.emotion))
    df.reset_index(inplace=True)

    dictionary = {}
    extraction(fmode,file_path,df,mode,dictionary)
    Y=build_y_set(df,classes)
    size = len(df.index)
    X = []

    if mode == 'RNN':
        print('\n#### CREATING X SET ######')
        sample_it = 0
        pbar = tqdm(total=size,mininterval=0,miniters=1)
        while sample_it<size:
            column = 0
            if 'formant' in fmode:
                Tx = len(dictionary['F1'][sample_it])
            elif 'MFCC' in fmode:
                Tx = len(dictionary['mfcc1'][sample_it])
            else:
                Tx = len(dictionary[fmode[0]][sample_it])
            example = np.zeros((Tx,len(dictionary)))
            for e in dictionary:
                if len(example[:,column])<len(dictionary[e][sample_it]):
                    mismatch = len(dictionary[e][sample_it])-len(example[:, column])
                    dictionary[e][sample_it]=dictionary[e][sample_it][: -mismatch]
                example[:, column] = dictionary[e][sample_it]
                column += 1
            sample_it+=1
            X.append(example)
            pbar.update(1)
        pbar.close()

    elif mode == 'OTHERS':
        X=dictionary['X']

    return X,Y