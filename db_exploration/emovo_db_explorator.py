import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
import os
import csv
import pandas as pd
from scipy.io import wavfile
from python_speech_features import mfcc
import wave

# sample rate and bit depth
emovo = wave.open(r'F:\EMOVO\merged_5\gio-f1-b1.wav')
sample_rate = emovo.getframerate()
bit_depth = emovo.getsampwidth()

# HELPER FUNCTIONS

def gender_csv(emodb_filename):
    if 'f' in emodb_filename:
        gender = 'female'
    else:
        gender = 'male'
    return gender

def id_csv(emodb_filename):
    id_check =['m1','m2','m3','f1','f2','f3']
    id_return = ['1','2','3','4','5','6']
    for i in id_check:
        if i in emodb_filename:
            return int(id_return[id_check.index(i)])

def emotion_csv(emodb_filename):
    emo_check=["pau","gio","rab","neu","tri"]
    emo_return = ["fear", "joy", "anger","neutrality","sadness"]
    for i in emo_check:
        if i in emodb_filename:
            return emo_return[emo_check.index(i)]

def sentence_csv(emodb_filename):
    sentence_identifier=['b1', 'b2', 'b3', 'l1', 'l2', 'l3', 'l4' ,'n1', 'n2', 'n3', 'n4', 'n5', 'd1', 'd2']
    sentence_return = ["Gli operai si alzano presto","I vigili sono muniti di pistola","La cascata fa molto rumore",
                 "L autunno prossimo Tony partira per la Spagna: nella prima meta di ottobre",
                 "Ora prendo la felpa di la ed esco per fare una passeggiata",
                 "Un attimo dopo s e incamminato ed e inciampato",
                 "Vorrei il numero telefonico del Signor Piatti","La casa forte vuole col pane",
                 "La forza trova il passo e l aglio rosso", "Il gatto sta scorrendo nella pera",
                 "Insalata pastasciutta coscia d agnello limoncello",
                 "Uno quarantatre dieci mille cinquantasette venti",
                 "Sabato sera cosa fara?","Porti con te quella cosa?"]
    for i in sentence_identifier:
        if  i in emodb_filename:
            return sentence_return[sentence_identifier.index(i)]

def sentence_type_csv(emodb_filename):
    sentence_identifier = ['b1', 'b2', 'b3', 'l1', 'l2', 'l3', 'l4', 'n1', 'n2', 'n3', 'n4', 'n5', 'd1', 'd2']
    for i in sentence_identifier:
        if i in emodb_filename:
            if 'b' in i:
                return 'short'
            elif 'l' in i:
                return 'long'
            elif 'n' in i:
                return 'nosense'
            elif 'd' in i:
                return 'domanda'

def add_length_df(df):
    df.set_index('fname', inplace=True)
    # adding length for each .wav sample
    for f in df.index:
        rate, signal = wavfile.read(r'F:\EMOVO\merged_5/' + f)
        df.at[f, 'length'] = signal.shape[0] / rate

def calc_fft(y, rate):
   fft = np.fft.fft(y)
   magnitude = np.abs(fft)
   frequency = np.linspace(0, rate, len(magnitude))
   l_frequency = frequency[:int(len(frequency)/2)]
   l_magnitude = magnitude[:int(len(frequency) / 2)]
   return l_frequency, l_magnitude


def plot_signals(signals):
    fig, axes = plt.subplots(ncols=5, sharex = True,
                             sharey = True, figsize=(15, 5))
    fig.suptitle('Time Series', size=16)
    i = 0
    for x in range(5):
        axes[x].set_title(list(signals.keys())[i])
        axes[x].plot(list(signals.values())[i])
        axes[x].get_xaxis().set_visible(False)
        axes[x].get_yaxis().set_visible(False)
        i += 1

def plot_fft(fft):
    fig, axes = plt.subplots(ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Fourier Transforms', size=16)
    i = 0
    for x in range(5):
        data = list(fft.values())[i]
        freq, magnitude = data[0], data[1]
        axes[x].set_title(list(fft.keys())[i])
        axes[x].plot(freq, magnitude)
        axes[x].get_xaxis().set_visible(False)
        axes[x].get_yaxis().set_visible(False)
        i += 1

def plot_mfccs(mfccs):
    fig, axes = plt.subplots(ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Mel Frequency Cepstrum Coefficients', size=16)
    i = 0
    for x in range(5):
            axes[x].set_title(list(mfccs.keys())[i])
            axes[x].imshow(list(mfccs.values())[i],
                    cmap='hot', interpolation='nearest')
            axes[x].get_xaxis().set_visible(False)
            axes[x].get_yaxis().set_visible(False)
            i += 1


def noise_reduction(y, rate, threshold): ##not using it right now, better with VAD library
    mask = []
    y = pd.Series(y).apply(np.abs)  # we make it absolute
    y_mean = y.rolling(window=170, min_periods=1,
                       center=True).mean()  # we create a rolling window to check for silence inside speech pause general value (170ms)
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask

def csv_generator():
    with open('EMOVO.csv', 'w', newline='') as f:
        fields = ['fname', 'gender', 'ID_speaker', 'emotion', 'sentence', 'type']
        writer = csv.DictWriter(f, fieldnames=fields)
        # csv file
        writer.writeheader()
        for j in range(0, len(os.listdir(r'F:\EMOVO\merged_5'))):
            listOfFile = os.listdir(r'F:\EMOVO\merged_5')

            writer.writerow({'fname': listOfFile[j].replace(" ", "_"), 'gender': gender_csv(listOfFile[j]),
                             'ID_speaker': id_csv(listOfFile[j]), 'emotion': emotion_csv(listOfFile[j]),
                             'sentence': sentence_csv(listOfFile[j]), 'type': sentence_type_csv(listOfFile[j])})

def plotter():
    csv_generator()
    df = pd.read_csv('EMOVO.csv')  # To view as DataFrame select all the code to run df=... and then shift+alt+e, find in the box the variable df-->right click "view as DataFrame"
    add_length_df(df)
    # .wav length by emotion
    classes = list(np.unique(df.emotion))
    class_dist = df.groupby(['emotion'])['length'].mean()
    print("average length for emotion class")
    print(class_dist)

    # plotting with pychart
    fig, ax = plt.subplots()
    ax.set_title('Emotion Distribution length', y=1.05)
    ax.pie(class_dist, labels=class_dist.index, autopct='%1.2f%%', shadow=False,
           startangle=90)  # autopct is the decimal showed in the chart
    ax.axis('equal')  # looks like a cirlce instead of an elix
    df.reset_index(inplace=True)  # setback fname instead of index

    signals = {}
    fft = {}
    mfccs = {}

    # visualization one one sample for each emotion
    for c in classes:
        wav_file = df[df.emotion == c].iloc[0, 0]  # for each emotion i take only one sample
        signal, rate = librosa.load(
            r'F:\EMOVO\merged_5\\' + wav_file,
            sr=sample_rate)
        mask = noise_reduction(signal, rate, 0.0005)  # generic treshold
        signal = signal[mask]
        signals[c] = signal
        fft[c] = calc_fft(signal, rate)
        mel = mfcc(signal[:rate], rate, numcep=13, nfilt=26, nfft=int(rate / 40)).T
        mfccs[c] = mel

    plot_signals(signals)
    plot_fft(fft)
    plot_mfccs(mfccs)

    plt.show()