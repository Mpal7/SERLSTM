import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
import os
import csv
import pandas as pd
from scipy.io import wavfile
import wave

# sample rate and bit depth
emovo = wave.open(r'F:\EMOVO\merged_5\gio-f1-b1.wav')
sample_rate = emovo.getframerate()
bit_depth = emovo.getsampwidth()

filepath= r'F:\UNI_MAGISTRAL_GESTIONALE\Tesi_magistrale\Emotional_ behavior_ audio_recognition\Dataset\TorontoEmotionalSpeechSet\TESSTorontoemotionalspeechsetdata\Merged'
# HELPER FUNCTIONS
def age_csv(filename):
    if 'Y' in filename:
        if filename.index('Y') == 0:
            age = 'young'
    else:
        age = 'old'
    return age


def emotion_csv(filename):
    emo_check = ["fear", "happy", "angry","neutral","sad"]
    for i in emo_check:
        if i in filename:
            return i

def add_length_df(df):
    df.set_index('fname', inplace=True)
    # adding length for each .wav sample
    for f in df.index:
        rate, signal = wavfile.read(filepath+'\\'+f)
        df.at[f, 'length'] = signal.shape[0] / rate

def csv_generator():
    with open('TESS.csv', 'w', newline='') as f:
        fields = ['fname', 'age', 'emotion']
        writer = csv.DictWriter(f, fieldnames=fields)
        # csv file
        writer.writeheader()
        for j in range(0, len(os.listdir(filepath))):
            listOfFile = os.listdir(filepath)

            writer.writerow({'fname': listOfFile[j], 'age': age_csv(listOfFile[j]), 'emotion': emotion_csv(listOfFile[j])})
