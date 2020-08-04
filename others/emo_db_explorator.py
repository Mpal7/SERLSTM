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

filepath= r'F:\EMODB\wav'
# HELPER FUNCTIONS

def emotion_csv(filename):
    emo_check = ['W','F','N','T']
    emo_write = ["Anger", "Joy", "Neutral","Sad"]
    for i in emo_check:
        if i in filename:
            return emo_write[emo_check.index(i)]

def csv_generator():
    with open('EMODB.csv', 'w', newline='') as f:
        fields = ['fname', 'emotion']
        writer = csv.DictWriter(f, fieldnames=fields)
        # csv file
        writer.writeheader()
        for j in range(0, len(os.listdir(filepath))):
            listOfFile = os.listdir(filepath)

            writer.writerow({'fname': listOfFile[j], 'emotion': emotion_csv(listOfFile[j])})

csv_generator()