import librosa
import pysptk
import numpy as np
import scipy.io.wavfile
from spafe.features.lpc import  lpc, lpcc
import parselmouth
from parselmouth.praat import call
import scipy.io.wavfile as wav
from speechpy.feature import mfcc
from pyAudioAnalysis import ShortTermFeatures
input = r'F:\EMOVO\Sad2/pitch-noise-tri-f1-b1.wav'
sound=parselmouth.Sound(input)
print(sound.duration)
print(np.divide(sound.duration,0.010))
fs, signal = wav.read(input)
F, f_names = ShortTermFeatures.feature_extraction(signal, fs, 0.020*fs, 0.010*fs)
F_slice = F[:,0:8]
feature_vector = mfcc(signal, fs, frame_stride=0.01, num_cepstral=13)
