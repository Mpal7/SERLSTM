import librosa
import pysptk
import numpy as np
import scipy.io.wavfile
from spafe.features.lpc import  lpc, lpcc
import parselmouth
from parselmouth.praat import call
import scipy.io.wavfile as wav
from speechpy.feature import mfcc
input = r'F:\EMOVO\Sad2/pitch-noise-tri-f1-b1.wav'
fs, signal = wav.read(input)

mel_coefficients = mfcc(signal, fs, frame_stride=0.1, num_cepstral=13)

path = (input)
signal = parselmouth.Sound(path)
# compare with pitch = sound.to_pitch_ac(time_step = 0.01,pitch_floor=150,very_accurate=True)
pitch = signal.to_pitch_ac(time_step=0.1,pitch_floor = 150,very_accurate=True)
x_sample = pitch.selected_array['frequency']
x_sample = np.reshape(x_sample, (len(x_sample), 1))
