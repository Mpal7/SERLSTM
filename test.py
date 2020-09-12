import librosa
import pysptk
import numpy as np
import scipy.io.wavfile
from spafe.features.lpc import  lpc, lpcc
import parselmouth
from parselmouth.praat import call
from speechpy.feature import mfcc
input = r'F:\EMOVO\Sad2/pitch-noise-tri-f1-b1.wav'
import scipy.io.wavfile as wav
import os
print(os.getcwd())
