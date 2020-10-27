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
features = ['mfcc','formants']
def get_feature_vector_from_formants(filepath,feature_vector):
    path = (filepath)
    sound = parselmouth.Sound(path)  # read the sound
    pitch = sound.to_pitch_ac()
    meanF0 = call(pitch, "Get mean", 0, 0, "Hertz")  # get mean pitch
    if meanF0 > 150:
        maxFormant = 5500  # women
    else:
        maxFormant = 5000  # men
    formant = call(sound, "To Formant (burg)", 0 , 5, maxFormant, 0.2, 50)    # even if i need 3 formants i calculate 5, it behaves better apparently
    local_formant = []
    for x in formant.xs():
        for f in range(1, 4):
            local_formant.append(formant.get_value_at_time(f, x))
    formant_array = np.reshape(local_formant, (formant.get_number_of_frames(), 3))
    #padding array to match first feature in features array length with special masking value
    if features.index('formants') is not 0:
        if formant_array.shape[0] < feature_vector.shape[0]:
            formant_array = np.pad(formant_array, ((0, feature_vector.shape[0] - formant_array.shape[0]), (0, 0)),
                                   'constant', constant_values=(0, 100))
        elif formant_array.shape[0] > feature_vector.shape[0]:
            pad_len = formant_array.shape[0] - feature_vector.shape[0]
            pad_len //= 2
            formant_array = formant_array[pad_len:pad_len + feature_vector.shape[0]]
    #standardization
    for i in range(0, 3):
        formant_array[:, i] = formant_array[:, i] / np.linalg.norm(formant_array[:, i])
    if features.index('formants') is not 0:
        formant_array = np.concatenate((feature_vector, formant_array), axis=1)
    return formant_array

eeee=get_feature_vector_from_formants(input,feature_vector)
