import numpy as np
from pyAudioAnalysis import audioBasicIO, ShortTermFeatures

import pandas as pd
import scipy.io.wavfile as wav
from speechpy.feature import mfcc
import parselmouth
from parselmouth.praat import call

toronto_path_cleaned = r'F:\UNI_MAGISTRAL_GESTIONALE\Tesi_magistrale\Emotional_ behavior_ audio_recognition\Dataset\TorontoEmotionalSpeechSet\TESSTorontoemotionalspeechsetdata\Merged_cleaned_VAD/'
toronto_path = r'F:\UNI_MAGISTRAL_GESTIONALE\Tesi_magistrale\Emotional_ behavior_ audio_recognition\Dataset\TorontoEmotionalSpeechSet\TESSTorontoemotionalspeechsetdata\Merged/'
emovo_path = r'F:\EMOVO\merged_5/'
emovo_path_cleaned = r'F:\EMOVO\clean_5_vad_silence/'
emovo_path_augmented = r'F:\EMOVO\clean_5_vad_silence_augmented/'
df = pd.read_csv('db_exploration/EMOVO.csv')
path = r'F:\EMODB\wav/'
mean_signal_length = 32000
input = r'F:\EMOVO\clean_5_vad_silence/gio-f1-b1.wav'
l=[]
fs,signal=wav.read(input)
mel_coefficients = mfcc(signal, fs, num_cepstral=39)
sound = parselmouth.Sound(input)  # read the sound
pitch = sound.to_pitch_ac(pitch_floor=79)
x_sample = pitch.selected_array['frequency']
x_sample = x_sample[:-1]
x_sample = np.reshape(x_sample,(len(x_sample),1))
meanF0 = call(pitch, "Get mean", 0, 0, "Hertz")  # get mean pitch
formant = call(sound, "To Formant (burg)", 0.01, 5, 5500, 0.01,50)
local_formant=[]
it=0
for x in formant.xs():
    for f in range(1, 4):
        if it > 0:
            local_formant.append(formant.get_value_at_time(f, x))
    it+=1
formant_array=np.reshape(local_formant,(formant.get_number_of_frames()-1,3))
for i in range(0,3):
    formant_array[:,i]=formant_array[:,i]/np.linalg.norm(formant_array[:,i])

concatenated=np.concatenate((mel_coefficients,x_sample),axis=1)
[Fs, x] = audioBasicIO.read_audio_file(input)
#F, f_names = ShortTermFeatures.feature_extraction(x, Fs, 0.025 * Fs, 0.010 * Fs,deltas=False).T
"test per staging di github"
"queste sono le differenze con git diff"

a = [[1, 2], [3, 4]]
print(np.array(a).shape)
b=np.pad(a,((0,1),(0,0)),'constant')
print(b.shape)
pitch = sound.to_pitch_ac(time_step = 0.01,pitch_floor=150,very_accurate=False)
print(pitch.get_number_of_frames())