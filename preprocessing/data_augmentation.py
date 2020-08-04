import librosa
from tqdm import tqdm
import os
import numpy as np
import scipy.io.wavfile as wav


directory = r'F:\EMOVO\clean_5_vad_silence_augmented/'
directory_from =r'F:\EMOVO\clean_5_vad_silence/'
main_directory = r'F:\EMOVO\\'
write = False #write augmented data in folder
mean_signal_length=32000
pad_sli= True

#this module it's intended for data augmentation to use parselmouth features



def padding_slicing(signal):
    s_len = len(signal)

    if s_len < mean_signal_length:
        pad_len = mean_signal_length - s_len
        pad_rem = pad_len % 2
        pad_len //= 2
        signal = np.pad(signal, (pad_len, pad_len + pad_rem),
                        'constant', constant_values=0)
    else:
        pad_len = s_len - mean_signal_length
        pad_len //= 2
        signal = signal[pad_len:pad_len + mean_signal_length]
    return signal

def noise(data):
    """
    Adding White Noise.
    """
    noise_amp = 0.005 * np.random.uniform() * np.amax(data)
    data = data.astype('float64') + noise_amp * np.random.normal(size=data.shape[0])
    return data

def pitch_augmentation(data, sample_rate):
    """
    Pitch Tuning.
    """
    bins_per_octave = 12
    pitch_pm = 2
    pitch_change = pitch_pm * 2 * (np.random.uniform())
    data = librosa.effects.pitch_shift(data.astype('float64'),
                                       sample_rate, n_steps=pitch_change,
                                       bins_per_octave=bins_per_octave)
    return data


def shift(data):
 """
 Random Shifting.
 """
 s_range = int(np.random.uniform(low=-5, high = 5)*500)
 return np.roll(data, s_range)

def dyn_change(data):
 """
 Random Value Change.
 """
 dyn_change = np.random.uniform(low=1.5,high=3)
 return (data * dyn_change)

def augmentation_pad_sli():
    if write == True:
        for file in os.listdir(r'F:\EMOVO\clean_5_vad_silence/'):
            fs, data = wav.read(directory_from + file)
            signal_pitch = pitch_augmentation(data, fs)
            signal_pitch_noise = noise(signal_pitch)
            padsli_signal = padding_slicing(signal_pitch_noise)
            padsli_signal = padsli_signal.astype(np.int16)
            wav.write(r'F:\EMOVO\clean_5_vad_silence_augmented/pitch_noise_padsli_'+file, fs, padsli_signal)

#augmentation_pad_sli()

def pad_sli_maker():
    if pad_sli == True:
        class_labels_origin = ("Sad", "Happy", "Angry", "Neutral", "Scared")
        _CLASS_LABELS = ("Sad1", "Happy1", "Angry1", "Neutral1", "Scared1")
        for e in class_labels_origin:
            for file in tqdm(os.listdir(main_directory + e)):
                fs, data = wav.read(directory_from + file)
                padsli_signal = padding_slicing(data)
                padsli_signal = padsli_signal.astype(np.int16)
                wav.write(main_directory + _CLASS_LABELS[class_labels_origin.index(e)] + '/'+ file, fs, padsli_signal)
#pad_sli_maker()





