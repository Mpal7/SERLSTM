import librosa
import pysptk
import numpy as np
import scipy.io.wavfile
from spafe.features.lpc import  lpc, lpcc
import parselmouth
input = r'F:\EMOVO\clean_5_vad_silence/gio-f1-b1.wav'

# init input vars
num_ceps = 13
lifter = 0
normalize = True

# read wav
fs, sig = scipy.io.wavfile.read(input)

# compute lpcs
lpcs = lpc(sig=sig, fs=fs, num_ceps=num_ceps)

# compute lpccs
lpccs = lpcc(sig=sig, fs=fs, num_ceps=num_ceps, lifter=lifter, normalize=normalize)



sound,sr=librosa.load(input,sr=16000)
print(sound.shape)
X = librosa.util.frame(sound, frame_length=320, hop_length=320).T
lpccf=[]
for e in X:
    lpcpy = pysptk.lpc(e)
    lpcc = pysptk.lpc2c(lpcpy)
    lpccn = lpcc / np.linalg.norm(lpcc)
    lpccf.append(lpccn)
lpccf = np.array(lpccf)