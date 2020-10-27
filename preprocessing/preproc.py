import parselmouth
from parselmouth.praat import call
import os
import pandas as pd
from tqdm import tqdm
#from emovo_db_explorator import csv_generator, add_length_df
from db_exploration.emovo_db_explorator import csv_generator,add_length_df
from preprocessing.vad import voiced_only_sample_generator_activator

path = r'F:\EMOVO\Surprise_notcleaned/'
path_cleaned = r'F:\EMOVO\Surprise/'
# keep only anger,fear,joy,sadness,neutrality
# generate csv

# downsampling,stereo to mono, nr, vad
def pre_proc_start(path,path_cleaned):
    if len(os.listdir(path_cleaned)) == 0:
        for f in tqdm(os.listdir(path)):
            sample = parselmouth.Sound(path + f)
            sample_vad_silence = voiced_only_sample_generator_activator(sample, 0.25, 0.17, 'silence')
            sample_16k = parselmouth.Sound.resample(sample_vad_silence, new_frequency=16000)
            sample_mono = sample_16k.convert_to_mono()
            denoised_silence = call(sample_mono, "Remove noise", 0, 0, 0.025, 80, 10000, 40,
                                    "Spectral subtraction")
            denoised_silence.save(path_cleaned + f, "WAV")

pre_proc_start(path,path_cleaned)