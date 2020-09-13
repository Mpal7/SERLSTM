import parselmouth
from parselmouth.praat import call
import os
import pandas as pd
from tqdm import tqdm
#from emovo_db_explorator import csv_generator, add_length_df
from db_exploration.emovo_db_explorator import csv_generator,add_length_df
from preprocessing.vad import voiced_only_sample_generator_activator

path = r'F:\EMOVO\merged_5/'
path_cleaned = r'F:\EMOVO\clean_5_vad/'
# keep only anger,fear,joy,sadness,neutrality
# generate csv
def csv_gen():
    csv_generator()
    df = pd.read_csv(r'C:\Users\mp95\PycharmProjects\Thesis\db_exploration/EMOVO.csv')
    add_length_df(df)
    df.reset_index(inplace=True)
    return df

# downsampling,stereo to mono, nr, vad
def pre_proc_start(path,path_cleaned):
    if len(os.listdir(path_cleaned)) == 0:
        df = csv_gen()
        for f in tqdm(df.fname):
            sample = parselmouth.Sound(path + f)
            sample_mono_vad_silence = voiced_only_sample_generator_activator(sample, 0.25, 0.17, 'silence')
            sample_16k = parselmouth.Sound.resample(sample_mono_vad_silence, new_frequency=16000)
            sample_mono = sample_16k.convert_to_mono()
            denoised_silence = call(sample_mono, "Remove noise", 0, 0, 0.025, 80, 10000, 40,
                                    "Spectral subtraction")
            denoised_silence.save(path_cleaned + f, "WAV")

pre_proc_start(path,path_cleaned)