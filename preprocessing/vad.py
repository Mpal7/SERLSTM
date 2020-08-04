import parselmouth
from parselmouth.praat import call
import numpy


def create_empty_sound(duration):
    return parselmouth.Sound(numpy.zeros(int( 16000 * duration)), 16000)


def extract_interval_durations(text_grid, tier_number):
    intervals = []

    n_intervals = call(text_grid, "Get number of intervals", tier_number)
    for interval_number in range(1, n_intervals + 1):
        label = call(text_grid, "Get label of interval", tier_number, interval_number)
        if label != "":
            start = call(text_grid, "Get starting point", tier_number, interval_number)
            end = call(text_grid, "Get end point", tier_number, interval_number)
            duration = end - start
            intervals.append((label, duration))

    return intervals


def extract_interval_start_ending_time(text_grid, tier_number,vad_dic):
    intervals = []
    label_count_dict = {'sounding':0,'pause':0}

    n_intervals = call(text_grid, "Get number of intervals", tier_number)
    for interval_number in range(1, n_intervals + 1):
        label = call(text_grid, "Get label of interval", tier_number, interval_number)
        if label != "":
            start = call(text_grid, "Get starting point", tier_number, interval_number)
            end = call(text_grid, "Get end point", tier_number, interval_number)
            duration = end-start
            intervals.append((label, start, end))
            if vad_dic:
                if label == 'silent' and interval_number!=1 and interval_number!= n_intervals:
                    label_count_dict['pause'] = + 1
                else:
                    label_count_dict[label] = + 1

    return intervals, label_count_dict,


def voiced_only_sample_generator(intervals, margin, sample,sounding_count,mode):
    unvoiced_margin_frame = create_empty_sound(margin)
    list_voice_frame = []

    if sounding_count == 1:
        for (label, start, end) in intervals:
            if label == 'sounding':
                voiced_sample = sample.extract_part(start, end)
            else:
                raise ValueError('audio is empty!')

    else:
        for (label, start, end) in intervals:
            if mode == 'silence':
                if label == 'sounding':
                    list_voice_frame.append(sample.extract_part(start, end))
            if mode == 'pause':
                if intervals.index((label,start,end)) == 0 or intervals.index((label,start,end)) == (len(intervals)-1):
                    if label == 'sounding':
                        list_voice_frame.append(sample.extract_part(start, end))
                else:
                    list_voice_frame.append(sample.extract_part(start, end))

        voiced_sample = parselmouth.Sound.concatenate(list_voice_frame)
    return voiced_sample


def write_interval_durations_to_textfile(intervals, filename):
    with open(filename, 'w') as f:
        for (label, duration) in intervals:
            f.write(label)
            f.write("\t")
            f.write(str(duration))
            f.write("\n")


def write_interval_durations_to_textfile_activator(filename):
    sound = parselmouth.Sound(r'F:\EMOVO\merged_5/' + filename)
    # creates a text_grid with the parts voiced and unvoiced
    text_grid = call(sound, "To TextGrid (silences)", 100, 0.0, -25.0, 0.25, 0.1, "silent", "sounding")
    interval_list = extract_interval_durations(text_grid, 1)
    write_interval_durations_to_textfile(interval_list, r'F:\EMOVO\durations.txt')

def vad_dictionary(sample):
    vad_dic = True
    text_grid_silence = call(sample, "To TextGrid (silences)", 100, 0.0, -25.0, 0.175, 0.1, "silent",
                     "sounding")
    interval_list, label_dic = extract_interval_start_ending_time(text_grid_silence, 1,vad_dic)
    return label_dic['pause']


#mode--> pause (average human pause 170ms), silence
def voiced_only_sample_generator_activator(sample,minimum_silent,minimum_sounding,mode):
    margin = 0.1
    vad_dic = False
    # "To TextGrid (silences)"--> minimum pitch, time step, silence tresh_hold,minimum silent interval duration, minimum sounding interval duration, silent interval label, sounding interval label
    text_grid = call(sample, "To TextGrid (silences)", 100, 0.0, -25.0, minimum_silent, minimum_sounding, "silent", "sounding")
    interval_list, label_dic, = extract_interval_start_ending_time(text_grid, 1,vad_dic)
    voiced_sample = voiced_only_sample_generator(interval_list, margin, sample, label_dic['sounding'], mode)
    return voiced_sample
