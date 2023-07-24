# -*- coding: utf-8 -*-
"""
Created on Mon May 25 13:34:01 2020

Some basic functions for processing wav files with speech recordings, and 
extract voice and speech features. 
It recommneded that the recordings will start with a beep sound at the beginning
so their volume can be normalized.

@author: YaelRL (File function by Itai Barkai)
"""
import os
from os.path import isfile, join
from os import listdir
import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
import scipy.signal
import matplotlib.pyplot as plt
from RMS_EF import RMS_env
import pydub
from pydub import AudioSegment, silence
import statistics 
from statistics import mean

def File(path, name):
    #loading a wav file
    #System Param

    MaxInt16 = 32767.0    
    
    filepath = os.path.abspath(path + '\\' + name )
    
    rate , b = wav.read(filepath)
    #factor = MaxInt16 if b.dtype == np.int16 else 1.0

    if (len(b.shape) == 2):    #Stereo
        XL = np.divide(b[:, 0], MaxInt16)
        XR = np.divide(b[:, 1], MaxInt16)
        X = 0.5 * (XL + XR)
        print ('\n!! Stereo file truncated to mono')
    else:
        X = b
        X = np.divide(X, MaxInt16)
     
    return (X, rate)


def cap_outliers(X, upper=0.95, lower=0.05):
#     #adjusts values that are higher than the upper percentile
#     #or smaller than the lower percentile to the respective percentile
#    (this was not used in our data processing)
    lower_bound = np.quantile(X, q=lower, method='higher')
    upper_bound = np.quantile(X, q=upper, method='lower')
    X[X < lower_bound] = lower_bound
    X[X > upper_bound] = upper_bound
    return X

def normalize_wav(path, name):
    #input: a wav file path and a file name
    #normalizes the file to -14 dB using the beep sound,
    # and eliminates the beep
    # output: numpy array representing wav file data, and sample rate

    #fs is the sampling rate (normally 44100)
    X, fs = File(path, name)
        
    # identifying the beep sound using spectrogram
    f, t, spec = scipy.signal.spectrogram(X, fs=fs)

    # beep frequency is 1000 Hz, represented by f[6] (or spec[6])
    #looking for the timing where 1000 Hz frequency is dominant (75% or more of frequencies)
    beep_times = t[spec[6, :] > 0.75*np.sum(spec[:, :], axis=0)]
    
    #picking the timing for the beginning and end of the beep
    start_beep, end_beep = int(beep_times[0] * fs), int(beep_times[-1] * fs)
    
    #safety check for beep length (should be around 0.3 seconds)
    # if ((end_beep/fs - start_beep/fs) < 0.2):
    #     print(str(name) + " beep too short " + str(end_beep/fs - start_beep/fs))
    # elif ((end_beep/fs - start_beep/fs) > 0.4):
    #     print(str(name) + "beep too long " + str(end_beep/fs - start_beep/fs))

    #finding beep intensity in the normalized recording
    beep_intensity = np.mean(X[start_beep:end_beep]**2)**0.5
    
    #normalizing the data to -14dB (by a factor of 0.08)
    X_norm = X*0.08/beep_intensity 
    #cutting off the beep
    X_processed = X_norm[end_beep+50:]
    return X_processed, fs


# X, fs = normalize_wav('audio/original_wavs', '10.wav')
# wav.write("audio/10_proc.wav", fs, X)

def normalize_dir(path, newpath):
    #normalizes all the wav files in a given directory and stores the
    #processed files in a new directory
    
    #making a list of all wav file names in the directory (without sub-directories)
    files = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith('.wav')]
    for file_name in files:
        X, fs = normalize_wav(path, file_name)
        wav.write(newpath+'/'+file_name, fs, X.astype(np.float32))



def extract_audio_params(dirpath):
    #input: path of a wav files' directory
    #the function extracts audio parameters and outputs
    # them to an excell file, in the same directory
    Fs = float(44100)
    #initializing parameters' lists
    peak = ["peak"]
    peak_sd = ["peak_sd"]
    peak_diff = ["peak_diff"]
    rms = ["rms"]
    cf = ["cf"]
    id = ['id']

    #making a list of all wav file names in the directory (without sub-directories)
    files = [f for f in listdir(dirpath) if isfile(join(dirpath, f)) and f.endswith('.wav')]
    for file_name in files:
        X, rate = File(dirpath, file_name)
        if (int(rate) != int(Fs)):
            print("!! Load file at ", int(rate)," sample rate, while sample rate is", int(Fs))

        Xabs = np.abs(X)
    #Xpeak = RMS_env(X,Fs,.5)
        XRMS = RMS_env(X, Fs, 10)

    #Find 1000 top peak values
        size = 1000
        peak_indices = np.argpartition(-Xabs, size)  # find indices
        result_args = peak_indices[:size]
        temp = np.partition(-Xabs, size)  # finds values based on indices
        peaks = -temp[:size]

        PEAK = np.average(peaks)
        PEAK_SD = np.std(peaks)
        PEAK_DIFF = np.max(peaks) - np.min(peaks)
        RMS = np.average(XRMS)
        CF = 20*np.log10(PEAK/RMS)
                
        # saving the file number only (without '.wav')
        id.append(file_name[:-4])
        peak.append(PEAK)
        peak_sd.append(PEAK_SD)
        peak_diff.append(PEAK_DIFF)
        rms.append(RMS)
        cf.append(CF)
    
    
    #saving output to csv
    np.savetxt(dirpath + '/audio_parameters.csv', [p for p in zip(id, peak, peak_sd, rms, cf)], delimiter=',', fmt='%s')


def extract_silence_params(dirpath):
    #the function extracts silence/voice parameters and outputs
    # them to an excel file, in the same directory
    
    #obtaining a list of audio files in the directory (without directories)
    file_names = [f for f in listdir(dirpath) if isfile(join(dirpath, f)) and f.endswith('.wav')]

    #initializing parameter lists
    id = ['id']
    silence_sd = ["silence_sd"]
    not_silence_sd = ['not_silence_sd']
    silence_num = ['silence_num']
    silence_total = ['silence_total_length']
    not_silence_total = ['not_silence_total_length']
    silence_avg = ['silence_avg_length']
    not_silence_avg = ['not_silence_avg_length']
    speech_peak_sd = ["speech_peak_sd"]
    speech_rms_sd = ["speech_rms_sd"]

    
    for file_name in file_names:
        wav_audio = AudioSegment.from_file(dirpath+'/'+file_name, format="wav")
        dBFS = wav_audio.dBFS
        #getting a list of silence durations: [[start, stop],..]
        silences = silence.detect_silence(wav_audio, min_silence_len=100, silence_thresh=dBFS-16)
        
        #creating a list of voice durations (inverse of silence durations)
        not_silences = [[silences[i][1], silences[i+1][0]] for i in range(len(silences)-1)]

        #deleting the first and last silence
        silences = silences[1:-1]

        #create a feature for silence_length_sd and not_silence_length_sd
        silences_len = [(silences[i][1] - silences[i][0]) for i in range(len(silences))]
        not_silences_len = [not_silences[i][1] - not_silences[i][0] for i in range(len(not_silences))]
        #standard deviation of silence/not silence segments
        sil_sd = np.std(silences_len[1:-1]) #all silences except for the first and last
        not_sil_sd = np.std(not_silences_len)
        #number of silence/not silence segments
        sil_num = len(silences)
        
        # creating a list of max volume for each speech segment
        speech_max = [wav_audio[s[0]:s[1]].max for s in not_silences]
        #calculating sd and storing it for output
        speech_peak_sd.append(np.std(speech_max))

        # creating a list of rms for each speech segment
        speech_rms = [wav_audio[s[0]:s[1]].rms for s in not_silences]
        #calculating sd and storing it for output
        speech_rms_sd.append(np.std(speech_rms))

        #saving the extracted parameters in lists
        #saving the file number only (without '.wav')
        id.append(file_name[:-4]) #file name is the patient's id. 
        silence_sd.append(sil_sd)
        not_silence_sd.append(not_sil_sd)
        silence_num.append(sil_num)
        silence_total.append(sum(silences_len))
        not_silence_total.append(sum(not_silences_len))
        silence_avg.append(mean(silences_len))
        not_silence_avg.append(mean(not_silences_len))
        
    np.savetxt(dirpath + '/silence_parameters.csv', [p for p in zip(id, silence_sd, not_silence_sd, silence_num, silence_total, not_silence_total, silence_avg, not_silence_avg, speech_rms_sd, speech_peak_sd)], delimiter=',', fmt='%s')



Fs = float(44100)
#initializing parameters' lists
peak = ["peak"]
dirpath= "audio/normalized_wavs" 
file_name="44.wav"
X, rate = File(dirpath, file_name)
Xabs = np.abs(X)
    #Xpeak = RMS_env(X,Fs,.5)
XRMS = RMS_env(X, Fs, 10)

    #Find 1000 top peak values
size = 1000
peak_indices = np.argpartition(-Xabs, size)  # find indices
result_args = peak_indices[:size]
temp = np.partition(-Xabs, size)  # finds values based on indices
peaks = -temp[:size]

PEAK = np.average(peaks)
PEAK_SD = np.std(peaks)

