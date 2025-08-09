from Sound_sensor_test import main
import numpy as np
from scipy.io.wavefile import write
import librosa
import matplotlib.pyplot as plt
import os
import pandas as pd 
import torch
from torch.utils.data import Dataset
import torchaudio
import serial

#Initializing the sound capturing
sound_caputred=main()
print("Sound Capturing function started")
#have not changed n_fft and hop_length, will try it with these first

    
class Tranform:
    def __init__(self, root_dir="waves_captured/captured_sound.wav", sample_rate=44100, n_mels=64, sound_caputred=sound_caputred):
        self.root_dir=root_dir
        self.sampling_r=sample_rate
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=n_mels,
        )
        self.amptodb = torchaudio.transforms.AmplitudeToDB(
            stype="power",
            top_db=None
        )
        self.sound_captured=sound_caputred
    def process_and_print(self):
        sound_array = np.array(self.sound_captured, dtype=np.int16)
        write('waves_captured/captured_sound.wav', 16000, sound_array)
        print("saved")
        waveform, sr = torchaudio.load(self.root_dir)
        if sr != self.sampling_r:
            waveform = torchaudio.transforms.Resample(waveform, sr, self.sample_rate)
        mel = self.melspec(waveform)
        mel_db = self.amptodb(mel)
        plt.figure(figsize=(10, 4))
        plt.imshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', cmap='coolwarm')
        plt.title('Mel Spectrogram')
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        plt.show()
        return mel_db
    
sample = Tranform(sound_caputred)

if __name__==main:
    sample.process_and_print()