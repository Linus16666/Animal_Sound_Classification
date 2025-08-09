from Sound_sensor_test import capture
import numpy as np
from scipy.io.wavfile import write
import librosa
import matplotlib.pyplot as plt
import os
import pandas as pd 
import torch
from torch.utils.data import Dataset
import torchaudio
import serial
from model import CRNN

#Initializing the sound capturing
#have not changed n_fft and hop_length, will try it with these first

    
class Tranform:
    def __init__(self, root_dir="waves_captured/captured_sound.wav", sample_rate=44100, n_mels=64, sound_caputred=None):
        self.root_dir=root_dir
        self.sampling_r=sample_rate
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sampling_r,
            n_mels=n_mels,
        )
        self.amptodb = torchaudio.transforms.AmplitudeToDB(
            stype="power",
            top_db=None
        )
        self.sound_captured=sound_caputred
    def process_and_print(self):
        sound_array = np.array(self.sound_captured, dtype=np.int16)
        write(self.root_dir, self.sampling_r, sound_array)
        print("saved")
        waveform, sr = torchaudio.load(self.root_dir)
        if sr != self.sampling_r:
            waveform = torchaudio.transforms.Resample(waveform, sr, self.sampling_r)
        mel = self.melspec(waveform)
        mel_db = self.amptodb(mel)
        self.mel_db = self.amptodb(mel)
        return mel_db
    def model_predict(self):
        if torch.cuda.is_available():
            device="cuda"
        else:
            device="cpu"
        input=self.mel_db
        input.unsqueeze(0)
        input.to(device)
        model=CRNN()
        model.load_state_dict(torch.load("path", map_location=device))
        with torch.no_grad():
            output = model(input)
            _, predicted = torch.max(output, 1)
        return predicted
        
def main():
    print("started_everything")
    sound_captured=capture()
    print("sound captured")
    transform=Tranform(sound_caputred=sound_captured)
    mel_spectrogram = transform.process_and_print()
    output=transform.model_predict()
    print(output)
    plt.imshow(mel_spectrogram.squeeze().numpy(), origin='lower', aspect='auto')
    plt.title('Mel Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()
    if mel_spectrogram is not None:
        print(mel_spectrogram)
    return mel_spectrogram


if __name__ == "__main__":
    main()