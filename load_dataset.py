import os
import pandas as pd 
import torch
from torch.utils.data import Dataset
import torchaudio

class Dataset(Dataset):
    def __init__(self, root_dir="data\ESC-50-master", folds=list[int], sample_rate=44100, n_mels=128):
        self.root_dir = root_dir
        self.audio_dir= os.path.join(self.root_dir, "audio")
        self.meta = pd.read_csv(os.path.join(self.root_dir, "meta\esc50.csv"))
        self.meta = self.meta[self.meta["fold"].isin(folds)].reset_index(drop=True)
        self.sample_rate = sample_rate
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=n_mels,
        )
        self.amptodb = torchaudio.transforms.AmplitudeToDB(
            stype="power",
            top_db=None
        )
    
    
    def __len__(self):
        return len(self.meta)
    
    def __getitem__(self, idx):
        row=self.meta.iloc[idx]
        path = os.path.join(self.audio_dir, row["filename"])
        waveform, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(waveform, sr, self.sample_rate)
        mel = self.melspec(waveform)
        mel_db = self.amptodb(mel) #db_multiplier=10, amin=1e-10
        return mel_db, row["target"]
    

