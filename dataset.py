import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchaudio

class ESC50Dataset(Dataset):
    def __init__(self, root_dir: str, folds: list[int], sample_rate: int = 44100, n_mels: int = 128):
        self.root_dir = root_dir
        self.audio_dir = os.path.join(root_dir, "audio")
        self.meta = pd.read_csv(os.path.join(root_dir, "meta", "esc50.csv"))
        self.meta = self.meta[self.meta["fold"].isin(folds)].reset_index(drop=True)
        self.sample_rate = sample_rate
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels
        )

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        path = os.path.join(self.audio_dir, row["filename"])
        waveform, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        mel = self.melspec(waveform)
        mel_db = torchaudio.functional.amplitude_to_DB(mel, multiplier=10.0, amin=1e-10, db_multiplier=0)
        return mel_db, row["target"]
