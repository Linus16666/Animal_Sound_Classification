from Sound_sensor_test import capture, send_label
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

ESC50_LABELS = [
    "dog", "rooster", "pig", "cow", "frog", "cat", "hen", "insects",
    "sheep", "crow", "rain", "sea_waves", "crackling_fire", "crickets",
    "chirping_birds", "water_drops", "wind", "pouring_water", "toilet_flush",
    "thunderstorm", "crying_baby", "sneezing", "clapping", "breathing",
    "coughing", "footsteps", "laughing", "brushing_teeth", "snoring",
    "drinking_sipping", "door_wood_knock", "mouse_click", "keyboard_typing",
    "door_wood_creaks", "can_opening", "washing_machine", "vacuum_cleaner",
    "clock_alarm", "clock_tick", "glass_breaking", "helicopter", "chainsaw",
    "siren", "car_horn", "engine", "train", "church_bells", "airplane",
    "fireworks", "hand_saw",
]

# Human-readable labels for the 16x2 LCD (max 16 chars each)
ESC50_DISPLAY = {
    "dog":              "Crow",
    "rooster":          "Crow",
    "pig":              "Crow",
    "cow":              "Crow",
    "frog":             "Crow",
    "cat":              "Crow",
    "hen":              "Crow",
    "insects":          "Crow",
    "sheep":            "Crow",
    "crow":             "Crow",
    "rain":             "Crow",
    "sea_waves":        "Crow",
    "crackling_fire":   "Crow",
    "crickets":         "Crow",
    "chirping_birds":   "Crow",
    "water_drops":      "Crow",
    "wind":             "Crow",
    "pouring_water":    "Crow",
    "toilet_flush":     "Crow",
    "thunderstorm":     "Crow",
    "crying_baby":      "Crow",
    "sneezing":         "Crow",
    "clapping":         "Crow",
    "breathing":        "Crow",
    "coughing":         "Crow",
    "footsteps":        "Crow",
    "laughing":         "Crow",
    "brushing_teeth":   "Crow",
    "snoring":          "Crow",
    "drinking_sipping": "Crow",
    "door_wood_knock":  "Crow",
    "mouse_click":      "Crow",
    "keyboard_typing":  "Crow",
    "door_wood_creaks": "Crow",
    "can_opening":      "Crow",
    "washing_machine":  "Crow",
    "vacuum_cleaner":   "Crow",
    "clock_alarm":      "Crow",
    "clock_tick":       "Crow",
    "glass_breaking":   "Crow",
    "helicopter":       "Crow",
    "chainsaw":         "Crow",
    "siren":            "Crow",
    "car_horn":         "Crow",
    "engine":           "Crow",
    "train":            "Crow",
    "church_bells":     "Crow",
    "airplane":         "Crow",
    "fireworks":        "Crow",
    "hand_saw":         "Crow",
}

class Tranform:
    def __init__(self, root_dir="waves_captured/captured_sound.wav", sample_rate=44100, n_mels=64, sound_caputred=None):
        self.root_dir=root_dir
        self.sampling_r=sample_rate
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=44100,   # always 44100 for model compatibility
            n_mels=n_mels,
        )
        self.amptodb = torchaudio.transforms.AmplitudeToDB(
            stype="power",
            top_db=None
        )
        self.sound_captured=sound_caputred
    def process_and_print(self):
        # Remove DC bias (~512 for GY-MAX4466), normalize to [-1, 1] range for transforms
        sound_array = np.array(self.sound_captured, dtype=np.float32)
        sound_array -= np.mean(sound_array)                   # center around zero
        max_val = np.max(np.abs(sound_array))
        if max_val > 0:
            sound_array = sound_array / max_val               # scale to [-1, 1] for transforms

        # Write to file for record at actual rate
        write_array = (sound_array * 32767.0).astype(np.int16)
        write(self.root_dir, self.sampling_r, write_array)    

        # Convert to tensor directly instead of loading from disk
        waveform = torch.from_numpy(sound_array).unsqueeze(0)
        sr = self.sampling_r

        if sr != 44100:                                        # resample to match model
            waveform = torchaudio.transforms.Resample(sr, 44100)(waveform)
        mel = self.melspec(waveform)
        mel_db = self.amptodb(mel)
        self.mel_db = mel_db                                   # reuse, don't recompute
        return mel_db
    def model_predict(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        x = self.mel_db.unsqueeze(0)
        x = x.to(dtype=torch.float32)    # assign back
        x = x.to(device)                  # assign back
        model = CRNN()
        model.load_state_dict(torch.load("models/model_epoch_150.pth", map_location=device))
        model.to(device)
        model.eval()                       # puts BatchNorm into inference mode
        with torch.no_grad():
            output = model(x)
            probs = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, 1)
        class_name = ESC50_LABELS[predicted.item()]
        confidence_pct = confidence.item() * 100.0
        return class_name, confidence_pct
        
def main():
    sound_captured, actual_sample_rate = capture()
    transform = Tranform(sound_caputred=sound_captured, sample_rate=actual_sample_rate)
    mel_spectrogram = transform.process_and_print()
    class_name, confidence = transform.model_predict()
    print(f"Predicted: {class_name}  ({confidence:.1f}% confidence)")
    display_text = ESC50_DISPLAY.get(class_name, class_name[:16])
    send_label(display_text, confidence)
    plt.imshow(mel_spectrogram.squeeze().numpy(), origin='lower', aspect='auto')
    plt.title('Mel Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()