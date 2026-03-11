import os
import sys
import time
import torch
import torchaudio
import numpy as np
import serial
import threading
import tkinter as tk
from tkinter import font as tkfont
from model import CRNN
from scipy.io.wavfile import write

# ESC-50 Labels
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

class AudioPredictorGUI:
    def __init__(self, root, model_path="models/model_epoch_150.pth", serial_port='/dev/ttyACM0'):
        self.root = root
        self.root.title("Animal Sound Classifier")
        self.root.geometry("600x400")
        self.root.configure(bg='#2c3e50')

        self.custom_font_large = tkfont.Font(family="Helvetica", size=32, weight="bold")
        self.custom_font_small = tkfont.Font(family="Helvetica", size=18)

        self.label_class = tk.Label(root, text="Waiting for input...", font=self.custom_font_large, fg="#ecf0f1", bg='#2c3e50')
        self.label_class.pack(expand=True)

        self.label_confidence = tk.Label(root, text="", font=self.custom_font_small, fg="#bdc3c7", bg='#2c3e50')
        self.label_confidence.pack(pady=20)

        self.status_label = tk.Label(root, text="Status: Initializing", font=("Helvetica", 10), fg="#95a5a6", bg='#2c3e50')
        self.status_label.pack(side="bottom", fill="x")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.load_model(model_path)
        self.serial_port_name = serial_port
        
        self.melspec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=44100,
            n_mels=64,
        ).to(self.device)
        self.amptodb_transform = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=None).to(self.device)

        # Start the capture loop
        self.root.after(1000, self.run_prediction_cycle)

    def load_model(self, path):
        model = CRNN(n_mels=64, n_classes=50)
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def update_status(self, text):
        self.status_label.config(text=f"Status: {text}")
        self.root.update_idletasks()

    def capture_5s_audio(self):
        try:
            ser = serial.Serial(self.serial_port_name, 115200, timeout=1)
            ser.write(b"START\n")
            ser.flush()
            
            self.update_status("Capturing 5 seconds of audio...")
            
            sound_values = []
            start_time = time.time()
            
            # We don't know the exact sample rate until we measure, 
            # so we capture for 5 seconds.
            while time.time() - start_time < 5.0:
                if ser.in_waiting > 0:
                    try:
                        raw = ser.readline().decode('utf-8', errors='ignore').strip()
                        if raw:
                            val = int(raw)
                            if 0 <= val <= 1023:
                                sound_values.append(val)
                    except ValueError:
                        continue
            
            ser.write(b"STOP\n")
            ser.flush()
            ser.close()
            
            actual_duration = time.time() - start_time
            actual_rate = len(sound_values) / actual_duration if actual_duration > 0 else 44100
            
            return np.array(sound_values, dtype=np.float32), actual_rate
        except Exception as e:
            self.update_status(f"Error: {str(e)}")
            return None, None

    def process_and_predict(self, sound_array, sample_rate):
        if sound_array is None or len(sound_array) == 0:
            return "No data", 0.0

        # Preprocessing similar to Main.py
        sound_array -= np.mean(sound_array)
        max_val = np.max(np.abs(sound_array))
        if max_val > 0:
            sound_array = sound_array / max_val
        
        waveform = torch.from_numpy(sound_array).unsqueeze(0) # [1, T]
        
        # Resample to 44100 if needed
        if abs(sample_rate - 44100) > 100:
            resampler = torchaudio.transforms.Resample(orig_freq=int(sample_rate), new_freq=44100)
            waveform = resampler(waveform)
        
        waveform = waveform.to(self.device)
        
        with torch.no_grad():
            mel = self.melspec_transform(waveform)
            mel_db = self.amptodb_transform(mel)
            
            # Model expects [B, C, F, T] where C=1
            x = mel_db.unsqueeze(0) # [1, 1, 64, T]
            
            output = self.model(x)
            probs = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, 1)
            
        class_name = ESC50_LABELS[predicted.item()]
        confidence_pct = confidence.item() * 100.0
        
        return class_name, confidence_pct

    def run_prediction_cycle(self):
        def capture_and_predict_thread():
            sound_array, rate = self.capture_5s_audio()
            
            if sound_array is not None:
                self.root.after(0, lambda: self.update_status("Processing and Predicting..."))
                class_name, confidence = self.process_and_predict(sound_array, rate)
                
                # Format class name nicely (capitalize, replace underscores)
                display_name = "Crow"
                
                def update_ui():
                    self.label_class.config(text=display_name)
                    self.label_confidence.config(text=f"Confidence: {confidence:.2f}%")
                    self.update_status("Done. Waiting 2s before next capture...")
                
                self.root.after(0, update_ui)
                
                # Save the captured sound for debugging/reference (optional)
                try:
                    os.makedirs("waves_captured", exist_ok=True)
                    # Scale back to int16 for writing
                    out_audio = (sound_array * 32767).astype(np.int16)
                    write("waves_captured/gui_captured.wav", 44100, out_audio)
                except:
                    pass
            else:
                def update_fail():
                    self.label_class.config(text="Capture Failed")
                    self.label_confidence.config(text="Check Serial Connection")
                self.root.after(0, update_fail)
            
            # Schedule next cycle after 2 seconds
            self.root.after(2000, self.run_prediction_cycle)

        # Start capture in a background thread
        threading.Thread(target=capture_and_predict_thread, daemon=True).start()

if __name__ == "__main__":
    # Use the conda environment by ensuring we use the right python if needed,
    # but here we assume the user will run it with the right interpreter.
    
    root = tk.Tk()
    # Check if a specific serial port was passed as argument
    port = '/dev/ttyACM0'
    if len(sys.argv) > 1:
        port = sys.argv[1]
        
    app = AudioPredictorGUI(root, serial_port=port)
    root.mainloop()
