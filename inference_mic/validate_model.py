#!/usr/bin/env python3
"""
validate_model.py — PC-side sanity check for crnn.tflite

Loads the TFLite model, runs the real captured WAV through the same
mel-spectrogram pipeline used in Main.py, and prints the predicted label.

Usage:
    python3 inference_mic/validate_model.py

Requirements:
    pip install ai-edge-litert torchaudio torch
"""

import sys
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T

# TFLite interpreter
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    print("ERROR: ai_edge_litert not installed. Run: pip install ai-edge-litert")
    sys.exit(1)

# ── ESC-50 class labels (same order as training) ─────────────────────────────
ESC50_LABELS = [
    "dog", "rooster", "pig", "cow", "frog",
    "cat", "hen", "insects", "sheep", "crow",
    "rain", "sea_waves", "crackling_fire", "crickets", "chirping_birds",
    "water_drops", "wind", "pouring_water", "toilet_flush", "thunderstorm",
    "crying_baby", "sneezing", "clapping", "breathing", "coughing",
    "footsteps", "laughing", "brushing_teeth", "snoring", "drinking_sipping",
    "door_wood_knock", "mouse_click", "keyboard_typing", "door_wood_creaks", "can_opening",
    "washing_machine", "vacuum_cleaner", "clock_alarm", "clock_tick", "glass_breaking",
    "helicopter", "chainsaw", "siren", "car_horn", "engine",
    "train", "church_bells", "airplane", "fireworks", "hand_saw",
]

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH = "inference_mic/crnn.tflite"
WAV_PATH   = "waves_captured/captured_sound.wav"

# ── Mel spectrogram parameters (must match onnx_to_tflite.py tracing) ────────
SAMPLE_RATE = 44100
N_FFT       = 2048
HOP_LENGTH  = 512
N_MELS      = 64
T_FRAMES    = 431   # 5 s @ 44100 Hz / hop 512 ≈ 431

def load_and_preprocess(wav_path: str) -> np.ndarray:
    """Load WAV, resample if needed, compute mel spectrogram → [1,1,64,431]."""
    waveform, sr = torchaudio.load(wav_path)

    # Mix to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to 44100 if needed
    if sr != SAMPLE_RATE:
        waveform = T.Resample(sr, SAMPLE_RATE)(waveform)

    mel_transform = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    amp_to_db = T.AmplitudeToDB()

    mel = mel_transform(waveform)   # [1, n_mels, T]
    mel_db = amp_to_db(mel)         # [1, n_mels, T]

    # Pad or trim to exactly T_FRAMES
    T_actual = mel_db.shape[-1]
    if T_actual < T_FRAMES:
        pad = T_FRAMES - T_actual
        mel_db = torch.nn.functional.pad(mel_db, (0, pad))
    else:
        mel_db = mel_db[..., :T_FRAMES]

    # Shape: [1, 1, N_MELS, T_FRAMES]
    inp = mel_db.unsqueeze(0).numpy().astype(np.float32)
    return inp


def run_inference(model_path: str, inp: np.ndarray) -> tuple[str, float]:
    interp = Interpreter(model_path=model_path)
    interp.allocate_tensors()

    in_detail  = interp.get_input_details()[0]
    out_detail = interp.get_output_details()[0]

    expected = tuple(in_detail["shape"])
    actual   = tuple(inp.shape)
    if expected != actual:
        print(f"WARNING: model expects input shape {expected}, got {actual}")

    interp.set_tensor(in_detail["index"], inp)
    interp.invoke()

    logits = interp.get_tensor(out_detail["index"])[0]   # [50]

    # Softmax
    logits -= logits.max()
    probs   = np.exp(logits)
    probs  /= probs.sum()

    class_idx  = int(np.argmax(probs))
    confidence = float(probs[class_idx]) * 100.0
    label      = ESC50_LABELS[class_idx] if class_idx < len(ESC50_LABELS) else f"class_{class_idx}"
    return label, confidence


def main():
    print(f"Loading WAV:   {WAV_PATH}")
    print(f"Loading model: {MODEL_PATH}")

    inp = load_and_preprocess(WAV_PATH)
    print(f"Input tensor shape: {inp.shape}")

    label, confidence = run_inference(MODEL_PATH, inp)
    print(f"\nPredicted: {label}  ({confidence:.1f}%)")


if __name__ == "__main__":
    main()
