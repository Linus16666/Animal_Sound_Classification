"""
Convert CRNN model to TFLite using ai_edge_torch (PyTorch → TFLite directly).
Saves output to inference_mic/crnn.tflite

Note: onnx_tf (ONNX→TF→TFLite) requires tensorflow_probability which is not
installed. ai_edge_torch produces the same TFLite result from the PyTorch source.
"""

import os
import torch
import litert_torch as ai_edge_torch
import numpy as np
from model import CRNN

OUTPUT_DIR = "inference_mic"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "crnn.tflite")
MODEL_PATH = "models/model_epoch_150.pth"
# Fixed time dimension used for tracing (model is flexible at runtime via LSTM mean pooling)
T_FRAMES = 431  # 5 s at 44100 Hz with n_fft=2048, hop_length=512, center=True

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading PyTorch model...")
model = CRNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

dummy_input = torch.randn(1, 1, 64, T_FRAMES)
print(f"Tracing with input shape: {list(dummy_input.shape)}")

print("Converting to TFLite via ai_edge_torch...")
edge_model = ai_edge_torch.convert(model, (dummy_input,))

edge_model.export(OUTPUT_PATH)
print(f"Saved: {OUTPUT_PATH}")

# --- Sanity check: run inference with the TFLite model ---
print("\nRunning sanity check...")
from ai_edge_litert.interpreter import Interpreter

interpreter = Interpreter(model_path=OUTPUT_PATH)
interpreter.allocate_tensors()

inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]
print(f"  TFLite input  shape : {inp['shape']}  dtype: {inp['dtype'].__name__}")
print(f"  TFLite output shape : {out['shape']}  dtype: {out['dtype'].__name__}")

test_data = np.random.randn(1, 1, 64, T_FRAMES).astype(np.float32)
interpreter.set_tensor(inp["index"], test_data)
interpreter.invoke()
result = interpreter.get_tensor(out["index"])

predicted_class = int(np.argmax(result))
confidence = float(np.max(result))
print(f"  Test prediction: class {predicted_class}  (raw logit max: {confidence:.4f})")
print("\nConversion successful.")
