import torch
import litert_torch
from model import CRNN
# Load your model
model = CRNN()
model.load_state_dict(torch.load("models/model_epoch_150.pth", map_location="cpu"))
model.eval()

dummy_input = (torch.randn(1, 1, 64, 400))  # match your input shape

edge_model = litert_torch.convert(model, dummy_input)
edge_model.export("crnn.tflite")
print("Done!")
print('Done!')