from load_dataset import Dataset
import matplotlib.pyplot as plt
import torch

ds = Dataset(root_dir="data\ESC-50-master", folds=[1, 2, 3], sample_rate=44100, n_mels=128)

loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True)

mel_db, label = next(iter(loader))

mel_db = mel_db.squeeze(0).squeeze(0)
mel_np = mel_db.numpy()

plt.figure(figsize=(10, 4))
plt.imshow(mel_np, aspect='auto', origin='lower', cmap='coolwarm')
plt.title(f'Mel Spectrogram - Label: {label.item()}')
plt.colorbar(format='%+2.0f dB')
plt.xlabel('Time')
plt.ylabel('Mel Frequency')
plt.tight_layout()
plt.show()
