import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from dataset import ESC50Dataset
from model import CRNN


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate CRNN on ESC-50")
    p.add_argument("--data", type=Path, default=Path("data/ESC-50-master"))
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--weights", type=Path, default=Path("crnn.pth"))
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    test_ds = ESC50Dataset(str(args.data), folds=[5])
    test_dl = DataLoader(test_ds, batch_size=args.batch_size)

    model = CRNN(n_mels=128, n_classes=50).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    correct = 0
    with torch.no_grad():
        for x, y in test_dl:
            x = x.to(device)
            y = y.to(device)
            preds = model(x)
            correct += (preds.argmax(dim=1) == y).sum().item()
    acc = correct / len(test_ds)
    print(f"Test accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
