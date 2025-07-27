import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm

from dataset import ESC50Dataset
from model import CRNN


def parse_args():
    p = argparse.ArgumentParser(description="Train CRNN on ESC-50")
    p.add_argument("--data", type=Path, default=Path("data/ESC-50-master"))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    train_ds = ESC50Dataset(str(args.data), folds=[1,2,3,4])
    val_ds = ESC50Dataset(str(args.data), folds=[5])
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size)

    model = CRNN(n_mels=128, n_classes=50).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for x, y in tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs}"):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        avg_loss = total_loss / len(train_ds)
        print(f"Epoch {epoch+1} training loss: {avg_loss:.4f}")

        # validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in val_dl:
                x = x.to(device)
                y = y.to(device)
                preds = model(x)
                correct += (preds.argmax(dim=1) == y).sum().item()
        acc = correct / len(val_ds)
        print(f"Validation accuracy: {acc:.4f}")
    torch.save(model.state_dict(), "crnn.pth")


if __name__ == "__main__":
    main()
