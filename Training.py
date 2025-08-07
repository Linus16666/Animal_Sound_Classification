from load_dataset import Dataset
from model import CRNN
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from pathlib import Path

import wandb

def parse_args():
    p = argparse.ArgumentParser(description="Train CRNN model on ESC-50 dataset")
    p.add_argument("--data", type=Path, default=Path("data/ESC-50-master"))
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--conv1_channels", type=int, default=32)
    p.add_argument("--conv2_channels", type=int, default=64)
    p.add_argument("--conv_kernel1", type=int, default=3)
    p.add_argument("--conv_kernel2", type=int, default=3)
    p.add_argument("--rnn_hidden", type=int, default=128)
    p.add_argument("--rnn_layers", type=int, default=2)
    p.add_argument("--rnn_type", choices=["GRU", "LSTM"], default="GRU")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()



def main():
    args = parse_args()
    device = torch.device(args.device)
    ds = Dataset(root_dir=args.data, folds=[1, 2, 3, 4], sample_rate=44100, n_mels=64)
    ds2 = Dataset(root_dir=args.data, folds=[5], sample_rate=44100, n_mels=64)

    wandb.init(project="animal_sound_classification", config={
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "conv1_channels": args.conv1_channels,
        "conv2_channels": args.conv2_channels,
        "conv_kernel1": args.conv_kernel1,
        "conv_kernel2": args.conv_kernel2,
        "rnn_hidden": args.rnn_hidden,
        "rnn_layers": args.rnn_layers,
        "rnn_type": args.rnn_type,
    })
    config = wandb.config

    train_loader = DataLoader(ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(ds2, batch_size=config.batch_size, shuffle=False)

    model = CRNN(
        n_mels=64,
        n_classes=50,
        conv_channels=(config.conv1_channels, config.conv2_channels),
        conv_kernels=(config.conv_kernel1, config.conv_kernel2),
        rnn_hidden=config.rnn_hidden,
        rnn_layers=config.rnn_layers,
        rnn_type=config.rnn_type,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    model.to(device)
    wandb.watch(model, log="all")

    activations = {}

    def save_activation(name):
        def hook(module, input, output):
            activations[name] = output.detach().cpu()
        return hook

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            module.register_forward_hook(save_activation(name))

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()*x.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{config.epochs}, Loss: {avg_loss:.4f}")

        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc="Validation"):
                x, y = x.to(device), y.to(device)
                output = model(x)
                _, predicted = torch.max(output, 1)
                correct += (predicted == y).sum().item()
        accuracy = correct / len(val_loader.dataset)
        print(f"Validation Accuracy: {accuracy:.4f}")

        log_data = {
            "train_loss": avg_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "val_accuracy": accuracy,
            "epoch": epoch + 1,
        }
        for name, tensor in activations.items():
            log_data[f"activations/{name}"] = wandb.Histogram(tensor)
        wandb.log(log_data, step=epoch + 1)
        activations.clear()

        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")
        if accuracy > 0.95:
            print(f"High accuracy achieved: {accuracy:.4f}, saving model.")
            torch.save(model.state_dict(), "best_model.pth")
            break

    wandb.finish()
        
if __name__ == "__main__":
    main()
