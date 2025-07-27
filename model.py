import torch
import torch.nn as nn
import torch.nn.functional as F

class CRNN(torch.nn.Module):
    def __init__(self, n_mels=128, n_classes=50): #classes depends on how i want to classify, need to thing about that
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        rnn_in = 64 * (n_mels // 4)
        self.rnn = nn.GRU(
            input_size=rnn_in,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        self.fc == nn.Linear(256, n_classes)
        
        
    def forward(self, x):
        x=self.conv1(x)
        x=self.conv2(x)
        B, C, F, T = x.size()
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(B, T, C * F)
        self.rnn.flatten_parameters()
        out, _ = self.rnn(x)
        out = out.mean(dim=1)
        return self.fc(out)