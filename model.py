import torch
import torch.nn as nn
import torch.nn.functional as F


class CRNN(torch.nn.Module):
    def __init__(
        self,
        n_mels: int = 64,
        n_classes: int = 50,
        conv_channels=(32, 64),
        conv_kernels=(3, 3),
        rnn_hidden: int = 128,
        rnn_layers: int = 2,
        rnn_type: str = "GRU",
    ):
        super().__init__()
        k1, k2 = conv_kernels
        c1, c2 = conv_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, c1, k1, padding=k1 // 2),
            nn.BatchNorm2d(c1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(c1, c2, k2, padding=k2 // 2),
            nn.BatchNorm2d(c2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        rnn_in = c2 * (n_mels // 4)
        RNN = getattr(nn, rnn_type)
        self.rnn = RNN(
            input_size=rnn_in,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )

        self.fc = nn.Linear(rnn_hidden * 2, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        B, C, F, T = x.size()
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(B, T, C * F)
        self.rnn.flatten_parameters()
        out, _ = self.rnn(x)
        out = out.mean(dim=1)
        return self.fc(out)
