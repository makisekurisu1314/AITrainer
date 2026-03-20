import torch
import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self, img_h=64, num_channels=3, num_classes=5):
        super().__init__()
        self.img_h = img_h

        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.rnn_input_size = 128 * (img_h // 4)

        self.rnn = nn.LSTM(
            input_size=self.rnn_input_size,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        self.fc = nn.Linear(128 * 2, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2)
        x = x.contiguous().view(b, w, c * h)
        x, _ = self.rnn(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = CRNN(img_h=64, num_channels=3, num_classes=5)
    dummy_input = torch.randn(8, 3, 64, 64)
    out = model(dummy_input)
    print("Output shape:", out.shape)
