import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(identity)
        out += identity
        out = self.relu(out)
        return out


class LightweightHRNet(nn.Module):
    def __init__(self, num_keypoints=21, width=18):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, 2, 1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, 1, 1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        )

        self.layer1 = nn.Sequential(
            BasicBlock(width, width),
            BasicBlock(width, width)
        )

        self.stage2_branch1 = nn.Sequential(
            BasicBlock(width, width),
            BasicBlock(width, width)
        )
        self.stage2_branch2 = nn.Sequential(
            nn.Conv2d(width, width*2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(width*2),
            nn.ReLU(inplace=True),
            BasicBlock(width*2, width*2)
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(width + width*2, width*2, 1, 1, 0),
            nn.BatchNorm2d(width*2),
            nn.ReLU(inplace=True)
        )

        self.head = nn.Conv2d(width*2, num_keypoints, 1, 1, 0)

    def forward(self, x):
        x = self.stem(x)
        x1 = self.layer1(x)
        b1 = self.stage2_branch1(x1)
        b2 = self.stage2_branch2(x1)
        b2_up = F.interpolate(b2, size=b1.shape[2:], mode='bilinear', align_corners=False)
        fused = self.fuse(torch.cat([b1, b2_up], dim=1))
        out = self.head(fused)
        out = F.interpolate(out, size=(64, 64), mode='bilinear', align_corners=False)
        return out


if __name__ == "__main__":
    model = LightweightHRNet()
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(y.shape)
