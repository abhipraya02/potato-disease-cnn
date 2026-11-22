import torch.nn as nn


def conv_block(in_channels, out_channels):
    """Conv -> BatchNorm -> ReLU -> MaxPool. Halves the image size."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    """A simple 4-block CNN trained from scratch (our baseline)."""

    def __init__(self, num_classes=3, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32),     # 224 -> 112
            conv_block(32, 64),    # 112 -> 56
            conv_block(64, 128),   # 56  -> 28
            conv_block(128, 256),  # 28  -> 14
        )
        self.pool = nn.AdaptiveAvgPool2d(1)  # 256 x 14 x 14 -> 256 x 1 x 1
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def count_parameters(model):
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)