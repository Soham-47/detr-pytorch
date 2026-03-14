import torch
from torch import nn

class DETR(nn.Module):
    def __init__(self, num_classes=91):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 2048, 3),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.transformer = nn.Transformer(d_model=256, nhead=8)
        self.linear = nn.Linear(256, num_classes)

    def forward(self, x):
        # Barebones forward pass
        return self.linear(torch.randn(1, 100, 256))

if __name__ == "__main__":
    model = DETR()
    print("DETR initialized successfully.")
