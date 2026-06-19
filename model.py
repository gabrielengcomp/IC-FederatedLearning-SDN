import torch
import torch.nn as nn

NUM_FEATURES = 51

class IDSNet(nn.Module):
    def __init__(self):
        super(IDSNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(NUM_FEATURES, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)