import torch
from torch import nn
import numpy as np
import random 
import os
from deform_conv import DeformConv3d, DeformableConv2d

SEED =0
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

class EEG3DAutoencoder(nn.Module):
    def __init__(self, hidden_layers):
        super(EEG3DAutoencoder, self).__init__()
        self.hidden_layers = hidden_layers
        self.encoder = nn.Sequential(
            nn.Conv3d(1, self.hidden_layers[0], 3, stride=1, padding=1),
            nn.BatchNorm3d(self.hidden_layers[0]),
            nn.ELU(True),
            nn.MaxPool3d(2, stride=2),
            nn.Conv3d(self.hidden_layers[0], self.hidden_layers[1], 3, stride=1, padding=1),
            nn.BatchNorm3d(self.hidden_layers[1]),
            nn.ELU(True),
            nn.MaxPool3d(2, stride=2)

        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(self.hidden_layers[1], self.hidden_layers[0], 2, stride=2),
            nn.BatchNorm3d(self.hidden_layers[0]),
            nn.ELU(True),
            nn.ConvTranspose3d(self.hidden_layers[0], 1, 2, stride=2),
            nn.BatchNorm3d(1),
            nn.ELU(True)
        )
        self.pool = nn.AdaptiveAvgPool3d((3, 6, 8))
        self.fc = nn.Linear(3 * 6 * 8 * self.hidden_layers[1], 2)

    def forward(self, x):
        original_shapes = x.shape
        codes = self.encoder(x)
        classes = self.pool(codes)
        classes = self.fc(classes.view(-1, 3 * 6 * 8 * self.hidden_layers[1]))

        x = self.decoder(codes)
        x = nn.functional.interpolate(x, size=original_shapes[2:])
        return codes, x, classes


class EEGAutoencoder(nn.Module):
    def __init__(self, hidden_layers):
        super(EEGAutoencoder, self).__init__()
        self.hidden_layers = hidden_layers
        self.encoder = nn.Sequential(
            nn.Conv2d(15, self.hidden_layers[0], 3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_layers[0]),
            nn.ELU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(self.hidden_layers[0], self.hidden_layers[1], 3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_layers[1]),
            nn.ELU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(self.hidden_layers[1], self.hidden_layers[2], 3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_layers[2]),
            nn.ELU(True),
            nn.MaxPool2d(2, stride=2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_layers[2], self.hidden_layers[1], 2, stride=2),
            nn.ELU(True),
            nn.BatchNorm2d(self.hidden_layers[1]),
            nn.ConvTranspose2d(self.hidden_layers[1], self.hidden_layers[0], 2, stride=2),
            nn.ELU(True),
            nn.BatchNorm2d(self.hidden_layers[0]),
            nn.ConvTranspose2d(self.hidden_layers[0], 15, 2, stride=2),
            nn.BatchNorm2d(15),
            nn.ELU(True)
        )
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.fc = nn.Linear(8 * 8 * self.hidden_layers[2], 2)

    def forward(self, x):
        original_shapes = x.shape
        codes = self.encoder(x)
        classes = self.pool(codes)
        classes = self.fc(classes.view(-1, 8 * 8 * self.hidden_layers[2]))

        x = self.decoder(codes)
        x = nn.functional.interpolate(x, size=original_shapes[2:])
        return codes, x, classes


class EEGDeformAutoencoder(nn.Module):
    def __init__(self, hidden_layers):
        super(EEGDeformAutoencoder, self).__init__()
        self.hidden_layers = hidden_layers
        self.encoder = nn.Sequential(
            DeformableConv2d(15, self.hidden_layers[0], 3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_layers[0]),
            nn.ELU(True),
            nn.MaxPool2d(2, stride=2),
            DeformableConv2d(self.hidden_layers[0], self.hidden_layers[1], 3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_layers[1]),
            nn.ELU(True),
            nn.MaxPool2d(2, stride=2),
            # DeformableConv2d(self.hidden_layers[1], self.hidden_layers[2], 3, stride=1, padding=1),
            # nn.BatchNorm2d(self.hidden_layers[2]),
            # nn.ELU(True),
            # nn.MaxPool2d(2, stride=2),
            # DeformableConv2d(self.hidden_layers[2], self.hidden_layers[3], 3, stride=1, padding=1),
            # nn.BatchNorm2d(self.hidden_layers[3]),
            # nn.ELU(True),
            # nn.MaxPool2d(2, stride=2),
        )
        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(self.hidden_layers[3], self.hidden_layers[2], 2, stride=2),
            # nn.ELU(True),
            # nn.BatchNorm2d(self.hidden_layers[2]),
            
            # nn.ConvTranspose2d(self.hidden_layers[2], self.hidden_layers[1], 2, stride=2),
            # nn.ELU(True),
            # nn.BatchNorm2d(self.hidden_layers[1]),
            nn.ConvTranspose2d(self.hidden_layers[1], self.hidden_layers[0], 2, stride=2),
            nn.ELU(True),
            nn.BatchNorm2d(self.hidden_layers[0]),
            nn.ConvTranspose2d(self.hidden_layers[0], 15, 2, stride=2),
            nn.BatchNorm2d(15),
            nn.ELU(True)
        )
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.fc = nn.Linear(8 * 8 * self.hidden_layers[1], 2)

    def forward(self, x):
        original_shapes = x.shape
        codes = self.encoder(x)
        classes = self.pool(codes)
        classes = self.fc(classes.view(-1, 8 * 8 * self.hidden_layers[1]))

        x = self.decoder(codes)
        x = nn.functional.interpolate(x, size=original_shapes[2:])
        return codes, x, classes


class EEG3DDeformAutoencoder(nn.Module):
    def __init__(self):
        super(EEG3DDeformAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            DeformConv3d(1, 32, 3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ELU(True),
            nn.MaxPool3d(2, stride=2),  # b, 16, 5, 5
            DeformConv3d(32, 64, 3, stride=1, padding=1),  # b, 8, 3, 3
            nn.BatchNorm3d(64),
            nn.ELU(True),
            nn.MaxPool3d(2, stride=2)
        )
        self.decoder = nn.Sequential(

            nn.ConvTranspose3d(64, 32, 2, stride=2),  # b, 16, 5, 5
            nn.BatchNorm3d(32),
            nn.ELU(True),
            nn.ConvTranspose3d(32, 1, 2, stride=2),  # b, 8, 15, 15
            nn.BatchNorm3d(1),
            nn.ELU(True)
        )
        self.pool = nn.AdaptiveAvgPool3d((3, 6, 8))
        self.fc = nn.Linear(3 * 6 * 8 * 64, 2)

    def forward(self, x):
        original_shapes = x.shape
        codes = self.encoder(x)
        classes = self.pool(codes)
        classes = self.fc(classes.view(-1, 3 * 6 * 8 * 64))

        x = self.decoder(codes)
        x = nn.functional.interpolate(x, size=original_shapes[2:])
        return codes, x, classes


