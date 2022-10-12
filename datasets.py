import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random 
import torch
import os

SEED =0
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

class EEGDataset(Dataset):
    def __init__(self, csv_file, transform=None, is_3D=True):
        self.data = pd.read_csv(csv_file)
        self.diagnosis_map = {'MDD': 1.0, 'Health': 0.0}
        self.transform = transform
        self.is_3D =is_3D

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        raw = self.data.iloc[idx]
        eeg = np.load("../"+raw['file'])
        label = self.diagnosis_map[raw['diagnosis']]

        if self.transform:
            eeg = self.transform(eeg)
        if self.is_3D:
            return eeg[np.newaxis, :, :, :], label
        else:
            return eeg, label