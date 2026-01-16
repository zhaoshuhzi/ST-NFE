import torch
from torch.utils.data import Dataset
import numpy as np

class MockDataset(Dataset):
    """
    Universal Mock Data Generator.
    For demonstration purposes, generates synthetic data.
    """
    def __init__(self, mode='pair', num_samples=100, config=None):
        self.mode = mode
        self.config = config
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Simulate EEG (HBN-EEG / ChineseEEG)
        eeg = torch.randn(self.config.EEG_CHANNELS, self.config.TIME_STEPS)
        
        # Simulate MRI (HCP)
        mri = torch.randn(*self.config.MRI_SHAPE)
        
        if self.mode == 'pair':
            # Simulate Text Labels for Meta-Learning
            text = torch.randint(0, self.config.VOCAB_SIZE, (self.config.SEQ_LEN,))
            return eeg, mri, text
        return eeg

def get_dataloader(config, mode='pair'):
    dataset = MockDataset(mode=mode, config=config)
    return torch.utils.data.DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
