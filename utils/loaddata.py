# Imports
import math
import time
import torch
from torch.utils.data import Dataset

# Dataset class
class EEGDataset:
    
    # Constructor
    def __init__(self, eeg_signals_path):
        # Load EEG signals
        loaded = torch.load(eeg_signals_path) #len : 5 ( subject 5 is duplicated)

        self.data = loaded["dataset"] # 11965
        self.labels = loaded["labels"] # (1, 40) ? (40, 1)
        self.images = loaded["images"] # (1996 1)?
        self.means = loaded["means"]
        self.stddevs = loaded["stddevs"]
        # Compute size
        self.size = len(self.data)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Process EEG
        eeg = ((self.data[i]["eeg"].float() - self.means)/self.stddevs).t()
        eeg = eeg[20:450,:] # 21 ~ 450 frame : extract 430 frames to analyze
        # Get label
        label = self.data[i]["label"]
        # Return
        return eeg, label

# Splitter class
class Splitter:

    def __init__(self, dataset, split_path, split_num=0, split_name="train"):
        # Set EEG dataset
        self.dataset = dataset
        # Load split
        loaded = torch.load(split_path)
        self.split_idx = loaded["splits"][split_num][split_name]
        # Filter data
        self.split_idx = [i for i in self.split_idx if 450 <= self.dataset.data[i]["eeg"].size(1) <= 600]
        # Compute size
        self.size = len(self.split_idx)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Get sample from dataset
        eeg, label = self.dataset[self.split_idx[i]]
        # Return
        return eeg, label

class EEGDetectionDataset(Dataset):
    def __init__(self, args, _data_type):
        data_type = 'eeg_{}'.format(_data_type)
        args.datafile = '../{}/{}/{}.pth'.format('eeg_dataset', args.dataset, data_type)
        self.data = torch.load(args.datafile)
        self.len = int(self.data['label'].shape[0])

    def __getitem__(self, i):
        return self.data['data'][i], self.data['label'][i]

    def __len__(self):
        return self.len

