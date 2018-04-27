from __future__ import division # IN PYTHON3, IT IS NOT NEEDED
import argparse
import os
import torch
import numpy as np

print("{:*^30}".format('WELCOME TO EEG DATASET'))

parser = argparse.ArgumentParser(description="Classify_to_Detect")
parser.add_argument('-vlen', '--valid-seqlen', default=400, type=int, help="Maximum valid length of EEG for every class")
parser.add_argument('-nc', '--num-cls', default=5, type=int, help="Num of EEG class signals per each data")
parser.add_argument('-tl', '--total-length', default=1500, type=int, help="Total length of single data")
opt = parser.parse_args()

print("ARGUMENT PARSING COMPLETE")

# Dataset class
class EEGDataset:
    
    # Constructor
    def __init__(self, eeg_signals_path, opt):
        # Load EEG signals
        loaded = torch.load(eeg_signals_path) #len : 5 ( subject 5 is duplicated)
        #print(loaded)

        self.data = loaded["dataset"] # 11965
        self.labels = loaded["labels"] # (1, 40) ? (40, 1)
        self.images = loaded["images"] # (1996 1)?
        self.means = loaded["means"]
        self.stddevs = loaded["stddevs"]
        # Compute size
        self.size = len(self.data)
        self.valid_length = opt.valid_seqlen


    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Process EEG
        eeg = ((self.data[i]["eeg"].float() - self.means)/self.stddevs).t()
        eeg = eeg[20:20+self.valid_length,:] # 21 ~ 450 frame : extract 430 frames to analyze
        # Get label
        label = self.data[i]["label"]
        # Return
        return eeg, label

# Splitter class
class Splitter:

    def __init__(self, dataset, split_path, split_name="train"):
        # Set EEG dataset
        self.dataset = dataset
        # Load split
        loaded = torch.load(split_path)
        self.split_idx = loaded["splits"][0][split_name]
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

def get_eeg(_eeg_data, cls_len):
    if cls_len == 400:
        eeg_data = _eeg_data[:, :]

    if cls_len == 300:
        _take = range(400)
        waste = [i*4 for i in range(100)]
        take = [i for i in _take if i not in waste]
        eeg_data = _eeg_data[take, :]

    if cls_len == 200:
        eeg_data = _eeg_data[0:400:2, :]

    if cls_len == 100:
        take = [i*4 for i in range(100)]
        eeg_data = _eeg_data[take, :]

    return eeg_data




# Load dataset
eeg_path = './{}/{}.pth'.format('cvpr17', 'eeg_signals_128_sequential_band_all_with_mean_std')
splits_path = './{}/{}.pth'.format('cvpr17', 'splits_by_image')
dataset = EEGDataset(eeg_path, opt)
# Create loaders
loaders = {split: Splitter(dataset, split_path = splits_path, split_name = split) for split in ["train", "val", "test"]}

print("LOAD DATASET COMPLETE")

## Result Tensor (EEG DATA)
## In result_tensor,
## 0. Initialize it with 0 ~ 1 random
## 1. 'data' : EEG tensor
## 2. 'label' : tensor of (num_gt = 5, start, end, cls)
result_tensor = {}
data_info = {'train' : [5000, len(loaders['train'])], 'val' : [1000, len(loaders['val'])], 'test' : [1000, len(loaders['test'])]}
cls_length = [400, 300, 200, 100, 100] # sum = 1100
space = [57, 78, 80, 61, 92, 0] # sum < 400

print("NEW DATASET CONFIGURATION\nDATA INFO : {}\nGROUND TRUTH LENGTH SET : {}\nSPACE BETWEEN THE GROUND TRUTH : {}\n".format(data_info, cls_length, space))

save_path = './cvpr19'

for split in ("train", "val", "test"):
    result_tensor['data'] = torch.Tensor(data_info[split][0], opt.total_length, 128)
    result_tensor['label'] = torch.Tensor(data_info[split][0], opt.num_cls, 3)
    iteration = int(np.ceil(data_info[split][0] * opt.num_cls / data_info[split][1]))
    #print("{} ITERATION : {}".format(split, iteration))
    num_data_idx = np.zeros(iteration * data_info[split][1])
    for i in range(iteration):
        num_data_idx[i*data_info[split][1]:(i+1)*data_info[split][1]] = np.random.permutation(data_info[split][1]) # which data will be imported?


    _idx = 0 # data_idx from num_data_idx
    # Start data making
    for i in range(data_info[split][0]):
        count = space[0]
        for j in range(opt.num_cls):
            idx = int(num_data_idx[_idx])
            # EEG CONTAIN
            _eeg_data = loaders[split][idx][0]
            eeg_data = get_eeg(_eeg_data, cls_length[j])
            result_tensor['data'][i, count:count+cls_length[j], :] = eeg_data
            
            # LABEL CONTAIN
            label_data = loaders[split][idx][1]
            result_tensor['label'][i, j, :] = torch.Tensor([count, count+cls_length[j], loaders[split][idx][1]])

            # NEXT DATA
            _idx += 1
            count += cls_length[j] + space[j+1]

        np.random.shuffle(cls_length)
        np.random.shuffle(space)

    save_filename = 'eeg_{}.pth'.format(split)
    save_file = '{}/{}'.format(save_path, save_filename)

    torch.save(result_tensor, save_file)
    print("EEG DATA IS SAVED IN {}".format(save_file))
