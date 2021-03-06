from __future__ import division # IN PYTHON3, IT IS NOT NEEDED
# Define options
import argparse
parser = argparse.ArgumentParser(description="Template")
# Dataset options
parser.add_argument('-ed', '--eeg-dataset', default="../eeg_dataset/proposals/proposals.pth", help="EEG dataset path")
parser.add_argument('-sp', '--splits-path', default="../eeg_dataset/proposals/split.pth", help="splits path")
parser.add_argument('-sn', '--split-num', default=0, type=int, help="split number")
# Model options
parser.add_argument('-ll', '--lstm-layers', default=1, type=int, help="LSTM layers")
parser.add_argument('-ls', '--lstm-size', default=256, type=int, help="LSTM hidden size")
parser.add_argument('-os', '--output-size', default=40, type=int, help="output layer size")
# Training options
parser.add_argument("-b", "--batch_size", default=16, type=int, help="batch size")
parser.add_argument('-o', '--optim', default="Adam", help="optimizer")
parser.add_argument('-lr', '--learning-rate', default=1e-3, type=float, help="learning rate")
parser.add_argument('-lrdb', '--learning-rate-decay-by', default=0.5, type=float, help="learning rate decay factor")
parser.add_argument('-lrde', '--learning-rate-decay-every', default=10, type=int, help="learning rate decay period")
parser.add_argument('-dw', '--data-workers', default=4, type=int, help="data loading workers")
parser.add_argument('-e', '--epochs', default=100, type=int, help="training epochs")
# Backend options
parser.add_argument('--no-cuda', default=False, help="disable CUDA", action="store_true")

# Parse arguments
opt = parser.parse_args()

# Imports
import sys
import os
import random
import math
import time
import torch; torch.utils.backcompat.broadcast_warning.enabled = True
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.backends.cudnn as cudnn; cudnn.benchmark = True
import numpy as np

# Dataset class
class EEGDataset:
    
    # Constructor
    def __init__(self, eeg_signals_path):
        # Load EEG signals
        loaded = torch.load(eeg_signals_path) #len : 5 ( subject 5 is duplicated)

        self.data = loaded["dataset"] # 11965
        # Compute size
        self.size = len(self.data)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Process EEG
        eeg = self.data[i]["eeg"].float()
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
        self.split_idx = [i for i in self.split_idx if self.dataset.data[i]["eeg"].size(0) == 100]
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


# Load dataset
dataset = EEGDataset(opt.eeg_dataset)
# Create loaders
loaders = {split: DataLoader(Splitter(dataset, split_path = opt.splits_path, split_num = opt.split_num, split_name = split), batch_size = opt.batch_size, drop_last = True, shuffle = True) for split in ["train", "val", "test"]}

#print(loaders['test'])

# Define model
class Model(nn.Module):

    def __init__(self, input_size, lstm_size, lstm_layers, output_size):
        # Call parent
        super(Model, self).__init__()
        # Define parameters
        self.input_size = input_size
        self.lstm_size = lstm_size
        self.lstm_layers = lstm_layers
        self.output_size = output_size
        # Define internal modules
        self.lstm = nn.LSTM(input_size, lstm_size, num_layers=lstm_layers, batch_first=True) # LSTM_SIZE : Hidden size
        self.output = nn.Linear(lstm_size, output_size)

    def forward(self, x):
        # Prepare LSTM initiale state
        batch_size = x.size(0)
        lstm_init = (torch.zeros(self.lstm_layers, batch_size, self.lstm_size), torch.zeros(self.lstm_layers, batch_size, self.lstm_size)) # lstm_init : tuple, with len 2
        if x.is_cuda: lstm_init = (lstm_init[0].cuda(), lstm_init[0].cuda())
        lstm_init = (Variable(lstm_init[0], volatile=x.volatile), Variable(lstm_init[1], volatile=x.volatile))
        # Forward LSTM and get final state
        x = self.lstm(x, lstm_init)[0][:,-1,:] # LAST HIDDEN OUTPUT
        #print(x1.data.view(16, -1, x1.shape[1])[:3, :, :9])
        x = self.output(x)
        return x

model = Model(128, opt.lstm_size, opt.lstm_layers, opt.output_size)
optimizer = getattr(torch.optim, opt.optim)(model.parameters(), lr = opt.learning_rate)
    
# Setup CUDA
if not opt.no_cuda:
    model.cuda()
    print("Copied to CUDA")

time_count = 0
# Start training
for epoch in range(1, opt.epochs+1):
    start = time.time()
    # Initialize loss/accuracy variables
    losses = {"train": 0, "val": 0, "test": 0}
    accuracies = {"train": 0, "val": 0, "test": 0}
    counts = {"train": 0, "val": 0, "test": 0}
    # Adjust learning rate for SGD
    if opt.optim == "SGD":
        lr = opt.learning_rate * (opt.learning_rate_decay_by ** (epoch // opt.learning_rate_decay_every))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    # Process each split
    for split in ("train", "val", "test"):
        # Set network mode
        if split == "train":
            model.train()
        else:
            model.eval()
        # Process all split batches
        for i, (input, target) in enumerate(loaders[split]): # input : EEG tensors shapes (batch_size, 430, 128), target : annotation
            target = target.long()
            # Check CUDA             
            if not opt.no_cuda:
                input = input.cuda()
                target = target.cuda(async = True)
            # Wrap for autograd
            input = Variable(input, volatile = (split != "train"))
            target = Variable(target, volatile = (split != "train"))

            # Forward
            output = model(input)
            #print(output)
            loss = F.cross_entropy(output, target)
            print("[LOSS: {}".format(loss))
            losses[split] += loss.data[0]
            # Compute accuracy
            _,pred = output.data.max(1) # _ : the value of the largest element in 128 X 1 result, pred : the index of that
            print("[Original label: {}".format(np.unique(target.data)))
            print("[Selected label: {}".format(np.unique(pred)))
            correct = pred.eq(target.data).sum() # Count the number of correct prediction
            accuracy = correct / input.data.size(0)
            accuracies[split] += accuracy
            counts[split] += 1
            # Backward and optimize
            if split == "train":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    # Print info at the end of the epoch
    print("Epoch {0}: TrL={1:.4f}, TrA={2:.4f}, VL={3:.4f}, VA={4:.4f}, TeL={5:.4f}, TeA={6:.4f}".format(epoch,
                                                                                                         losses["train"]/counts["train"],
                                                                                                         accuracies["train"]/counts["train"],
                                                                                                         losses["val"]/counts["val"],
                                                                                                         accuracies["val"]/counts["val"],
                                                                                                         losses["test"]/counts["test"],
                                                                                                         accuracies["test"]/counts["test"]))

    end = time.time()
    time_count += (end-start)
    print("Time cost : {:.4f}".format(end-start))

print("Total time cost : {:.4f}".format(time_count))
