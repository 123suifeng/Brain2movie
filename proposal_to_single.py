from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from models.rlstm import rlstm
from utils.generate_proposal import proposal_gen
import utils.loaddata as loaddata
from utils.net_utils import adjust_learning_rate, save_checkpoint
import argparse
import sys
import os
import random
import math
import torch; torch.utils.backcompat.broadcast_warning.enabled = True
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.backends.cudnn as cudnn; cudnn.benchmark = True
import numpy as np
import datetime

def arguments():
  parser = argparse.ArgumentParser(description='Brain2Movie')
  parser.add_argument('--dataset', dest='dataset', default='cvpr19', type=str)
  parser.add_argument('--nc', dest='num_cls', default=40, type=int) # 40 Categories + background
  parser.add_argument('--sep', dest='start_epoch', default=1, type=int)
  parser.add_argument('--epoch', dest='epochs', default=1000, type=int)
  parser.add_argument('--checkpoint_interval', dest='checkpoint_interval', default=5, type=int)
  parser.add_argument('--save_dir', dest='save_dir', help='save directory to save model', default='./save', type=str)
  parser.add_argument('--nw', dest='num_workers', help='Num or workers to load data', default=8, type=int)
  #parser.add_argument('--cuda', dest='cuda', help='Use cuda?', default=True, action='store_true')
  parser.add_argument('--bs', dest='batch_size', default=1, type=int)
  parser.add_argument('--tfboard', dest='tfboard', help='Use Tensorboard?', default=True, type=bool)
  parser.add_argument('--lr', dest='learning_rate', help='Start learning rate', default=1e-4, type=float)
  parser.add_argument('--lrd', dest='learning_rate_decay', help='learning rate decay', default=0.8, type=float)
  parser.add_argument('--weightdecay', dest='weight_decay', help='Decay per certain epochs', default=0.1, type=float)
  parser.add_argument('--op', dest='optimizer', help='Optimizer', default='Adam', type=str)

  parser.add_argument('--lstml', dest='lstm_layer', help='Depth of LSTM layer', default=1, type=int)
  parser.add_argument('--lstms', dest='lstm_size', help='Size of LSTM hidden layer', default=128, type=int)

  parser.add_argument('--ec', dest='encoding', help='Encoding EEG?', default=True, type=bool)
  parser.add_argument('--es', dest='encoding_scale', help='A scale of encoding EEG', default=2, type=int)
  parser.add_argument('--esi', dest='encoding_size', help='A scale of encoding EEG', default=1, type=int)

  parser.add_argument('--sth', dest='score_thresh', help='Threshold of posneg', default=0.7, type=float)
  parser.add_argument('--nms', dest='nms_thresh', help='Threshold of nms', default=0.5, type=float)
  parser.add_argument('--npb', dest='num_prop_before', help='The number of proposals before nms', default=500, type=int)
  parser.add_argument('--npa', dest='num_prop_after', help='The number of proposals after nms', default=4, type=int)

  parser.add_argument('--re', dest='resume', help='Do you want resume the training?', default=False, type=bool)

  args = parser.parse_args()
  return args

def main():
  ## CALL ARGUMENTS
  args = arguments()
  print("Called Arguments")
  ## DATASET CONFIGURATION
  if args.dataset == 'cvpr19':
    args.anchor_scale = [[400, 300, 200, 100]]

  ## CUDA CHECK
  if not torch.cuda.is_available():
    print("WARNING: Why don't you use CUDA?")

  ## DATALOADER (ITERATOR)
  data_type = 'train'
  EEGDetectionData = loaddata.EEGDetectionDataset(args, data_type) # Shape of data : (seq_len, num_ch) or (1, seq_len, num_ch)??
  train_loader = DataLoader(dataset=EEGDetectionData, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers) # Already shuffled
  seq_len, num_ch = EEGDetectionData[0][0].shape[0], EEGDetectionData[0][0].shape[1]
  args.seq_len, args.num_ch = seq_len, num_ch

  ## VARIABLE SETTING (IS IT EFFICIENT?)

  ## TRAINING
  args.training = True
  eeg_data = {}
  eeg_data['dataset'] = {}
  split_data = {}
  split_data['splits'] = []
  split_data['splits'].append({})
  split_data['splits'][0]['train'] = []
  split_data['splits'][0]['test'] = []
  split_data['splits'][0]['val'] = []
  idx = 0
  for _, data in enumerate(train_loader, 0):
    # READ BATCH DATA (IN OUR CASE, BATCH SIZE IS 1)
    inputs, labels = data
    inputs, labels = proposal_gen(inputs, labels, args)
    for i in range(4):
      print(idx)
      eeg_data['dataset'][idx] = {}
      eeg_data['dataset'][idx]['eeg'] = inputs[i, :, :]
      eeg_data['dataset'][idx]['label'] = labels[i]
  
      if idx < 15000:
        split_data['splits'][0]['train'].append(idx)
      elif 15000 <= idx < 18000:
        split_data['splits'][0]['test'].append(idx)
      elif idx >= 18000:
        split_data['splits'][0]['val'].append(idx)

      idx += 1

  savedata_name = "../eeg_dataset/proposals/proposals.pth"
  savesplit_name = "../eeg_dataset/proposals/split.pth"
  torch.save(eeg_data, savedata_name)
  torch.save(split_data, savesplit_name)


if __name__ == '__main__':
  main()
