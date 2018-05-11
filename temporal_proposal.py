from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from models.rlstm import rlstm
import utils.loaddata as loaddata
import utils.net_utils as net_utils
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


def arguments():
  parser = argparse.ArgumentParser(description='Brain2Movie')
  parser.add_argument('--dataset', dest='dataset', default='cvpr19', type=str)
  parser.add_argument('--nc', dest='num_cls', default=41, type=int) # 40 Categories + background
  parser.add_argument('--epoch', dest='epochs', default=1000, type=int)
  parser.add_argument('--checkpoint_interval', dest='checkpoint_interval', default=50, type=int)
  parser.add_argument('--save_dir', dest='save_dir', help='save directory to save model', default='./save', type=str)
  parser.add_argument('--nw', dest='num_workers', help='Num or workers to load data', default=8, type=int)
  #parser.add_argument('--cuda', dest='cuda', help='Use cuda?', default=True, action='store_true')
  parser.add_argument('--bs', dest='batch_size', default=1, type=int)
  parser.add_argument('--tfboard', dest='tfboard', help='Use Tensorboard?', default=True, type=bool)
  parser.add_argument('--lr', dest='learning_rate', help='Start learning rate', default=0.0001, type=float)
  parser.add_argument('--lrd', dest='learning_rate_decay', help='learning rate decay', default=0.1, type=float)
  parser.add_argument('--weightdecay', dest='weight_decay', help='Decay per certain epochs', default=0.1, type=float)
  parser.add_argument('--op', dest='optimizer', help='Optimizer', default='Adam', type=str)

  parser.add_argument('--lstml', dest='lstm_layer', help='Depth of LSTM layer', default=1, type=int)
  parser.add_argument('--lstms', dest='lstm_size', help='Size of LSTM hidden layer', default=10, type=int)

  parser.add_argument('--ec', dest='encoding', help='Encoding EEG?', default=True, type=bool)
  parser.add_argument('--es', dest='encoding_scale', help='A scale of encoding EEG', default=2, type=int)
  parser.add_argument('--esi', dest='encoding_size', help='A scale of encoding EEG', default=1, type=int)

  parser.add_argument('--nms', dest='nms_thresh', help='Threshold of nms', default=0.5, type=float)
  parser.add_argument('--npb', dest='num_prop_before', help='The number of proposals before nms', default=1000, type=int)
  parser.add_argument('--npa', dest='num_prop_after', help='The number of proposals after nms', default=128, type=int)

  args = parser.parse_args()
  return args

def main():
  ## CALL ARGUMENTS
  args = arguments()
  print("Called Arguments")
  
  ## INITIALIZE TENSORBOARD
  if args.tfboard:
    from utils.logger import Logger
    logger = Logger('./logs')

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

  ## ENCODING OR NOT
  if args.encoding:
    args.encoding_size = int((args.num_ch) / (args.encoding_scale))

  ## CALL MODEL
  model = rlstm(args)
  model.create_architecture()
  #if args.cuda:
  model.cuda() # CUDA

  ## OPTIMIZER
  lr = args.learning_rate
  params = []
  for key, value in dict(model.named_parameters()).items():
    if value.requires_grad:
      params += [{'params':[value],'lr':lr, 'weight_decay': args.weight_decay}]

  if args.optimizer == "Adam":
    optimizer = torch.optim.Adam(params)

  ## VARIABLE SETTING (IS IT EFFICIENT?)
  eeg_data = torch.FloatTensor(1)
  eeg_label = torch.LongTensor(1)
  classes = torch.LongTensor(args.num_prop_after)
  proposals = torch.FloatTensor(args.num_prop_after, 2)

  eeg_data = eeg_data.cuda()
  eeg_data = Variable(eeg_data)
  eeg_label = eeg_label.cuda()
  eeg_label = Variable(eeg_label)
  classes = classes.cuda()
  classes = Variable(classes)
  proposals = proposals.cuda()
  proposals = Variable(proposals)

  ## TRAINING
  args.training = True
  for epoch in range(1, args.epochs):
    # TRAIN MODE
    model.train()
    loss_check = 0
    start = time.time()
    if epoch % (args.checkpoint_interval) == 0:
      net_utils.adjust_learning_rate(optimizer, args.weight_decay)
      lr *= args.learning_rate_decay

    for i, data in enumerate(train_loader, 0):
      # READ BATCH DATA (IN OUR CASE, BATCH SIZE IS 1)
      inputs, labels = data
      eeg_data.data.resize_(inputs.size()).copy_(inputs)
      eeg_label.data.resize_(labels.size()).copy_(labels)

      cls_feat, rpn_feat, cls_loss, rpn_loss = model(eeg_data, eeg_label, proposals, classes)
   
      loss = cls_loss.mean() + rpn_loss.mean()
      loss_check += loss.data[0]
      # BACKWARD
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      
      # LOSS DISPLAY
      if i % args.checkpoint_interval == 1:
        end = time.time()
        if i > 1:
          loss_check /= args.checkpoint_interval
          print("[loss_check: {}]".format(loss_check))
        loss_cls = cls_loss.data[0]
        loss_rpn = rpn_loss.data[0]

        if args.tfboard:
          info = {'loss':loss_check, 'loss_cls':loss_cls, 'loss_rpn':loss_rpn}
          for tag, value in info.items():
            logger.scalar_summary(tag, value, i)

        loss_check = 0
        start = time.time()

    ## SAVE MODEL (TBI)

if __name__ == '__main__':
  main()
