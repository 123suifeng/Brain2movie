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

grads = {}
def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook

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
  parser.add_argument('--lr', dest='learning_rate', help='Start learning rate', default=8e-4, type=float)
  parser.add_argument('--lrd', dest='learning_rate_decay', help='learning rate decay', default=0.5, type=float)
  parser.add_argument('--weightdecay', dest='weight_decay', help='Decay per certain epochs', default=0.99, type=float)
  parser.add_argument('--op', dest='optimizer', help='Optimizer', default='Adam', type=str)

  parser.add_argument('--lstml', dest='lstm_layer', help='Depth of LSTM layer', default=1, type=int)
  parser.add_argument('--lstms', dest='lstm_size', help='Size of LSTM hidden layer', default=256, type=int)

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
  
  ## LOG PATH
  date = datetime.datetime.now()
  day = date.strftime('%m%d_%H%M')
  log_path = "./logs/{}".format(day)
  if not os.path.exists(log_path):
    os.mkdir(log_path)

  ## INITIALIZE TENSORBOARD
  if args.tfboard:
    from utils.logger import Logger
    logger = Logger(log_path)

  ## CONFIG SAVE AS TEXT
  configs = "LSTM Size: {}\nNumber of Proposals: {}\nStart Learning Rate: {}\nLearning Rate Decay: {}\nOptimizer: {}".format(args.lstm_size, args.num_prop_after, args.learning_rate, args.learning_rate_decay, args.optimizer)
  txt_file = "{}/configs.txt".format(log_path)
  f = open(txt_file, 'w')
  f.write(configs)
  f.close()

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

  ## OPTIMIZER
  #params = []
  #for key, value in dict(model.named_parameters()).items():
  #  if value.requires_grad:
  #    params += [{'params':[value],'lr':lr, 'weight_decay': args.weight_decay}]
  lr = args.learning_rate
  optimizer = getattr(torch.optim, args.optimizer)(model.parameters(), lr = args.learning_rate)

  #if args.cuda:
  model.cuda() # CUDA
  ## RESUME

  if args.resume:
    checkpoint = torch.load('./logs/0524_2054/save_model/thecho7_7.pth')
    args.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("Resume the training")

  ## TRAINING
  args.training = True
  for epoch in range(args.start_epoch, args.epochs):
    
    accuracies = {"train": 0, "val": 0, "test": 0}
    counts = {"train": 0, "val": 0, "test": 0}

    # TRAIN MODE
    model.train()
    loss_check = 0
    start = time.time()
    if epoch % (args.checkpoint_interval) == 0:
      adjust_learning_rate(optimizer, args.weight_decay)
      lr *= args.learning_rate_decay

    for i, data in enumerate(train_loader, 0):

      # READ BATCH DATA (IN OUR CASE, BATCH SIZE IS 1)
      inputs, labels = data
      inputs, labels = proposal_gen(inputs, labels, args)
      inputs = inputs.cuda()
      labels = labels.cuda(async = True)
      inputs = Variable(inputs)
      labels = Variable(labels)

      cls_feat = model(inputs)
      cls_loss = F.cross_entropy(cls_feat, labels.long()) # F.cross_entropy converts indices automatically
      loss_div = 1
      for j in range(args.num_prop_after):
        if int(cls_feat.data.max(1)[1][j]) == int(labels[j]):
          loss_div += 1
          accuracies['train'] += 1

      cls_loss = cls_loss.div(loss_div)
      _, cls_idx = cls_feat.data.max(1)
      result_print = []
      for j in range(args.num_prop_after):
        result_print.append(int(cls_idx[j]))
      #print("   Result labels: {}".format(cls_idx.view(-1, args.num_prop_after)))
      print("  Result labels: {}".format(result_print))
      print("CLASS LOSS: {}".format(float(cls_loss.data)))
      loss = cls_loss.mean()
      loss_check += loss.data[0]
      # BACKWARD
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      
      counts['train'] += args.num_prop_after

      # LOSS DISPLAY
      if i % args.checkpoint_interval == 1:
        end = time.time()
        if i > 1:
          loss_check /= args.checkpoint_interval
          print("[epoch: {} - loss_check: {}]".format(epoch, loss_check))
          print("[Iter Accuracy: {}".format(accuracies["train"]/counts["train"]))
        loss_cls = cls_loss.data[0]

        if args.tfboard:
          info = {'loss':loss_check, 'loss_cls':loss_cls}
          for tag, value in info.items():
            logger.scalar_summary(tag, value, i)

        loss_check = 0
        start = time.time()

    print("[Epoch Accuracy: {}".format(accuracies["train"]/counts["train"]))
    ## SAVE MODEL (TBI)
    model_path = "{}/{}".format(log_path, "save_model")
    if not os.path.exists(model_path):
      os.mkdir(model_path)
    save_name = os.path.join('{}'.format(model_path), 'thecho7_{}.pth'.format(epoch))
    save_checkpoint({
      'epoch': epoch,
      'model': model.state_dict(),
      'optimizer': optimizer.state_dict(),
    }, save_name)
    print('Saving Model: {}......'.format(save_name))


if __name__ == '__main__':
  main()
