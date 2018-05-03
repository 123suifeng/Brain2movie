import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn; cudnn.benchmark = True
from utils.generate_proposal import generate_proposals, nms, extract_proposals
import utils.net_utils as net_utils

class rlstm(nn.Module):
  def __init__(self, args):
    super(rlstm, self).__init__()
    ## Define Parameters
    self.args = args
    if args.encoding:
      self.elstm = nn.LSTM(args.num_ch, args.encoding_size, args.lstm_layer) # Encoding LSTM (Pre-processing) - (1500, 128) -> (1500, 64)
      self.num_ch = args.encoding_size
    else:
      self.num_ch = args.num_ch

    #self.proposals = generate_proposal(args) # Generate Proposals (How many proposals? - in argument), each proposal has size (100, 64)

    self.rlstm = nn.LSTM(self.num_ch, args.lstm_size, args.lstm_layer, batch_first=True) # Encoding EEG feature to classify and regress (100, 64) -> (batch_size (=1), lstm_size)
    self.cls_net = nn.Linear(args.lstm_size, args.num_cls) # (batch_size (=1), lstm_size) -> (batch_size, num_class)
    self.bbox_regnet = nn.Linear(args.lstm_size, args.num_cls*2) # (batch_size (=1), lstm_size) -> (batch_size, args.num_cls * 2 (=start, end))

  def forward(self, data, label):
    if args.encoding:
      elstm_init = (torch.zeros(self.lstm_layers, self.args.batch_size, self.num_ch), torch.zeros(self.lstm_layers, self.args.batch_size, self.num_ch))
      if data.is_cuda:
        elstm_init = (elstm_init[0].cuda(), elstm_init[1].cuda())
        elstm_init = (Variable(elstm_init[0]), Variable(elstm_init[1])) # set volatile?? (it means do not need backprop. Deprecated as of 0.4.0)

      data = self.elstm(data, elstm_init)[0] # gather all sequences

    proposals = generate_proposals(self.args, label) # proposals = [1000, 4 (start, end, score, class)]

    if args.training:
      proposals = nms(proposals, self.args) # [???, 4]
      proposals = proposals[:self.args.num_prop_after, :] # select 128 proposals (sorted by score)
      eeg_proposals = extract_proposals(self.args, data, proposals[:,:2]) # (128, 2) -> (128, 100, 128) (actual EEG signal)
      rlstm_init = (torch.zeros(self.lstm_layers, int(proposals.shape[0]), self.num_ch), torch.zeros(self.lstm_layers, int(proposals.shape[0]), self.num_ch))
      if data.is_cuda:
        rlstm_init = (rlstm_init[0].cuda(), rlstm_init[1].cuda())
        rlstm_init = (Variable(rlstm_init[0]), Variable(rlstm_init[1])) # set volatile?? (it means do not need backprop. Deprecated as of 0.4.0)
      base_feat = self.rlstm(eeg_proposals, rlstm_init)[0][:,-1,:] # base_feat = (128, lstm_size)
      cls_feat = self.cls_net(base_feat) # cls_feat = (128, 41 (num_cls))
      bbox_feat = self.bbox_regnet(base_feat) # bbox_feat = (128, 82 (num_cls * 2))

      cls_loss = F.cross_entropy(cls_feat, classes) # F.cross_entropy converts indices automatically
      bbox_loss = net_utils._smooth_l1_loss(bbox_feat, anchor, label[:,:,:2]) # label[:,:,:2] = start, end

    else:
      base_feat = self.rlstm(proposals)[0][:,-1,:]
      cls_feat = self.cls_net(base_feat)
      bbox_feat = self.bbox_regnet(base_feat)
      cls_loss = 0
      bbox_loss = 0

  return cls_feat, bbox_feat, cls_loss, bbox_loss
