import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn; cudnn.benchmark = True
from utils.generate_proposal import generate_proposals, nms, extract_proposals, cal_score, bbox_add_weight
import utils.net_utils as net_utils

class rlstm(nn.Module):
  def __init__(self, args):
    super(rlstm, self).__init__()
    ## Define Parameters
    self.args = args
    if args.encoding:
      self.elstm = nn.LSTM(args.num_ch, args.encoding_size, args.lstm_layer) # Encoding LSTM (Pre-processing) - (1500, 128) -> (1500, 64)
      self.args.num_ch = args.encoding_size
    else:
      self.args.num_ch = args.num_ch

    self.rlstm = nn.LSTM(self.args.num_ch, args.lstm_size, args.lstm_layer, batch_first=True) # Encoding EEG feature to classify and regress (100, 64) -> (batch_size (=1), lstm_size)
    self.cls_net = nn.Linear(args.lstm_size, args.num_cls) # (batch_size (=1), lstm_size) -> (batch_size, num_class)
    self.bbox_regnet = nn.Linear(args.lstm_size, 2) # (batch_size (=1), lstm_size) -> (batch_size, 2 (=start, end))

  def forward(self, data, label, proposals, classes):
    if self.args.encoding:
      data = self.elstm(data)[0] # gather all sequences

    _proposals = generate_proposals(self.args, label) # proposals = [1000, 4 (start, end, score, class)]

    if self.args.training:
      # Calculate score for each proposal
      _proposals, _scores, _classes = cal_score(_proposals, label, self.args)
      # NMS
      pick_idx = nms(_proposals, _scores, self.args) # pick_idx includes both pos and neg
      _proposals = _proposals[pick_idx, :]
      _scores = _scores[pick_idx, :]
      _classes = _classes[pick_idx, :]

      _proposals = _proposals[:self.args.num_prop_after, :] # select 128 proposals (sorted by score)
      _scores = _scores[:self.args.num_prop_after, :]
      _classes = _classes[:self.args.num_prop_after, :]

      bbox_inside_weight, bbox_outside_weight = bbox_add_weight(_scores, self.args)

      proposals.data.copy_(_proposals)
      classes.data.copy_(_classes[:,0]) # n x 1 -> n (dimension...)

      eeg_proposals = extract_proposals(self.args, data, proposals[:,:2]) # (128, 2) -> (128, 100, 128) (actual EEG signal)
      eeg_proposals = Variable(eeg_proposals.cuda())

      base_feat = self.rlstm(eeg_proposals)[0][:,-1,:] # base_feat = (128, lstm_size)
      cls_feat = self.cls_net(base_feat) # cls_feat = (128, 41 (num_cls))
      bbox_feat = self.bbox_regnet(base_feat) # bbox_feat = (128, 82 (num_cls * 2))

      cls_loss = F.cross_entropy(cls_feat, classes.long()) # F.cross_entropy converts indices automatically
      bbox_loss = net_utils._smooth_l1_loss(bbox_feat, proposals, bbox_inside_weight, bbox_outside_weight) # start, end

    else:
      base_feat = self.rlstm(proposals)[0][:,-1,:]
      cls_feat = self.cls_net(base_feat)
      bbox_feat = self.bbox_regnet(base_feat)
      cls_loss = 0
      bbox_loss = 0

    return cls_feat, bbox_feat, cls_loss, bbox_loss

  def _init_weights(self):
    def normal_init(m, mean, stddev, truncated=False):
      """
      weight initalizer: truncated normal and random normal.
      """
      # x is a parameter
      if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
      else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()

      normal_init(self.rlstm, 0, 0.01, cfg.TRAIN.TRUNCATED)
      normal_init(self.cls_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
      normal_init(self.bbox_regnet, 0, 0.01, cfg.TRAIN.TRUNCATED)

  def create_architecture(self):
    self._init_weights()
