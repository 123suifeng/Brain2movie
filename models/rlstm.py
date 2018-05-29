import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn; cudnn.benchmark = True
from utils.generate_proposal import generate_proposals, nms, extract_proposals, cal_score, bbox_add_weight
import utils.net_utils as net_utils
from utils.eeg_plot import plot_EEG, plot_EEG2

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

    self.rlstm = nn.LSTM(args.num_ch, args.lstm_size, args.lstm_layer, batch_first=True) # Encoding EEG feature to classify and regress (100, 64) -> (batch_size (=1), lstm_size)
    self.cls_net = nn.Linear(args.lstm_size, args.num_cls) # (batch_size (=1), lstm_size) -> (batch_size, num_class)
    self.bbox_net = nn.Linear(args.lstm_size, 2) # (batch_size, 2 ((start, end) or (center, length)?))

  def forward(self, data, labels, proposals, split, acc):
    num_props = data.shape[0] # Number of proposals

    # ELSTM
    if self.args.encoding:
      elstm_init = (torch.zeros(self.args.lstm_layer, num_props, self.args.num_ch), torch.zeros(self.args.lstm_layer, num_props, self.args.num_ch)) # lstm_init : tuple, with len 2
      elstm_init = (elstm_init[0].cuda(), elstm_init[0].cuda())
      elstm_init = (Variable(elstm_init[0], volatile=data.volatile), Variable(elstm_init[1], volatile=data.volatile))
      data = self.elstm(data, elstm_init)[0][:,-1,:] # Forward to RLSTM

    # RLSTM
    rlstm_init = (torch.zeros(self.args.lstm_layer, num_props, self.args.lstm_size), torch.zeros(self.args.lstm_layer, num_props, self.args.lstm_size)) # lstm_init : tuple, with len 2
    rlstm_init = (rlstm_init[0].cuda(), rlstm_init[0].cuda())
    rlstm_init = (Variable(rlstm_init[0], volatile=data.volatile), Variable(rlstm_init[1], volatile=data.volatile))
    base_feat = self.rlstm(data, rlstm_init)[0][:,-1,:] # base_feat = (num_props, lstm_size)

    # Classification
    cls_feat = self.cls_net(base_feat) # cls_feat = (num_props, 40 (=num_cls))

    # Localization
    bbox_feat = self.bbox_net(base_feat)

    if split == 'train':

      cls_loss = F.cross_entropy(cls_feat, labels[:, 0].long()) # F.cross_entropy converts indices automatically
      #scores, classes = cls_feat.data.max(1)
      bbox_inside_weight, bbox_outside_weight = bbox_add_weight(labels[:, 1], self.args)
      proposals = proposals.div(self.args.seq_len)
      proposals = Variable(proposals.cuda())
      bbox_loss = F.smooth_l1_loss(bbox_feat, proposals) # (center, length) ? (start, end) ?

      # Penalized Loss (Division Method)
      _, cls_idx = cls_feat.data.max(1)
      loss_div = 1
      for j in range(self.args.num_prop_after):
        if int(cls_idx[j]) == int(labels[:, 0][j]):
          loss_div += 1
          acc[split] += 1
      cls_loss = cls_loss.div(loss_div)

      bbox_loss *= 10
      print("[proposals]\n{}".format(proposals.data))

    else:
      pick = nms(bbox_feat, cls_feat, self.args)
      cls_feat = cls_feat[pick, :]
      bbox_feat = bbox_feat[pick, :]
      cls_loss = Variable(torch.zeros(1).cuda()) # Garbage Value
      bbox_loss = Variable(torch.zeros(1).cuda()) # Garbage Value

    # Result Print
    _, cls_idx = cls_feat.data.max(1)
    result_print = torch.Tensor(cls_idx.shape[0], 3) # (num_prop, 3 (cls, prop))
    result_print[:, 0] = cls_idx
    result_print[:, 1:] = bbox_feat.data

    print("  Result labels: {}".format(result_print))

    return cls_loss, bbox_loss, acc

  def _init_weights(self):
    def normal_init(m, mean, stddev):
      """
      weight initalizer: truncated normal and random normal.
      """
      # x is a parameter)
      if str(m)[:4] == 'LSTM':
        for weight in m.all_weights[0]:
          weight.data.normal_(mean, stddev)
        m.bias = False
      elif str(m)[:6] == 'Linear':
        m.weight.data.normal_(mean, stddev)
        
      #print(m.__dict__)


    '''
    normal_init(self.rlstm, 0, 0.5)
    normal_init(self.cls_net, 0.2, 0.1)
    normal_init(self.cls_net1, 0.0, 0.1)
    normal_init(self.cls_net2, 0.0, 0.1)
    normal_init(self.cls_net3, 0.0, 0.1)
    normal_init(self.cls_net4, 0.0, 0.1)
    normal_init(self.cls_net5, 0.2, 0.1)
    normal_init(self.cls_net6, 0.2, 0.1)
    normal_init(self.cls_net7, 0.0, 0.1)
    normal_init(self.bbox_regnet, 0.5, 0.01)
    '''

  def create_architecture(self):
    self._init_weights()
