from __future__ import division
import torch
import numpy as np
import random
from torch.autograd import Variable

def generate_proposals(args, label):  
  center = torch.Tensor([args.seq_len / 3])
  center = int(torch.ceil(center)[0])
  centers = [i*3 for i in range(center)] # If seq_len = 1500, 500 anchor centers
  anchor_scale = args.anchor_scale[0]

  ## Generate Initial Proposals
  _anchors = []
  for i in centers:
    for j in anchor_scale:
      _anchors.append([i-j/2, i+j/2])

  ## Discard invalid proposals (out of boundaries)
  _anchors = [i for i in _anchors if i[0] >= 0 and i[1] < args.seq_len]
  _anchors = torch.Tensor(_anchors)
  num_anchors = _anchors.shape[0]
  anchors = torch.Tensor(num_anchors, 2)
  anchors[:,:2] = _anchors

  return anchors

def cal_score(prop, label, args):
  ## Append scores for each proposal
  label_set = label[0,:,:] # (num_gt, 3 (start, end, cls))
  num_prop = args.num_prop_before # 1000

  num_anchors = prop.shape[0]
  _class = torch.Tensor(num_anchors, 1)
  _score = torch.Tensor(num_anchors, 1)

  for idx, a in enumerate(prop):
    iou = 0
    cls = 0
    for l in label_set:
      xa = max(a[0], l.data[0])
      xb = min(a[1], l.data[1])
      interArea = xb - xa + 1
      boxAArea = a[1] - a[0] + 1
      boxBArea = l.data[1] - l.data[0] + 1
      _iou = interArea / float(boxAArea + boxBArea - interArea)

      if _iou > iou:
        iou = _iou
        cls = l.data[2]
    _score[idx] = iou # score
    _class[idx] = cls # class

  an_idx = np.random.permutation(num_anchors)
  tt = []
  for i in an_idx:
    tt.append(i)

  prop = prop[tt[:num_prop], :]
  _score = _score[tt[:num_prop], :]
  _class = _class[tt[:num_prop], :]
    
  return prop, _score, _class

def nms(dets, score, args):
  t1 = dets[:,0]
  t2 = dets[:,1]
  total_num = score.shape[0]

  score_p = []
  for i in score:
    score_p.append(float(i))

  ind = np.argsort(score_p)
  
  area = (t2 - t1 + 1)
  pick = []
  while len(ind) > 1:
    i = ind[-1]
    pick.append(i)
    ind = ind[:-1]

    tt1 = np.maximum(t1[i], t1[ind])
    tt2 = np.maximum(t2[i], t2[ind])

    wh = np.maximum(0., tt2 - tt1 + 1.0)
    o = wh / (area[i] + area[ind] - wh)

    ind = ind[np.transpose(np.nonzero(o >= args.nms_thresh))[0]]

  ## Negative sample
  pick_n = [idx for idx, i in enumerate(score_p) if i < 0.3]
  pick_n = pick_n[:len(pick)]
  for i in pick_n:
    pick.append(i)

  random.shuffle(pick) # Optional
  return pick

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

def extract_proposals(args, eeg, proposals):
  eeg_proposals = torch.Tensor(int(proposals.shape[0]), args.anchor_scale[0][3], int(eeg.shape[2])).cuda()
  seq_len = proposals[:, 1] - proposals[:, 0]
  for idx, slen in enumerate(seq_len): # Assume that proposal length is multiple of 100
    scale = int(slen / 100)
    _range = range(int(proposals[idx, 0]), int(proposals[idx, 1]), scale)
    if len(_range) != 100:
      print("The length of Proposals should be multiple of 100\n")
      raise ValueError
    eeg_proposals[idx, :, :] = eeg.data[:,_range, :]

  return eeg_proposals

def bbox_add_weight(score, args):
  bbox_inside_weight = torch.zeros(score.shape[0], 2)
  bbox_outside_weight = torch.zeros(score.shape[0], 2)

  idx = [i for i, j in enumerate(score) if float(j) >= args.nms_thresh]

  bbox_inside_weight[idx, :] = 1
  bbox_outside_weight[idx, :] = 1

  bbox_inside_weight = bbox_inside_weight.cuda()
  bbox_outside_weight = bbox_outside_weight.cuda()
  bbox_inside_weight = Variable(bbox_inside_weight)
  bbox_outside_weight = Variable(bbox_outside_weight)

  return bbox_inside_weight, bbox_outside_weight
