from __future__ import division
import torch
import numpy as np

def generate_proposals(args, label):
  num_prop = args.num_prop_before # 1000
  center = torch.Tensor([args.seq_len / 3])
  center = int(torch.ceil(center)[0])
  centers = [i*3 for i in range(center)] # If seq_len = 1500, 500 anchor centers
  anchor_scale = args.anchor_scale[0]

  ## Generate Initial Proposals
  _anchors = []
  for i in centers:
    for j in anchor_scale:
      anchors.append([i-j/2, i+j/2])

  ## Discard invalid proposals (out of boundaries)
  _anchors = [i for i in _anchors if i[0] => 0 and i[1] < args.seq_len]
  _anchors = torch.Tensor(_anchors)
  num_anchors = _anchors.shape[0]
  anchors = torch.Tensor(num_anchors, 4)
  anchors[:,:2] = _anchors

  ## Append scores for each proposal
  label_set = label[0,:,:] # (num_gt, 3 (start, end, cls))
  for idx, a in enumerate(_anchors):
    iou = 0
    for l in label_set:
      xa = max(a[0], l[0])
      xb = min(a[1], l[1])
      interArea = xb - xa + 1
      boxAArea = a[1] - a[0] + 1
      boxBArea = l[1] - l[0] + 1

      _iou = interArea / float(boxAArea + boxBArea - interArea)
      if _iou > iou:
        iou = _iou
    anchors[idx, 2] = iou # score
    anchors[idx, 3] = l[2] # class
    
  return anchors

def nms(dets, args)
  t1 = dets[:,0]
  t2 = dets[:,1]
  score = dets[:,2]

  ind = np.argsort(score)
  area = (t2 - t1 + 1).astype(float)
  pick = []
  while len(ind) > 0:
    i = ind[-1]
    pick.append(i)
    ind = ind[:-1]

    tt1 = np.maximum(t1[i], t1[ind])
    tt2 = np.maximum(t2[i], t2[ind])

    wh = np.maximum(0., tt2 - tt1 + 1.0)
    o = wh / (area[i] + area[ind] - wh)

    ind = ind[np.nonzero(o <= args.nms_thresh)[0]]

  return dets[pick, :]

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
  eeg_proposals = torch.Tensor(int(proposals.shape[0]), args.anchor_scale[0][0], int(eeg.shape[1]))
  seq_len = proposals[:, 1] - proposals[:, 0]
  for idx, slen in enumerate(seq_len): # Assume that proposal length is multiple of 100
    scale = slen / 100
    _range = xrange(proposals[idx, 0], proposals[idx, 1], scale)
    if len(_range) != 100:
      print("The length of Proposals should be multiple of 100\n")
      raise ValueError

    eeg_proposals[idx, :, :] = eeg[_range, :]

  return eeg_proposals
