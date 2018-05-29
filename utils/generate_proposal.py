from __future__ import division
import torch
import numpy as np
import random
from torch.autograd import Variable
from operator import itemgetter

def generate_proposals(args):  
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
  anchors = torch.Tensor(_anchors)
  
  
  return anchors

def cal_score(prop, label, args):
  ## Append scores for each proposal
  label_set = label[0,:,:] # (num_gt, 3 (start, end, cls))

  num_anchors = prop.shape[0]
  _class = torch.Tensor(num_anchors)
  _score = torch.Tensor(num_anchors)

  eff_prop = []
  for idx, a in enumerate(prop):
    iou = 0
    cls = 0
    for l in label_set:
      xa = max(a[0], l[0])
      xb = min(a[1], l[1])
      interArea = xb - xa + 1
      boxAArea = a[1] - a[0] + 1
      boxBArea = l[1] - l[0] + 1
      _iou = interArea / float(boxAArea + boxBArea - interArea)

      if _iou > iou:
        iou = _iou
        cls = l[2]
    _score[idx] = iou # score
    ## Collect POS, NEG proposals according to the score. Else, discard.
    if float(_score[idx]) >= args.score_thresh:
      _class[idx] = cls # class
      eff_prop.append(idx)
    ## DELETE BACKGROUND !!!!! (I guess the network cannot train the background because it is randomly generated (DO NOT HAVE ANY PATTERN))
    #elif float(_score[idx]) <= 0.3:
    #  _class[idx] = 0 # bg
    #  eff_prop.append(idx)

  return prop[eff_prop,:], _score[eff_prop], _class[eff_prop]

def select_train_props(score, cls, args):
  total_num = score.shape[0]
  cls_unique = np.unique(cls)
  ind = np.argsort(score) # (TBM (To Be Modified))
  
  ## Keep several positive samples (Absolute positive including background)
  keep = []

  if args.num_prop_after >= len(cls_unique):
    for i in range(len(cls_unique)):
      fg_pick = [ind[idx] for idx in range(len(ind)) if (cls[ind[idx]] == cls_unique[i] and score[ind[idx]] >= args.score_thresh)] # Class-wise collection with condition over 0.7 (score threshold)
      if len(fg_pick) >= 1:
        #random_idx = np.random.permutation(int(np.ceil(len(fg_pick) / 2)) + 1)[:2] # How many props per class? (To avoid biased training)
        #fg_pick = itemgetter(*random_idx)(fg_pick)
        #keep = np.concatenate((keep, fg_pick))
        random_idx = np.random.permutation(int(np.ceil(len(fg_pick) / 2)) + 1)[0]
        fg_pick = fg_pick[random_idx]
        keep.append(fg_pick)
      else:
        print("Class {} is discarded (All negatives)".format(cls_unique[i]))

    if (args.num_prop_after - len(cls_unique)) > 0:
      remain_num = args.num_prop_after - len(cls_unique)
      rand = np.random.permutation(total_num)[:remain_num]
      #if isinstance(rand, int):
      if remain_num == 1:
        keep.append(int(rand[0]))
      else:
        for j in range(remain_num):
          keep.append(int(rand[j]))

  return keep

#def nms(dets, score, args): # Add class information to avoid the bias
def nms(bbox_feat, cls_feat, args): # Add class information to avoid the bias
  bbox_feat, _ = torch.sort(bbox_feat, 1)
  # NMS
  # dets: bbox_feat, score: cls_feat
  t1 = torch.Tensor(1)
  t2 = torch.Tensor(1)
  score = torch.Tensor(1)
  t1.resize_(bbox_feat.shape[0]).copy_(bbox_feat.data[:,0])
  t2.resize_(bbox_feat.shape[0]).copy_(bbox_feat.data[:,1])
  score.resize_(bbox_feat.shape[0]).copy_(cls_feat.data.max(1)[0])
  #t1 = bbox_feat.data[:,0]
  #t2 = bbox_feat.data[:,1]
  #score = cls_feat.data.max(1)[0]

  #t1 = dets[:,0]
  #t2 = dets[:,1]
  ind = np.argsort(score)

  area = torch.abs(t2 - t1)
  pick = []
  '''
  while len(ind) > 1:

    i = ind[-1]
    pick.append(i)
    ind = ind[:-1]
    tt1 = np.maximum(t1[i], t1[ind])
    tt2 = np.minimum(t2[i], t2[ind])
    wh = np.maximum(0., tt2 - tt1)
    #o = wh / (area[i] + area[ind] - wh)
    o = wh / area[ind]
    ind = ind[np.transpose(np.nonzero(o >= args.nms_thresh))[0]]
  '''

  while len(ind) > 1:

    last = len(ind) - 1
    i = ind[last]
    pick.append(i)
    tt1 = np.maximum(t1[i], t1[ind[:last]])
    tt2 = np.minimum(t2[i], t2[ind[:last]])
    l = np.maximum(0., tt2 - tt1)
    o = l / area[ind[:last]]
    if len(np.nonzero(o < args.nms_thresh)) != 0:
      ind = np.delete(ind, np.concatenate(([last], np.where(o > args.nms_thresh)[0])))
    else:
      break

  return pick

def extract_proposals(args, eeg, proposals):
  # Downsampling into size of 100
  eeg_proposals = torch.Tensor(int(proposals.shape[0]), args.anchor_scale[0][3], int(eeg.shape[2])).cuda()
  seq_len = proposals[:, 1] - proposals[:, 0]
  for idx, slen in enumerate(seq_len): # Assume that proposal length is multiple of 100
    scale = int(slen / 100)
    _range = range(int(proposals[idx, 0]), int(proposals[idx, 1]), scale)
    if len(_range) != 100:
      print("The length of Proposals should be multiple of 100\n")
      raise ValueError
    eeg_proposals[idx, :, :] = eeg[:,_range, :]

  return eeg_proposals

def proposal_gen(inputs, labels, args):
  # Generate proposals according to the length of data
  proposals = generate_proposals(args) # Just coordinate (start, end)

  if args.training:
    # Calculate score for each proposal
    proposals, scores, classes = cal_score(proposals, labels, args)

    # Select proposals to train
    keep = select_train_props(scores, classes, args) # Only Positive proposals (TBM)
    pick_idx = keep[:args.num_prop_after]
    proposals = proposals[pick_idx, :]
    classes = classes[pick_idx]
    classes = torch.squeeze(classes)
    scores = scores[pick_idx]
    labels = torch.stack((classes, scores)).t() # classes: labels[:,0], score: labels[:,1]

    # Print Target Labels
    class_print = []
    for i in range(args.num_prop_after):
      class_print.append(int(classes[i]))
    print("Selected labels: {}".format(class_print))
    #print("          Score: {}".format(_scores))

  else:
    # Garbages, GT should be used only for evaluation in test session
    labels = torch.zeros(0)

  eeg_proposals = extract_proposals(args, inputs, proposals[:,:2]) # (128, 2) -> (128, 100, 128) (actual EEG signal)

  return eeg_proposals, proposals, labels

def bbox_add_weight(score, args):
  bbox_inside_weight = torch.zeros(score.shape[0], 2)
  bbox_outside_weight = torch.zeros(score.shape[0], 2)

  idx = [i for i, j in enumerate(score) if float(j) >= args.score_thresh]

  bbox_inside_weight[idx, :] = 1
  bbox_outside_weight[idx, :] = 1

  bbox_inside_weight = bbox_inside_weight.cuda()
  bbox_outside_weight = bbox_outside_weight.cuda()
  bbox_inside_weight = Variable(bbox_inside_weight, requires_grad = True)
  bbox_outside_weight = Variable(bbox_outside_weight, requires_grad = True)

  return bbox_inside_weight, bbox_outside_weight
