from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from matplotlib.ticker import MultipleLocator
import torch

def plot_EEG(data, iid):
  batch_size, length, ch = data.shape

  # Display only 8 EEGs
  num = 8
  num_ch = 10
  fig = plt.figure("EEG DATA")
  t = 100.0 * np.arange(int(length)) / int(length)

  for idx in range(1, num+1):
    # NOT DISPLAY THE WHOLE CH, BUT ONLY 10 CHS
    eeg = data[idx-1, :, :num_ch]
    name = "EEG%d".format(idx)
    name = fig.add_subplot(4, 2, idx)
    name.set_xlim(0,100)
    #name.set_xticks(np.arange(100))
    name.set_xlabel('Time')
    dmin = eeg.min()
    dmax = eeg.max()
    dr = (dmax - dmin) * 0.7
    y0 = dmin
    y1 = ((num / 2) - 1) * dr + dmax
    name.set_ylim(y0, y1)

    segs = []
    ticklocs = []
    for i in range(num_ch):
      segs.append(np.hstack((t[:, np.newaxis], eeg[:, i, np.newaxis])))
      ticklocs.append(i*dr)

    offsets = np.zeros((num_ch, 2), dtype=float)
    offsets[:,1] = ticklocs

    lines = LineCollection(segs, offsets=offsets, transOffset=None)
    name.add_collection(lines)

  plt.tight_layout()
  #plt.show()
  filename = "./eeg_figs/sample_{:03d}.png".format(iid)
  fig.savefig(filename)
  plt.clf()

def plot_EEG2(data, iid):
  batch_size, length, ch = data.shape

  # Display only 8 EEGs
  num = 1
  num_ch = 10
  fig = plt.figure("EEG DATA")
  t = 1500.0 * np.arange(int(length)) / int(length)

  for idx in range(1, num+1):
    # NOT DISPLAY THE WHOLE CH, BUT ONLY 10 CHS
    eeg = data[idx-1, :, :num_ch]
    name = "EEG%d".format(idx)
    name = fig.add_subplot(4, 2, idx)
    name.set_xlim(0,100)
    #name.set_xticks(np.arange(100))
    name.set_xlabel('Time')
    dmin = eeg.min()
    dmax = eeg.max()
    dr = (dmax - dmin) * 0.7
    y0 = dmin
    y1 = ((num / 2) - 1) * dr + dmax
    name.set_ylim(y0, y1)

    segs = []
    ticklocs = []
    for i in range(num_ch):
      segs.append(np.hstack((t[:, np.newaxis], eeg[:, i, np.newaxis])))
      ticklocs.append(i*dr)

    offsets = np.zeros((num_ch, 2), dtype=float)
    offsets[:,1] = ticklocs

    lines = LineCollection(segs, offsets=offsets, transOffset=None)
    name.add_collection(lines)

  plt.tight_layout()
  #plt.show()
  filename = "./eeg_figs/sample_{:03d}.png".format(iid)
  fig.savefig(filename)
  plt.clf()

def plot_EEG3(data, iid):
  length, ch = data.shape

  # Display only 8 EEGs
  num = 1
  num_ch = 2
  fig = plt.figure("EEG DATA")
  t = 1500.0 * np.arange(int(length)) / int(length)

  for idx in range(1, num+1):
    # NOT DISPLAY THE WHOLE CH, BUT ONLY 10 CHS
    eeg = data[:, :num_ch]
    name = "EEG%d".format(idx)
    name = fig.add_subplot(1, 1, idx)
    name.set_xlim(0,100)
    #name.set_xticks(np.arange(100))
    name.set_xlabel('Time')
    dmin = eeg.min()
    dmax = eeg.max()
    dr = (dmax - dmin) * 0.7
    y0 = dmin
    y1 = ((num / 2) - 1) * dr + dmax
    name.set_ylim(y0, y1)

    segs = []
    ticklocs = []
    for i in range(num_ch):
      segs.append(np.hstack((t[:, np.newaxis], eeg[:, i, np.newaxis])))
      ticklocs.append(i*dr)

    offsets = np.zeros((num_ch, 2), dtype=float)
    offsets[:,1] = ticklocs

    lines = LineCollection(segs, offsets=offsets, transOffset=None)
    name.add_collection(lines)

  plt.tight_layout()
  #plt.show()
  filename = "./eeg_figs/sample_{:03d}.png".format(iid)
  fig.savefig(filename)
  plt.clf()
