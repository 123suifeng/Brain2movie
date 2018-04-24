import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn; cudnn.benchmark = True

class Generator(nn.Module):
  def __init__(self, hidden_size, num_Gen_ch):
    super(Generator, self).__init__()

    self.params = {0:[hidden_size,  num_Gen_ch * 8, 4, 1, 0], 1:[num_Gen_ch * 8, num_Gen_ch * 4, 4, 2, 1], 2:[num_Gen_ch * 4, num_Gen_ch * 2, 4, 2, 1], 3:[num_Gen_ch * 2, num_Gen_ch, 4, 2, 1], 4:[num_Gen_ch, 3, 4, 2, 1]}
    self.Gnet = nn.Sequential()
    for i in range(len(self.params)):
      layer_name = 'Gnet.conv{}-{}'.format(self.params[i][0], self.params[i][1])

      self.Gnet.add_module(layer_name, nn.ConvTranspose2d(self.params[i][0], self.params[i][1], self.params[i][2], self.params[i][3], self.params[i][4], bias=False))

      if i != len(self.params) - 1:
        self.Gnet.add_module('Gnet.batchnorm{}'.format(i), nn.BatchNorm2d(self.params[i][1]))
        self.Gnet.add_module('Gnet.relu{}'.format(i), nn.ReLU(True))    
      else:
        self.Gnet.add_module('Gnet.tanh', nn.Tanh())

  def forward(self, input):
    output = self.Gnet(input)
    return output

class Discriminator(nn.Module):
  def __init__(self, num_Dis_ch):
    super(Discriminator, self).__init__()
    self.params = {0:[3, num_Dis_ch, 4, 2, 1], 1:[num_Dis_ch, num_Dis_ch*2, 4, 2, 1], 2:[num_Dis_ch*2, num_Dis_ch*4, 4, 2, 1], 3:[num_Dis_ch*4, num_Dis_ch*8, 4, 2, 1], 4:[num_Dis_ch*8, 1, 4, 1, 0]}
    self.Dnet = nn.Sequential()
    for i in range(len(self.params)):
      layer_name = 'Dnet.conv{}-{}'.format(self.params[i][0], self.params[i][1])

      self.Dnet.add_module(layer_name, nn.Conv2d(self.params[i][0], self.params[i][1], self.params[i][2], self.params[i][3], self.params[i][4], bias=False))

      if i in [1, 2, 3]:
         self.Dnet.add_module('Dnet.batchnorm{}'.format(i), nn.BatchNorm2d(self.params[i][1]))
      if i in [0, 1, 2, 3]:
         self.Dnet.add_module('Dnet.relu{}'.format(i), nn.LeakyReLU(0.2, inplace=True))      
      if i == len(self.params) - 1:
         self.Dnet.add_module('Dnet.sigmoid', nn.Sigmoid())

  def forward(self, input):
    output = self.Dnet(input)
    return output.view(-1, 1).squeeze(1)

def LossFunc(opt):
  if opt.lossfunc == 'bce':
    criterion = nn.BCELoss()
  elif opt.lossfunc == '':
    print("??")
  else:
    print("GAN LOSS FUNCTION IS NOT IMPLEMENTED")
    raise NotImplementedError

  criterion.cuda()

  return criterion


def Optimizer(opt, gen, dis):
  if opt.gan_optim == 'Adam':
    g_optimizer = optim.Adam(gen.parameters(), lr=opt.g_learning_rate)
    d_optimizer = optim.Adam(dis.parameters(), lr=opt.d_learning_rate)

  elif opt.gan_optim == 'rmsprop':
    g_optimizer = optim.RMSProp(gen.parameters(), lr=opt.g_learning_rate)
    d_optimizer = optim.RMSProp(dis.parameters(), lr=opt.d_learning_rate)

  else:
    print("GAN OPTIMIZER IS NOT IMPLEMENTED")
    raise NotImplementedError

  return g_optimizer, d_optimizer

def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    m.weight.data.normal_(0.0, 0.02)
  elif classname.find('BatchNorm') != -1:
    m.weight.data.normal_(1.0, 0.02)
    m.bias.data.fill_(0)
