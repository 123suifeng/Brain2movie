from __future__ import division # IN PYTHON3, IT IS NOT NEEDED
# Define options
import argparse
parser = argparse.ArgumentParser(description="Template")
# Dataset options
parser.add_argument('-ed', '--eeg-dataset', default="datasets/eeg_signals_128_sequential_band_all_with_mean_std.pth", help="EEG dataset path")
parser.add_argument('-ss', '--seq-start', default=20, type=int, help="start frame of sequence")
parser.add_argument('-se', '--seq-end', default=450, type=int, help="end frame of sequence")
parser.add_argument('-sp', '--splits-path', default="datasets/splits_by_image.pth", help="splits path")
parser.add_argument('-sn', '--split-num', default=0, type=int, help="split number")
parser.add_argument('-imsz', '--image-size', default=64, type=int, help="Input Image resize")
# Model options
parser.add_argument('-ll', '--lstm-layers', default=1, type=int, help="LSTM layers")
parser.add_argument('-ls', '--lstm-size', default=50, type=int, help="LSTM hidden size")
parser.add_argument('-os', '--output-size', default=40, type=int, help="output layer size")
parser.add_argument('-gn', '--generator-size', default=64, type=int, help="generator layer size")
parser.add_argument('-dn', '--discriminator-size', default=64, type=int, help="discriminator layer size")
# Training options
parser.add_argument("-b", "--batch-size", default=16, type=int, help="batch size")
parser.add_argument('-o', '--optim', default="Adam", help="optimizer")
parser.add_argument('-go', '--gan-optim', default="Adam", help="optimizer")
parser.add_argument('-llr', '--lstm-learning-rate', default=0.001, type=float, help="lstm learning rate")
parser.add_argument('-glr', '--g-learning-rate', default=0.001, type=float, help="generator learning rate")
parser.add_argument('-dlr', '--d-learning-rate', default=0.001, type=float, help="discriminator learning rate")
parser.add_argument('-lrdb', '--learning-rate-decay-by', default=0.5, type=float, help="learning rate decay factor")
parser.add_argument('-lrde', '--learning-rate-decay-every', default=10, type=int, help="learning rate decay period")
parser.add_argument('-dw', '--data-workers', default=4, type=int, help="data loading workers")
parser.add_argument('-e', '--epochs', default=100, type=int, help="training epochs")
parser.add_argument('-lf', '--lossfunc', default='bce', help="gan loss function")
# Backend options
parser.add_argument('--no-cuda', default=False, help="disable CUDA", action="store_true")

# Parse arguments
opt = parser.parse_args()

# Imports
import sys
import os
import random
import math
import time
import torch; torch.utils.backcompat.broadcast_warning.enabled = True
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn; cudnn.benchmark = True
import models.lstm_model as lm
import models.gan_model as gm
import utils.loaddata

# Create result folder (if exists, just declare the directory)
result_path = "%s/%d_%d_%d".format('result', opt.seq_start, opt.seq_end, opt.lstm_size)
if not os.path.exists(result_path):
    os.makedirs(result_path)

result_filename = "%s/%d_%d_%d.txt".format(result_path, opt.seq_start, opt.seq_end, opt.lstm_size)
result_file = open(result_file, 'w')

# Load dataset
dataset = loaddata.EEGDataset(opt.eeg_dataset)
# Create loaders
loaders = {split: DataLoader(loaddata.Splitter(dataset, split_path = opt.splits_path, split_num = opt.split_num, split_name = split), batch_size = opt.batch_size, drop_last = True, shuffle = True) for split in ["train", "val", "test"]}

models = {}
optimizers = {}

models['lstm'] = lm.Model(128, opt.lstm_size, opt.lstm_layers, opt.output_size)
models['gen'] = gm.Generator(opt.lstm_size, opt.generator_size)
models['dis'] = gm.Discriminator(opt.discriminator_size)
optimizers['lstm'] = lm.Optimizer(opt, models['lstm'])
optimizers['gen'], optimizers['dis'] = gm.Optimizer(opt, models['gen'], models['dis'])

# Apply Initial weight to models
for model in models:
    if model != 'lstm':
        models[model].apply(gm.weights_init)

# Setup tensors
gan_input = torch.FloatTensor(opt.batch_size, 3, opt.image_size, opt.image_size)
gan_label = torch.FloatTensor(opt.batch_size)
noise = torch.FloatTensor(opt.batch_size, opt.lstm_size, 1, 1)
fixed_noise = torch.FloatTensor(opt.batch_size, opt.lstm_size, 1, 1).normal_(0, 1)
real_label = 1
fake_label = 0

# Setup CUDA
if not opt.no_cuda:
    for model in models:
        models[model].cuda()

    gan_input, gan_label, noise, fixed_noise = gan_input.cuda(), gan_label.cuda(), noise.cuda(), fixed_noise.cuda()

    print("Copied to CUDA")

fixed_noise = Variable(fixed_noise)
 
# Start training
for epoch in range(1, opt.epochs+1):
    # Initialize loss/accuracy variables
    losses = {"train": 0, "val": 0, "test": 0}
    accuracies = {"train": 0, "val": 0, "test": 0}
    counts = {"train": 0, "val": 0, "test": 0}

    # Process each split
    for split in ("train", "val", "test"):
        # Set network mode
        if split == "train":
            models['lstm'].train()
        else:
            models['lstm'].eval()
        # Process all split batches
        for i, (input, target) in enumerate(loaders[split]): # input : EEG tensors shapes (batch_size, 430, 128), target : annotation

            # Check CUDA
            if not opt.no_cuda:
                input = input.cuda(async = True)
                target = target.cuda(async = True)
            # Wrap for autograd
            input = Variable(input, volatile = (split != "train"))
            target = Variable(target, volatile = (split != "train"))
            # Forward
            # 1. LSTM Forward operation
            output, pre_output = models['lstm'](input)
            loss = F.cross_entropy(output, target)
            losses[split] += loss.data[0]
            # Compute accuracy
            _,pred = output.data.max(1)
            correct = pred.eq(target.data).sum()
            accuracy = correct / input.data.size(0)
            #if split == "val":
                #print('correct : {} -- input.data.size(0) : {} -- accuracy : {}'.format(correct, input.data.size(0), accuracy))
            accuracies[split] += accuracy
            counts[split] += 1
            # Backward and optimize
            if split == "train":
                optimizers['lstm'].zero_grad()
                loss.backward()
                optimizers['lstm'].step()


            # 2. Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            models['dis'].zero_grad()
            # TRAIN WITH REAL
            data = torch.FloatTensor(opt.batch_size, 3, opt.image_size, opt.image_size).fill_(0) # This is a real image
            data.cuda()
            gan_input.copy_(data)
            gan_label.resize_(opt.batch_size).fill_(real_label) # Fill with 1 (It is a real image!!!)
            gan_input_v = Variable(gan_input)
            gan_label_v = Variable(gan_label)
            gan_output = models['dis'](gan_input_v)
            errD_real = gm.LossFunc(opt)(gan_output, gan_label_v)
            errD_real.backward()
            D_x = gan_output.data.mean()
            
            # TRAIN WITH FAKE
            #gan_input_v = Variable(pre_output) # This is a latent vector from the classifier (lstm_size, 1)
            gan_input2 = pre_output.data
            gan_input2.resize_as_(noise)
            gan_input_v = Variable(gan_input2)
            fake = models['gen'](gan_input_v) # Generate an image from the generator
            #print('fake', fake.detach())
            gan_label_v = Variable(gan_label.fill_(fake_label)) # Fill with 0 (It is a fake image!!!)
            gan_output = models['dis'](fake)
            errD_fake = gm.LossFunc(opt)(gan_output, gan_label_v)
            errD_fake.backward(retain_graph=True)
            D_G_z1 = gan_output.data.mean()
            errD = errD_real + errD_fake
            optimizers['dis'].step()

            # 3. Update G network: maximize log(D(G(z)))
            models['gen'].zero_grad()
            gan_label_v = Variable(gan_label.fill_(real_label)) # Generator tries to fake the discriminator
            gan_output = models['dis'](fake)
            errG = gm.LossFunc(opt)(gan_output, gan_label_v)
            errG.backward()
            D_G_z2 = gan_output.data.mean()
            optimizers['gen'].step()
    # Print info at the end of the epoch
    print_text = "Epoch {0}: TrL={1:.4f}, TrA={2:.4f}, VL={3:.4f}, VA={4:.4f}, TeL={5:.4f}, TeA={6:.4f}\n".format(epoch,
                                                                                                         losses["train"]/counts["train"],
                                                                                                         accuracies["train"]/counts["train"],
                                                                                                         losses["val"]/counts["val"],
                                                                                                         accuracies["val"]/counts["val"],
                                                                                                         losses["test"]/counts["test"],
                                                                                                         accuracies["test"]/counts["test"])
    print(print_text)
    result_file.write(print_text)


    fake = models['gen'](fixed_noise)
    image_name = "%s/%02d.png".format(result_path, epoch)
    vutils.save_image(fake.data, image_name, normalize=True)

