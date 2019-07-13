#!/usr/bin/env python

import os, sys
import numpy as np
import cv2

import torch
from torch.autograd import Variable
import torch.nn as nn

# Construct the sampler network
class SamplerNet(torch.nn.Module):
	def __init__(self, ptnet_model_pretrained_file, n_frames, nbatch):
		super(SamplerNet, self).__init__()
		# Then we collect these features over time from the point cloud
		# sequence and train a simple TCN model.
		self.t_linear = nn.Linear(4608,256)
		self.t_conv1 = nn.Conv2d(256, 64, (1,7))   # 1x9 --> 1x5
		self.t_bn1 = nn.BatchNorm2d(64)
		self.t_conv2 = nn.Conv2d(64, 128, (1,3))   # 1x5 --> 1x3
		self.t_bn2 = nn.BatchNorm2d(128)
		self.t_conv3 = nn.Conv2d(128, 256, (1,3))  # 1x3 --> 1x1
		self.t_bn3 = nn.BatchNorm2d(256)

		# And the final FC and regression layer
		self.fc1 = nn.Linear(1280,1280)

		self.relu = nn.ReLU()
		self.tanh = nn.Tanh()
		self.linear = nn.Linear(1280,1)
		self.n_frames = n_frames
		self.nbatch = nbatch


	def forward(self, feats, ):
		B = self.nbatch
		T = self.n_frames
		# Reshape back out to get the frames
		feats = self.t_linear(feats)
		feats = feats.view(B,T,256)
		######################################################
		## Do TCN
		feats = torch.transpose(feats, 1, 2)  # [B x 256 x T]
		feats = feats.unsqueeze(2)  		  # [B x 256 x 1 x T]
		h = self.relu(self.t_bn1(self.t_conv1(feats)))
		h = self.relu(self.t_bn2(self.t_conv2(h)))
		h = self.relu(self.t_bn3(self.t_conv3(h)))
		h = h.view(h.size(0), -1)             # flatten
		##
		######################################################

		######################################################
		## Do the regression
		test = self.fc1(h)
		y = self.linear(self.fc1(h))
		##
		######################################################

		return y
