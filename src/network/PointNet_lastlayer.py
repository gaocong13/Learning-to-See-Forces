#!/usr/bin/env python

import os, sys
import numpy as np
import cv2

import torch
from torch.autograd import Variable
import torch.nn as nn



class PointNet(torch.nn.Module):
	def __init__(self, NUM_POINT, nb_classes):
		super(PointNet, self).__init__()

		# input transform parameters
		self.it_conv1 = nn.Conv2d(1, 64, (1,3))
		self.it_bn1 = nn.BatchNorm2d(64)
		self.it_conv2 = nn.Conv2d(64, 128, (1,1))
		self.it_bn2 = nn.BatchNorm2d(128)
		self.it_conv3 = nn.Conv2d(128, 1024, (1,1))
		self.it_bn3 = nn.BatchNorm2d(1024)
		self.it_pool4 = nn.MaxPool2d((NUM_POINT,1), stride=(NUM_POINT,1))
		self.it_fc5 = nn.Linear(1024, 512)
		self.it_fc6 = nn.Linear(512, 256)
		self.it_fc7 = nn.Linear(256, 3*3)

		# Start PointNet parameters
		self.conv1 = nn.Conv2d(1, 64, (1,3))
		self.bn1 = nn.BatchNorm2d(64)
		self.conv2 = nn.Conv2d(64, 64, (1,1))
		self.bn2 = nn.BatchNorm2d(64)

		# feature transform parameters
		self.ft_conv1 = nn.Conv2d(64, 64, (1,1))
		self.ft_bn1 = nn.BatchNorm2d(64)
		self.ft_conv2 = nn.Conv2d(64, 128, (1,1))
		self.ft_bn2 = nn.BatchNorm2d(128)
		self.ft_conv3 = nn.Conv2d(128, 1024, (1,1))
		self.ft_bn3 = nn.BatchNorm2d(1024)
		self.ft_pool4 = nn.MaxPool2d((NUM_POINT,1), stride=(NUM_POINT,1))
		self.ft_fc5 = nn.Linear(1024, 512)
		self.ft_fc6 = nn.Linear(512, 256)
		self.ft_fc7 = nn.Linear(256, 64*64)

		# ...back to PointNet parameters...
		self.conv3 = nn.Conv2d(64, 64, (1,1))
		self.bn3 = nn.BatchNorm2d(64)
		self.conv4 = nn.Conv2d(64, 128, (1,1))
		self.bn4 = nn.BatchNorm2d(128)
		self.conv5 = nn.Conv2d(128, 1024, (1,1))
		self.bn5 = nn.BatchNorm2d(1024)

		self.pool6 = nn.MaxPool2d((NUM_POINT,1), stride=(NUM_POINT,1))

		self.fc7 = nn.Linear(1024, 512)
		self.fc8 = nn.Linear(512, 256)
		self.fc9 = nn.Linear(256, nb_classes)

		self.relu = nn.ReLU()


	def input_transform(self, X):
		"""
		Input is batch_size x N x 3
		Output is transformation matrix of size batch_size x 3 x 3
		"""

		# First, add channels dimension for X so we can do 2d convolutions
		h = X.unsqueeze(1)  # dim=1 is the channels dimension in PyTorch for convolutions

		h = self.relu(self.it_bn1(self.it_conv1(h)))
		h = self.relu(self.it_bn2(self.it_conv2(h)))
		h = self.relu(self.it_bn3(self.it_conv3(h)))
		h = self.it_pool4(h)

		# Flatten
		h = h.view(h.size(0), -1)

		h = self.relu(self.it_fc5(h))
		h = self.relu(self.it_fc6(h))
		h = self.it_fc7(h)

		# Reshape to transformation matrix
		y = h.view((h.size(0),3,3))
		return y


	def feature_transform(self, X):
		"""
		Input is batch_size x K x N x 1  [K feature maps]
		Output is transformation matrix of size batch_size x K x K
		"""
		h = self.relu(self.ft_bn1(self.ft_conv1(X)))
		h = self.relu(self.ft_bn2(self.ft_conv2(h)))
		h = self.relu(self.ft_bn3(self.ft_conv3(h)))
		h = self.ft_pool4(h)

		# Flatten
		h = h.view(h.size(0), -1)

		h = self.relu(self.ft_fc5(h))
		h = self.relu(self.ft_fc6(h))
		h = self.ft_fc7(h)

		# Reshape to transformation matrix
		y = h.view((h.size(0),64,64))
		return y


	def forward(self, X):
		"""
		Input is point cloud of size batch_size x N x 3
		Output is batch_size x 40 (class-conditional probs)
		"""

		transform = self.input_transform(X)
		point_cloud_transformed = torch.bmm(X,transform)
		input_image = point_cloud_transformed.unsqueeze(1)

		h = self.relu(self.bn1(self.conv1(input_image)))
		h = self.relu(self.bn2(self.conv2(h)))
		transform = self.feature_transform(h)

		h = torch.transpose(h, 1, 2)
		h = h.squeeze(3)
		h = torch.bmm(h, transform)
		h = h.unsqueeze(3)
		h = torch.transpose(h, 1, 2)

		h = self.relu(self.bn3(self.conv3(h)))
		h = self.relu(self.bn4(self.conv4(h)))
		h = self.relu(self.bn5(self.conv5(h)))
		h = self.pool6(h)

		# Flatten
		h = h.view(h.size(0), -1)

		h = self.relu(self.fc7(h))
		h = self.relu(self.fc8(h))
		y = self.fc9(h)

		return y


	def feature(self, X):
		"""
		Input is point cloud of size batch_size x N x 3
		Output is batch_size x 40 (class-conditional probs)
		"""

		transform = self.input_transform(X)
		point_cloud_transformed = torch.bmm(X,transform)
		input_image = point_cloud_transformed.unsqueeze(1)

		h = self.relu(self.bn1(self.conv1(input_image)))
		h = self.relu(self.bn2(self.conv2(h)))
		transform = self.feature_transform(h)

		h = torch.transpose(h, 1, 2)
		h = h.squeeze(3)
		h = torch.bmm(h, transform)
		h = h.unsqueeze(3)
		h = torch.transpose(h, 1, 2)

		h = self.relu(self.bn3(self.conv3(h)))
		h = self.relu(self.bn4(self.conv4(h)))
		h = self.relu(self.bn5(self.conv5(h)))
		h = self.pool6(h)

		# Flatten
		h = h.view(h.size(0), -1)

		f = self.relu(self.fc7(h))
		#f = self.relu(self.fc8(h))

		return f



