#!/usr/bin/env python

import torch
import torch.nn as nn
import torchvision.models as models
from network.PointNet_lastlayer import PointNet

# Construct the VGG sampler network
class vggSamplerNet(torch.nn.Module):
	def __init__(self, n_frames):
		super(vggSamplerNet, self).__init__()
		# Load in pre-trained VGG16
		self.vgg16_model = models.vgg16(pretrained=True)
		self.vgg16_model.classifier = nn.Sequential(*(self.vgg16_model.classifier[i] for i in range(2)))
		for param in self.vgg16_model.parameters():
			param.requires_grad = False  # freeze all weights
		self.vgg16_model.cuda()
		self.n_frames = n_frames

	def forward(self, X):
		B = X.size(0)  # batch size of input
		T = X.size(1)  # number of frames of sequence of each input
		y = X.view(B*T,3, 224, 224)
		## Get the features from the pre-trained model
		feats = self.vgg16_model(y)  # [B*T x 256] <--- 256 is feature dims of hidden layer we are capturing
		# Reshape back out to get the frames
		feats = feats.view(B,T,4096)

		return y, feats

def repeat(x, n_repeats):
	rep = torch.ones(1,n_repeats)
	rep = rep.long()
	x = x.view(-1,1)
	result = torch.matmul(x, rep)
	result = result.view(-1,)
	return result

def repeat2(x, n_repeats):
        rep = torch.ones(1,n_repeats)
        x = x.view(-1,1)
        result = torch.matmul(x, rep)
        result = torch.t(result)
        result = result.contiguous().view(-1,)
        return result
# Construct the PointNet sampler network
class ptSamplerNet(torch.nn.Module):
	def __init__(self, ptnet_model_pretrained_file, MAX_NUM_POINTS, NUM_POINT, n_frames):
		super(ptSamplerNet, self).__init__()

		# Load in pre-trained PointNet
		self.ptnet_model = PointNet(NUM_POINT, 40)
		self.ptnet_model.load_state_dict(torch.load(ptnet_model_pretrained_file))
		for param in self.ptnet_model.parameters():
			param.requires_grad = False  # freeze all weights
		self.ptnet_model.cuda()

		self.h_samp = torch.arange(0, MAX_NUM_POINTS, MAX_NUM_POINTS/NUM_POINT)

		# The sampler parameters
		self.conv1 = nn.Conv2d(1, 64, (1,3))
		self.bn1 = nn.BatchNorm2d(64)
		self.conv2 = nn.Conv2d(64, 128, (1,1))
		self.bn2 = nn.BatchNorm2d(128)
		self.pool3 = nn.MaxPool2d((MAX_NUM_POINTS,1), stride=(MAX_NUM_POINTS,1))
		self.conv4 = nn.Conv2d(128, NUM_POINT, (1,1))
		self.bn4 = nn.BatchNorm2d(NUM_POINT)
		self.conv5 = nn.Conv2d(1, 1, (1,1))

		# Then we collect these features over time from the point cloud
		# sequence and train a simple TCN model.
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

		self.NUM_POINT = NUM_POINT
		self.n_frames = n_frames


	def forward(self, X):
		"""
		Input is point cloud of size batch_size(B) x n_frames(T) x N x 3
		Output is batch_size x 1 of regression predictions

		Try to learn to down-sample Nx3 point clouds to Mx3 point clouds, where M << N
		"""
		B = X.size(0)  # batch size of input
		T = X.size(1)  # number of frames of sequence of each input
		N = X.size(2)  # number of 3D points on each frame of each sequence
		M = self.NUM_POINT  # number of desired 3D points from on each frame of each sequence after down-sampling\
		## Do the uniform down-sampling
		X3 = X.view(-1,3)
		base = repeat(torch.arange(0, B*T)*N, M)           # [(B*T*M)]
		base = base.long()
		h_use = self.h_samp.float()
		h_use = repeat2(h_use, B*T)
		h_use = h_use.long()
		y = X3[h_use.cuda()+base.cuda()]                    # [(B*T*M) X 3]
		y = y.view(B*T,M,3)
		## Get the features from the pre-trained model
		feats = self.ptnet_model.feature(y)  # [B*T x 256] <--- 256 is feature dims of hidden layer we are capturing
		# Reshape back out to get the frames
		feats = feats.view(B,T,512)

		return y, feats
