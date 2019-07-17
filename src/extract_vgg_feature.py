#!/usr/bin/env python

import os
import numpy as np
import cv2

import torch
from torch.autograd import Variable
from network.SamplerNet import vggSamplerNet
from tqdm import tqdm

# Where are the depth images and forces stored
data_dir = '../data/'
# Back and Forward 7 frames
frame_rad = 7
n_frames = frame_rad*2 + 1
trainval_indices = ['1-1']

min_force = 1e12
max_force = -1e12

# Load in
all_files = []
all_forces = []
for ti in trainval_indices:
	# Get the force file for this one
	force_file = os.path.join(data_dir, 'Force_ori/data%s-z.txt' % ti)
	force_readings = np.loadtxt(force_file)
	N = force_readings.shape[0]
	N -= 1  # ignore last image sample

	for i in range(frame_rad,N-frame_rad):
		# The temporal window here will go back frame_rad and forward frame_rad.  The force
		# reading will come from the center of the window
		all_forces.append(force_readings[i])  # just take z-component (at the center of the sequence)

		# Update min/max force for normalization
		if force_readings[i] > max_force:
			max_force = force_readings[i]
		if force_readings[i] < min_force:
			min_force = force_readings[i]

		these_files = []
		for k in range(i-frame_rad,i+frame_rad+1):
			# Get the image filename for this force reading (centered on the temporal window)
			j=k+1  # image file names are 1-based, not 0-based
			image_file = os.path.join(data_dir, 'rgbimage%s/data%d.png' % (ti,j))
			these_files.append(image_file)

		# Save them
		all_files.append(these_files)

# Randomly order them and split into train and validation
nTotal = len(all_files)
print('\nCollected', nTotal, 'total samples')

nbatch = 5
# Construct the model
snet = vggSamplerNet(n_frames)
snet.cuda()

Xmf = np.zeros((nbatch,n_frames,224, 224,3), dtype=np.float32)
ymf = np.zeros((nbatch,1), dtype=np.float32)

feature_vgg = []
force_grd = []
# Extract features
snet.train(False)
snet.vgg16_model.train(False)  # <--- never training
num_batches = nTotal // nbatch
cost_save = []
for batch_idx in tqdm(range(num_batches)):
	start_idx = batch_idx * nbatch
	end_idx = (batch_idx+1) * nbatch
	inds = np.arange(start_idx, min(end_idx,nTotal))
	Xmf[:] = 0
	ymf[:] = 0
	# Collect a batch
	for b in range(nbatch):
		# Get the point cloud sequence and store
		these_files = all_files[inds[b]]
		rgb_images = []
		for depth_file in these_files:
			rgb_img = cv2.imread(depth_file, -1)
			rgb_images.append(rgb_img)
		Xmf[b] = np.array(rgb_images).astype(np.float32)
		# Store the target force measurement
		ymf[b] = all_forces[inds[b]]

	Xtrain = Variable(torch.from_numpy(Xmf).cuda())
	y_pred, vgg_feats = snet(Xtrain)
	feature_vgg.append(vgg_feats.cpu().data.numpy())
	force_grd.append(ymf)
	print("Batch:{:07d} / {:07d}".format(batch_idx+1, num_batches))

print(np.shape(feature_vgg))
print(np.shape(force_grd))
np.save(data_dir+'/feature_vgg.npy', feature_vgg)
np.save(data_dir+'/force_grd.npy', force_grd)
