#!/usr/bin/env python

import os
import numpy as np
import cv2

import torch
from torch.autograd import Variable
from network.SamplerNet import ptSamplerNet

from tqdm import tqdm

MAX_NUM_POINTS = 22801  # 151*151
NUM_POINT = 2048  # number of 3D points per point cloud

# Some pre-defined Kinect v1 parameters

# The maximum depth used, in meters.
maxDepth = 10

#  RGB Intrinsic Parameters
fx_rgb = 5.1885790117450188e+02
fy_rgb = 5.1946961112127485e+02
cx_rgb = 3.2558244941119034e+02
cy_rgb = 2.5373616633400465e+02

# RGB Distortion Parameters
k1_rgb =  2.0796615318809061e-01
k2_rgb = -5.8613825163911781e-01
p1_rgb = 7.2231363135888329e-04
p2_rgb = 1.0479627195765181e-03
k3_rgb = 4.9856986684705107e-01

# Depth Intrinsic Parameters
fx_d = 5.8262448167737955e+02
fy_d = 5.8269103270988637e+02
cx_d = 3.1304475870804731e+02
cy_d = 2.3844389626620386e+02

# Depth Distortion Parameters
k1_d = -9.9897236553084481e-02
k2_d = 3.9065324602765344e-01
p1_d = 1.9290592870229277e-03
p2_d = -1.9422022475975055e-03
k3_d = -5.1031725053400578e-01

# Rotation
R = -np.array([[-1.0000, 0.0050, 0.0043],
	[-0.0051, -1.0000, -0.0037],
	[-0.0043, 0.0037, -1.0000]])

# 3D Translation (meters)
t_x = 2.5031875059141302e-02
t_z = -2.9342312935846411e-04
t_y = 6.6238747008330102e-04

# Parameters for making depth absolute
depthParam1 = 351.3
depthParam2 = 1092.5

# Use pre-computed point cloud normalization parameters
mX = -0.316556731064
mY = -0.222886944433
mZ = 0.775189828051
max_vec_len = 0.40638

# Convert depth image to point cloud using geometries
def depth_to_pc(depth_file):
	"""
	Convert a Kinect depth image to a point cloud.  Input is a depth image file.
	Output is an [N x 3] matrix of 3D points in units of meters.

	(The point cloud is normalized to zero mean and to fit within the unit circle)
	"""

	imgDepthAbs = cv2.imread(depth_file, -1)             # read "as is", in mm
	imgDepthAbs = imgDepthAbs.astype(np.float32)/1000.0  # convert to meters

	H = imgDepthAbs.shape[0]
	W = imgDepthAbs.shape[1]

	xx, yy, = np.meshgrid(range(0,W), range(0,H))

	xx = xx.flatten()
	yy = yy.flatten()
	imgDepthAbs = imgDepthAbs.flatten()
	npoints = len(imgDepthAbs)

	# Project all depth pixels into 3D (in the depth camera's coordinate system)
	points3d = np.zeros((npoints,3), dtype=np.float32)
	X = (xx - cx_d) * imgDepthAbs / fx_d
	Y = (yy - cy_d) * imgDepthAbs / fy_d
	Z = imgDepthAbs

	# Zero mean the points
	points3d[:,0] = X - mX
	points3d[:,1] = Y - mY
	points3d[:,2] = Z - mZ

	# Now get the point with the maximum length and normalize each point to that length
	# so all points fit within the unit sphere.
	points3d /= max_vec_len

	return points3d

# Where are the depth images and forces stored
data_dir = '../data'
# Back and Forward 7 frames
frame_rad = 7
n_frames = frame_rad*2 + 1

# Which indices should we train with and which should we test with?
trainval_indices = ['1-1']

min_force = 1e12
max_force = -1e12

# Load in
all_files = []
all_forces = []
for ti in trainval_indices:
	# Get the force file for this one
	force_file = os.path.join(data_dir, 'Force_ori/data%s-z.txt' % ti)
	force_readings = np.loadtxt(force_file, delimiter=',')
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
			image_file = os.path.join(data_dir, 'depthimage%s/data%d.png' % (ti,j))
			these_files.append(image_file)

		# Save them
		all_files.append(these_files)

# Randomly order them and split into train and validation
nTotal = len(all_files)
print('\nCollected', nTotal, 'total samples')

nbatch = 5
# Construct the model
ptnet_model_pretrained_file = data_dir+'/pt_net_00041.pt'
snet = ptSamplerNet(ptnet_model_pretrained_file, MAX_NUM_POINTS, NUM_POINT, n_frames)
snet.cuda()

Xmf = np.zeros((nbatch,n_frames,MAX_NUM_POINTS,3), dtype=np.float32)
ymf = np.zeros((nbatch,1), dtype=np.float32)

feature_ptnet = []
grd_ptnet = []
# Extract features
snet.train(False)
snet.ptnet_model.train(False)  # <--- never training
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
		point_clouds = []
		for depth_file in these_files:
			pc = depth_to_pc(depth_file)
			point_clouds.append(pc)

		Xmf[b] = np.array(point_clouds).astype(np.float32)
		# Store the target force measurement
		ymf[b] = all_forces[inds[b]]

	grd_ptnet.append(ymf)
	# Convert to tensor variables
	Xtrain = Variable(torch.from_numpy(Xmf).cuda())
	# Forward pass
	y_pred, pt_feats = snet(Xtrain)
	feature_ptnet.append(pt_feats.cpu().data.numpy())
	print("Batch:{:07d} / {:07d}".format(batch_idx+1, num_batches))

np.save(data_dir+'/feature_ptnet.npy', feature_ptnet)
