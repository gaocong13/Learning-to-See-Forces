#!/usr/bin/env python

import os, sys
import numpy as np

import torch
from torch.autograd import Variable
from network.TCNNet import TCNNet

from tqdm import tqdm

NUM_FEATURE = 4608
# Where are the depth images and forces stored
result_dir = '../result/'

frame_rad = 7
n_frames = frame_rad*2 + 1
nbatch = 5
lr = 0.00001
SAVE_MODEL_CYCLE = 100
# Load in
all_files1 = np.load('../data/feature_vgg.npy')
all_files2 = np.load('../data/feature_ptnet.npy')
all_forces = np.load('../data/force_grd.npy')


all_files1 = np.reshape(all_files1, [-1,n_frames,4096])
all_files2 = np.reshape(all_files2, [-1,n_frames,512])
all_files = np.concatenate((all_files1, all_files2), axis = 2)
all_forces = np.reshape(all_forces, [-1,1])

print(np.shape(all_files))
print(np.shape(all_forces))

# Randomly order them and split into train and validation
nTotal = len(all_files)
nTrain = int(0.8*float(nTotal))
nVal = nTotal-nTrain

# Randomize frames
rand_inds = np.random.permutation(nTotal)

train_files = []
train_forces = []
val_files = []
val_forces = []
for i in range(nTrain):
	j = rand_inds[i]
	train_files.append(all_files[j])
	f = all_forces[j]
	train_forces.append(f)

for i in range(nTrain,nTotal):
	j = rand_inds[i]
	val_files.append(all_files[j])

	f = all_forces[j]
	val_forces.append(f)

print('\nCollected', nTrain, 'training samples and', nVal, 'testing samples\n')

out_model_dir = '../model/'

nb_epochs = 10000
start_epoch = 0

# Construct the model
ptnet_model_pretrained_file = '../data/pt_net_00041.pt'
snet = TCNNet(ptnet_model_pretrained_file, n_frames, nbatch)
snet.cuda()

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, snet.parameters()), lr)  # only optimize parameters that requires_grad ON

Xmf = np.zeros((nbatch,n_frames,NUM_FEATURE), dtype=np.float32)
ymf = np.zeros((nbatch,1), dtype=np.float32)

train_loss = []
vali_loss = []
vali_err = []
# Train
for epoch in tqdm(range(start_epoch,nb_epochs)):
	snet.train(True)

	# First train
	rand_inds = np.random.permutation(nTrain)

	sum_cost = 0.0
	Nd_cost = 0.0

	num_batches = nTrain // nbatch
	cost_save = []
	for batch_idx in range(num_batches):
		start_idx = batch_idx * nbatch
		end_idx = (batch_idx+1) * nbatch
		inds = np.arange(start_idx, min(end_idx,nTrain))

		Xmf[:] = 0
		ymf[:] = 0

		# Collect a batch
		for b in range(nbatch):
			# Get the features
			features = train_files[rand_inds[inds[b]]]
			Xmf[b] = np.array(features).astype(np.float32)
			# Store the target force measurement
			ymf[b] = train_forces[rand_inds[inds[b]]]

		# Convert to tensor variables
		Xtrain = Variable(torch.from_numpy(Xmf).cuda())
		ytrain = Variable(torch.from_numpy(ymf).cuda())

		# Forward pass
		y_pred = snet(Xtrain)

		# Compute loss
		loss = criterion(y_pred, ytrain)

		# Zero gradients, perform a backward pass, and update the weights.
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		cost = loss.item()
		cost_save.append(cost)
		sum_cost += cost
		Nd_cost += 1.0
		avg_cost = sum_cost/Nd_cost

	print("\rEpoch: {:02d} | AvgCost {:2.5f}".format(epoch, avg_cost), sys.stdout.flush())
	train_loss.append(avg_cost)

	# learning rate decays 0.1 every 1000 epochs
	if epoch%1000 == 0:
		lr = lr*0.1

	if epoch % SAVE_MODEL_CYCLE == 0:
		model_name = 'model' + str(epoch) + '.pt'
		out_model_file = os.path.join(out_model_dir, model_name)
		torch.save(snet.state_dict(), out_model_file)

	snet.train(False)

	# Now test
	#print('...evaluating...')
	test_err = 0.0
	Nd = 0.0
	num_batches = nVal // nbatch
	pre_save = []
	grd_save = []
	for batch_idx in range(num_batches):

		start_idx = batch_idx * nbatch
		end_idx = (batch_idx+1) * nbatch
		inds = np.arange(start_idx, min(end_idx,nVal))

		Xmf[:] = 0.0
		ymf[:] = 0.0

		# Collect a batch
		for b in range(nbatch):
			# Get the feature sequence and store
			features = val_files[inds[b]]
			Xmf[b] = np.array(features).astype(np.float32)

			# Store the target force measurement
			ymf[b] = val_forces[inds[b]]

		Xtest = Variable(torch.from_numpy(Xmf).cuda(), requires_grad=False)

		# Predict
		y_pred = snet(Xtest)

		# Evaluate
		for b in range(nbatch):
			expected_label = ymf[b,0]
			f = y_pred.data[b,0]
			predicted_label = f
			pre_save.append(predicted_label)
			grd_save.append(expected_label)
			test_err += abs(expected_label-predicted_label)
			Nd += 1.0

	# Calculate validation loss
	pre_save = np.array(pre_save)
	grd_save = np.array(grd_save)
	vali_mse = np.sum(np.square(pre_save-grd_save))/Nd
	vali_loss.append(vali_mse)
	vali_err.append(test_err/Nd)

	pre_name_latest = result_dir + 'pre_latest.txt'
	grd_name_latest = result_dir + 'grd_latest.txt'
	np.savetxt(pre_name_latest, pre_save)
	np.savetxt(grd_name_latest, grd_save)

	if epoch % SAVE_MODEL_CYCLE == 0:
		pre_name = result_dir + 'pre' + str(epoch) + '.txt'
		grd_name = result_dir + 'grd' + str(epoch) + '.txt'
		np.savetxt(pre_name, pre_save)
		np.savetxt(grd_name, grd_save)

	print('Vali loss:', vali_mse, 'Validation Error:', test_err/Nd, '\n')
