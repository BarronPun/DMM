import numpy as np
import tensorflow as tf
from DDN_vis_memory import DDN_vis_memory
import argparse

parser = argparse.ArgumentParser(description='DDN')
parser.add_argument('--layers', type=str, default='[128, 64, 32]')
# parser.add_argument('--f_dim', type=int, default=128)
parser.add_argument('--lambda_w', type=float, default=0.001)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--num_epoch', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--dataset_name', type=str, default='Citeulike-a')
parser.add_argument('--sparsity', type=str, default='full')

args = parser.parse_args()

# @param
layers = eval(args.layers)
f_dim = layers[0]
lambda_w = args.lambda_w
learning_rate = args.learning_rate
num_epoch = args.num_epoch
batch_size = args.batch_size
dataset_name = args.dataset_name
sparsity = args.sparsity

if dataset_name == 'Citeulike-a':
	if sparsity == '0.1':
		train_path = 'data/Citeulike-a/train_cold_users.dat'
		valid_path = 'data/Citeulike-a/valid_cold_users.dat'
		test_path = 'data/Citeulike-a/test_cold_users.dat'
	elif sparsity == 'full':
		train_path = 'data/Citeulike-a/train_users.dat'
		valid_path = 'data/Citeulike-a/valid_users.dat'
		test_path = 'data/Citeulike-a/test_users.dat'
	else:
		train_path = 'data/Citeulike-a/train_cold_users_%s.dat'%sparsity
		valid_path = 'data/Citeulike-a/valid_cold_users_%s.dat'%sparsity
		test_path = 'data/Citeulike-a/test_cold_users_%s.dat'%sparsity

	pretrain_path = 'data/Citeulike-a/model_citeulikea/%d/pretrain_citeulikea_%d'%(layers[2],layers[2])
	num_users = 5551
	num_items = 16980
	input_dim = 8000
elif dataset_name == 'Citeulike-t':
	if sparsity == '0.1':
		train_path = 'data/Citeulike-t/train_cold_users.dat'
		valid_path = 'data/Citeulike-t/valid_cold_users.dat'
		test_path = 'data/Citeulike-t/test_cold_users.dat'
	elif sparsity == 'full':
		train_path = 'data/Citeulike-t/train_users.dat'
		valid_path = 'data/Citeulike-t/valid_users.dat'
		test_path = 'data/Citeulike-t/test_users.dat'
	else:
		train_path = 'data/Citeulike-t/train_cold_users_%s.dat'%sparsity
		valid_path = 'data/Citeulike-t/valid_cold_users_%s.dat'%sparsity
		test_path = 'data/Citeulike-t/test_cold_users_%s.dat'%sparsity

	pretrain_path = 'data/Citeulike-t/model_citeuliket/%d/pretrain_citeuliket_%d'%(layers[2],layers[2])
	num_users = 7947
	num_items = 25975
	input_dim = 20000

mDDN = DDN_vis_memory(layers, f_dim, num_users, num_items, lambda_w, learning_rate, num_epoch, batch_size, 
			train_path, valid_path, test_path, dataset_name, pretrain_path, input_dim, sparsity)

mDDN.run_model()
