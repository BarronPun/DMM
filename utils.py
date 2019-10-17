import numpy as np
import random as rd
import scipy.io
import h5py


def load_train_data(content, path, num_users, num_items, batch_size):
	'''
	Generate the training and testing data
	'''
	train_tuple = []
	user_idx_list = np.random.choice(num_users, batch_size, replace=False)
	for u_idx in user_idx_list:
		# 2 positive items, 20 negative items
		item_list = content[u_idx].strip().split(' ')[1:]
		item_list = np.array(item_list, dtype=np.int32) # To np.array, type = int
		# rand_idx = np.random.choice(len(item_list), 2, replace=False)
		# pos_item_idx_list = item_list[rand_idx]
		# neg_item_idx_list = []
		# count = 0
		# rand_idx = np.random.randint(num_items, size=1)[0]
		# while count < 20:
		# 	if (rand_idx in item_list) or (rand_idx in neg_item_idx_list):
		# 		rand_idx = np.random.randint(num_items, size=1)[0]
		# 	else:
		# 		neg_item_idx_list.append(rand_idx)
		# 		rand_idx = np.random.randint(num_items, size=1)[0]
		# 		count += 1
		# neg_item_idx_list = np.array(neg_item_idx_list)
		# for p_idx in pos_item_idx_list:
		# 	for n_idx in neg_item_idx_list:
		# 		train_tuple.append([u_idx, p_idx, n_idx])
		pos_item_idx = rd.choice(item_list)
		neg_item_list = list(set(range(num_items))-set(item_list))
		neg_item_idx = rd.choice(neg_item_list)
		train_tuple.append([u_idx, pos_item_idx, neg_item_idx])

	return np.array(train_tuple)


def generate_set(train_path, valid_path, test_path):
	train_set = {}
	valid_set = {}
	test_set = {}
	with open(train_path, 'r') as f:
		content = f.readlines()
		for line in content:
			linetuple = line.strip().split(' ')
			train_set[int(linetuple[0])] = np.array(linetuple[1:], dtype=np.int)
	with open(valid_path, 'r') as f:
		content = f.readlines()
		for line in content:
			linetuple = line.strip().split(' ')
			valid_set[int(linetuple[0])] = np.array(linetuple[1:], dtype=np.int)
	with open(test_path, 'r') as f:
		content = f.readlines()
		for line in content:
			linetuple = line.strip().split(' ')
			test_set[int(linetuple[0])] = np.array(linetuple[1:], dtype=np.int)
	return train_set, valid_set, test_set


def load_cvae_data(dataset_name):
	data_dir = "data/%s/"%(dataset_name)
	# data_dir = 'data/flickr/'
	if dataset_name == 'Citeulike-a':
		variables = scipy.io.loadmat(data_dir + "mult_nor.mat")
		data = variables['X'] # when loaded with scipy.io
	elif dataset_name == 'Citeulike-t':
		variables = h5py.File(data_dir+"mult_nor.mat", 'r')
		data = variables['X'][:]
		data = data.T

	print(type(data))
	print(np.array(data).shape)

	return data


