import tensorflow as tf
import numpy as np
from utils import load_train_data, generate_set, load_cvae_data
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class DDN_vis_memory():
	def __init__(self, layers, feature_dim, num_users, num_items, lambda_w, learning_rate, num_epoch, batch_size, train_path,
					valid_path, test_path, dataset_name, pretrain_path, input_dim, sparsity):
		self.layers = layers
		self.f_dim = feature_dim
		self.num_users = num_users
		self.num_items = num_items
		self.lambda_w = lambda_w
		self.lr = learning_rate
		self.num_epoch = num_epoch
		self.batch_size = batch_size
		self.train_path =train_path
		self.valid_path = valid_path
		self.test_path = test_path
		self.dataset_name = dataset_name
		self.pretrain_path = pretrain_path
		self.sparsity = sparsity

		self.user_mean_distribution = []
		self.user_cov_distribution = []
		self.item_mean_distribution = []
		self.item_cov_distribution = []


		### div line ###parameters for cvae
		self.dims = layers[0:2]
		self.input_dim = input_dim
		self.activations = ['sigmoid', 'sigmoid']
		self.n_z = layers[-1]
		# self.batch_size = 128
		self.weights = []
		self.content_data = load_cvae_data(self.dataset_name)

		### div line ###parameters for cvae

		with open(self.train_path, 'r') as f:
			self.train_content = f.readlines()

		self.train_set, self.valid_set, self.test_set = generate_set(self.train_path, self.valid_path, self.test_path)

		z_mean, z_log_sigma_sq = self.get_pretrain_weights(tf.constant(self.content_data, dtype=tf.float32))

		tf.reset_default_graph()
		
		self.z_mean = tf.constant(z_mean)
		self.z_log_sigma_sq = tf.constant(z_log_sigma_sq)





	def build_model(self):
		self.user_idx = tf.placeholder(dtype=tf.int32)
		self.item_pos_idx = tf.placeholder(dtype=tf.int32)
		self.item_neg_idx = tf.placeholder(dtype=tf.int32)

		self.user_attention_items_idx = tf.placeholder(dtype=tf.int32)

		# self.content_pos = tf.placeholder(tf.float32, [None, self.input_dim], name='content_pos')
		# self.content_neg = tf.placeholder(tf.float32, [None, self.input_dim], name='content_neg')

		


		num_layers = len(self.layers)
		with tf.variable_scope("DDN_Variable"):
			self.Um = tf.get_variable(name='user_latent_matrix', shape=[self.num_users, self.f_dim], dtype=tf.float32)
			self.Vm = tf.get_variable(name='item_latent_matrix', shape=[self.num_items, self.f_dim], dtype=tf.float32)
			# user's mean and covariance networks weights and bias
			
			user_mean_w = {}
			user_mean_b = {}
			user_cov_w = {}
			user_cov_b = {}
			item_mean_w = {}
			item_mean_b = {}
			item_cov_w = {}
			item_cov_b = {}
			for i in range(num_layers-1):
				# user_mean_w[i] = tf.get_variable(name='user_mean_w'+str(i), initializer=tf.random_normal(shape=[self.layers[i], self.layers[i+1]],
				# 					mean=0.01, stddev=0.02, dtype=tf.float32))
				# user_mean_b[i] = tf.get_variable(name='user_mean_b'+str(i), initializer=tf.random_normal(shape=[self.layers[i+1], ], 
				# 					mean=0.01, stddev=0.02, dtype=tf.float32))
				# user_cov_w[i] = tf.get_variable(name='user_cov_w'+str(i), initializer=tf.random_normal(shape=[self.layers[i], self.layers[i+1]],
				# 					mean=0.01, stddev=0.02, dtype=tf.float32))
				# user_cov_b[i] = tf.get_variable(name='user_cov_b'+str(i), initializer=tf.random_normal(shape=[self.layers[i+1], ],
				# 					mean=0.01, stddev=0.02, dtype=tf.float32))
				# item_mean_w[i] = tf.get_variable(name='item_mean_w'+str(i), initializer=tf.random_normal(shape=[self.layers[i], self.layers[i+1]],
				# 					mean=0.01, stddev=0.02, dtype=tf.float32))
				# item_mean_b[i] = tf.get_variable(name='item_mean_b'+str(i), initializer=tf.random_normal(shape=[self.layers[i+1], ],
				# 					mean=0.01, stddev=0.02, dtype=tf.float32))
				# item_cov_w[i] = tf.get_variable(name='item_cov_w'+str(i), initializer=tf.random_normal(shape=[self.layers[i], self.layers[i+1]],
				# 					mean=0.01, stddev=0.02, dtype=tf.float32))
				# item_cov_b[i] = tf.get_variable(name='item_cov_b'+str(i), initializer=tf.random_normal(shape=[self.layers[i+1], ], 
				# 					mean=0.01, stddev=0.02, dtype=tf.float32))
				
				user_mean_w[i] = tf.get_variable(name='user_mean_w'+str(i), initializer=tf.contrib.layers.xavier_initializer(), shape=[self.layers[i], self.layers[i+1]])
				user_mean_b[i] = tf.get_variable(name='user_mean_b'+str(i), initializer=tf.contrib.layers.xavier_initializer(), shape=[self.layers[i+1], ])
				user_cov_w[i] = tf.get_variable(name='user_cov_w'+str(i), initializer=tf.contrib.layers.xavier_initializer(), shape=[self.layers[i], self.layers[i+1]])
				user_cov_b[i] = tf.get_variable(name='user_cov_b'+str(i), initializer=tf.contrib.layers.xavier_initializer(), shape=[self.layers[i+1], ])
				item_mean_w[i] = tf.get_variable(name='item_mean_w'+str(i), initializer=tf.contrib.layers.xavier_initializer(), shape=[self.layers[i], self.layers[i+1]])
				item_mean_b[i] = tf.get_variable(name='item_mean_b'+str(i), initializer=tf.contrib.layers.xavier_initializer(), shape=[self.layers[i+1], ])
				item_cov_w[i] = tf.get_variable(name='item_cov_w'+str(i), initializer=tf.contrib.layers.xavier_initializer(), shape=[self.layers[i], self.layers[i+1]])
				item_cov_b[i] = tf.get_variable(name='item_cov_b'+str(i), initializer=tf.contrib.layers.xavier_initializer(), shape=[self.layers[i+1], ])

		with tf.variable_scope("Attention"):
			self.attention_w = tf.get_variable(name='attention_w', initializer=tf.contrib.layers.xavier_initializer(), shape=[self.n_z, self.n_z], dtype=tf.float32)
			self.attention_b = tf.get_variable(name='attention_b', initializer=tf.contrib.layers.xavier_initializer(), shape=[self.n_z, ], dtype=tf.float32)

		x_u = tf.gather(self.Um, self.user_idx)
		x_v_pos = tf.gather(self.Vm, self.item_pos_idx)
		x_v_neg = tf.gather(self.Vm, self.item_neg_idx)

		x_u = tf.reshape(x_u, (1, tf.shape(x_u)[0]))
		x_v_pos = tf.reshape(x_v_pos, (1, tf.shape(x_v_pos)[0]))
		x_v_neg = tf.reshape(x_v_neg, (1, tf.shape(x_v_neg)[0]))

		# calculate the mean and covariance vectors of the distribution of user, positive item, negative item
		'''
		for i in range(num_layers-1):
			if i == 0:
				u_mean = tf.add(tf.matmul(x_u, user_mean_w[i]), user_mean_b[i])
				u_mean = tf.nn.elu(u_mean)
				u_cov = tf.add(tf.matmul(x_u, user_cov_w[i]), user_cov_b[i])
				u_cov = tf.add(tf.nn.elu(u_cov), tf.ones(self.layers[i+1]))
				i_pos_mean = tf.add(tf.matmul(x_v_pos, item_mean_w[i]), item_mean_b[i])
				i_pos_mean = tf.nn.elu(i_pos_mean)
				i_pos_cov = tf.add(tf.matmul(x_v_pos, item_cov_w[i]), item_cov_b[i])
				i_pos_cov = tf.add(tf.nn.elu(i_pos_cov), tf.ones(self.layers[i+1]))
				i_neg_mean = tf.add(tf.matmul(x_v_neg, item_mean_w[i]), item_mean_b[i])
				i_neg_mean = tf.nn.elu(i_neg_mean)
				i_neg_cov = tf.add(tf.matmul(x_v_neg, item_cov_w[i]), item_cov_b[i])
				i_neg_cov = tf.add(tf.nn.elu(i_neg_cov), tf.ones(self.layers[i+1]))

			else:
				u_mean = tf.add(tf.matmul(u_mean, user_mean_w[i]), user_mean_b[i])
				u_mean = tf.nn.elu(u_mean)
				u_cov = tf.add(tf.matmul(u_cov, user_cov_w[i]), user_cov_b[i])
				u_cov = tf.add(tf.nn.elu(u_cov), tf.ones(self.layers[i+1]))
				i_pos_mean = tf.add(tf.matmul(i_pos_mean, item_mean_w[i]), item_mean_b[i])
				i_pos_mean = tf.nn.elu(i_pos_mean)
				i_pos_cov = tf.add(tf.matmul(i_pos_cov, item_cov_w[i]), item_cov_b[i])
				i_pos_cov = tf.add(tf.nn.elu(i_pos_cov), tf.ones(self.layers[i+1]))
				i_neg_mean = tf.add(tf.matmul(i_neg_mean, item_mean_w[i]), item_mean_b[i])
				i_neg_mean = tf.nn.elu(i_neg_mean)
				i_neg_cov = tf.add(tf.matmul(i_neg_cov, item_cov_w[i]), item_cov_b[i])
				i_neg_cov = tf.add(tf.nn.elu(i_neg_cov), tf.ones(self.layers[i+1]))
		'''
		u_mean = self.mean_network(x_u, user_mean_w, user_mean_b, num_layers)
		u_cov = self.covariance_network(x_u, user_cov_w, user_cov_b, num_layers)
		i_pos_mean = self.mean_network(x_v_pos, item_mean_w, item_mean_b, num_layers)
		i_pos_cov = self.covariance_network(x_v_pos, item_cov_w, item_cov_b, num_layers)
		i_neg_mean = self.mean_network(x_v_neg, item_mean_w, item_mean_b, num_layers)
		i_neg_cov = self.covariance_network(x_v_neg, item_cov_w, item_cov_b, num_layers)

		pos_content_mean = tf.gather(self.z_mean, self.item_pos_idx)
		pos_content_sigma = tf.gather(self.z_log_sigma_sq, self.item_pos_idx)
		neg_content_mean = tf.gather(self.z_mean, self.item_neg_idx)
		neg_content_sigma = tf.gather(self.z_log_sigma_sq, self.item_neg_idx)

		pos_content_mean = tf.reshape(pos_content_mean, (1, tf.shape(pos_content_mean)[0]))
		pos_content_sigma = tf.reshape(pos_content_sigma, (1, tf.shape(pos_content_sigma)[0]))
		neg_content_mean = tf.reshape(neg_content_mean, (1, tf.shape(neg_content_mean)[0]))
		neg_content_sigma = tf.reshape(neg_content_sigma, (1, tf.shape(neg_content_sigma)[0]))

		# Attention for user
		u_mean_att = tf.nn.tanh(tf.add(tf.matmul(u_mean, self.attention_w), self.attention_b))
		item_att_content = tf.gather(self.z_mean, self.user_attention_items_idx)
		weights = tf.nn.softmax(tf.matmul(u_mean_att, item_att_content, transpose_b=True))
		weights = tf.reshape(weights, shape=(tf.shape(self.user_attention_items_idx)[0], 1))
		new_u_mean = tf.reduce_sum(weights*item_att_content, 0)


		# regularization loss
		reg_loss = tf.constant(0, dtype=tf.float32)
		for i in range(num_layers-1):
			reg_loss = tf.add(reg_loss, tf.add(tf.nn.l2_loss(user_mean_w[i]), tf.nn.l2_loss(user_mean_b[i])))
			reg_loss = tf.add(reg_loss, tf.add(tf.nn.l2_loss(user_cov_w[i]), tf.nn.l2_loss(user_cov_b[i])))
			reg_loss = tf.add(reg_loss, tf.add(tf.nn.l2_loss(item_mean_w[i]), tf.nn.l2_loss(item_mean_b[i])))
			reg_loss = tf.add(reg_loss, tf.add(tf.nn.l2_loss(item_cov_w[i]), tf.nn.l2_loss(item_cov_b[i])))
		reg_loss = tf.add(reg_loss, tf.add(tf.nn.l2_loss(self.attention_w), tf.nn.l2_loss(self.attention_b)))
		reg_loss = self.lambda_w*2*reg_loss # because tf.nn.l2_loss defaultly has a coefficiency 1/2

		# Wasserstein distance between user and items
		was_u_neg_i = self.cal_was_loss_tf(new_u_mean, i_neg_mean, u_cov, i_neg_cov)
		was_u_pos_i = self.cal_was_loss_tf(new_u_mean, i_pos_mean, u_cov, i_pos_cov)

		# Wasserstein distance between item and its content distribution
		was_pos_i_content = self.cal_was_loss_tf(i_pos_mean, pos_content_mean, i_pos_cov, tf.exp(pos_content_sigma))
		was_neg_i_content = self.cal_was_loss_tf(i_neg_mean, neg_content_mean, i_neg_cov, tf.exp(neg_content_sigma))

		sub_neg_pos = was_u_neg_i - was_u_pos_i
		sub_content_neg_pos = -was_pos_i_content - was_neg_i_content
		log_sigmoid = tf.log(tf.nn.sigmoid(sub_neg_pos+sub_content_neg_pos))
		was_loss = -tf.reduce_sum(log_sigmoid)

		'''Record the distance between the users and the positive items'''
		self.distances = was_u_pos_i

		# total loss
		self.loss = was_loss + reg_loss

		loss_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope="DDN_Variable|Attention")

		optimizer = tf.train.AdamOptimizer(self.lr)

		self.train_op = optimizer.minimize(self.loss, var_list=loss_vars)
		# gvs = optimizer.compute_gradients(self.loss, var_list=loss_vars)
		# capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs if grad is not None]
		# self.train_op = optimizer.apply_gradients(capped_gvs)

		

		return u_mean, u_cov, i_pos_mean, i_pos_cov




	def mean_network(self, x_u, weights, biases, num_layers):
		'''network used to calculate the mean'''
		mean = x_u
		for i in range(num_layers-1):
			mean = tf.add(tf.matmul(mean, weights[i]), biases[i])
			mean = tf.nn.elu(mean)
		return mean


	def covariance_network(self, x_u, weights, biases, num_layers):
		'''network used to calculate the covariance'''
		cov = x_u
		for i in range(num_layers-1):
			cov = tf.add(tf.matmul(cov, weights[i]), biases[i])
			cov = tf.add(tf.nn.elu(cov), tf.ones(self.layers[i+1]))
		return cov



	# note that this function is used to calculate the was loss
	# but due to batch training, we return a vector with each factor of it a wasserstein distance
	def cal_was_loss_tf(self, mean1, mean2, cov1, cov2):
		part1 = tf.reduce_sum(tf.square(tf.subtract(mean1, mean2)), 1)
		part2 = tf.reduce_sum(cov1 + cov2 - 2*tf.sqrt(tf.sqrt(cov1)*cov2*tf.sqrt(cov1)), 1)
		was_loss = part1 + part2
		return was_loss # was_loss is a vector


	def cal_was_distance(self, mean1, mean2, cov1, cov2):
		part1 = np.sum(np.square(mean1-mean2))
		part2 = np.sum(cov1+cov2-2*np.sqrt(np.sqrt(cov1)*cov2*np.sqrt(cov1)))
		was_distance = part1 + part2
		return was_distance



	def run_model(self):
		# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
		# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6) # allocate certain gpu memory
		# config = tf.ConfigProto(gpu_options=gpu_options)
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True

		total_batch = int(self.num_users/self.batch_size) + 1
		u_mean, u_cov, i_mean, i_cov = self.build_model()
		init = tf.global_variables_initializer()
		with tf.Session(config=config) as sess:
			sess.run(init) # initialize all the variables
			# load weights
			epoch_pre = 999999999
			for epoch in range(self.num_epoch):
				epoch_cost = 0
				for itr in range(total_batch):
					train_tuple = load_train_data(self.train_content, self.train_path, self.num_users, self.num_items, self.batch_size)
					for jj in range(self.batch_size):
						cost, _ = sess.run([self.loss, self.train_op], feed_dict={
							self.user_idx: train_tuple[jj, 0],
							self.item_pos_idx: train_tuple[jj, 1],
							self.item_neg_idx: train_tuple[jj, 2],
							self.user_attention_items_idx: self.train_set[train_tuple[jj, 0]]
							# self.content_pos: self.content_data[train_tuple[:, 1]],
							# self.content_neg: self.content_data[train_tuple[:, 2]]
							})

						epoch_cost = epoch_cost + cost
				
				'''
				for u in range(self.num_users):
					time1 = time.time()
					for i in range(self.num_items):
						self.R[u, i] = self.cal_was_distance(self.user_mean_distribution[u], self.item_mean_distribution[i], 
													self.user_cov_distribution[u], self.item_cov_distribution[i])
					time2 = time.time()
					break
				'''
				
				# time1 = time.time()
				# self.R[0, 0] = self.cal_was_distance(self.user_mean_distribution[0], self.item_mean_distribution[0], 
				# 									self.user_cov_distribution[0], self.item_cov_distribution[0])
				# time2 = time.time()
				print('##########################')
				print('Epoch '+str(epoch+1))
				print('##############')
				print('The total cost is: '+str(epoch_cost))
				# print('It took %.3f seconds to train an epoch'%(time2-time1))
				print('\n')

				# if epoch_pre > epoch_cost:
				# 	epoch_pre = epoch_cost
				# else:
				# 	break

			'''Testing'''
			'''Fix some bugs, test_set'''
			item_test_set = list(self.test_set.keys())
			num_item_test = len(item_test_set)
			HR = np.array([0.0]*5, dtype=np.float)
			NDCG = np.array([0.0]*5, dtype=np.float)
			for user in range(self.num_users):
				pos_item_test = self.test_set[user]
				negative_item_set = list(set(range(self.num_items))-set(self.train_set[user]))
				if user in self.valid_set:
					negative_item_set = list(set(negative_item_set)-set(self.valid_set[user]))
				if user in self.test_set:
					negative_item_set = list(set(negative_item_set)-set(self.test_set[user]))
				neg_item_test = np.random.choice(negative_item_set, 99, replace=False)
				# item_to_test = np.append(pos_item_test, neg_item_test)
				item_to_test = np.append(neg_item_test, pos_item_test)
				users = [user]*100
				ratings = []
				for kk in range(100):
					value = sess.run(self.distances, feed_dict={
							self.user_idx: users[kk],
							self.item_pos_idx: item_to_test[kk],
							self.user_attention_items_idx: self.train_set[users[kk]]
						})
					ratings.append(value)
				item_score = [(item_to_test[i], ratings[i]) for i in range(len(item_to_test))]

				item_score = sorted(item_score, key=lambda x: x[1])

				item_sort = [pair[0] for pair in item_score]

				'''ouput user 0's ratings for checking '''
				# if user == 0:
				# 	rating_sort = [pair[1] for pair in item_score]
				# 	with open('CHECKING_cold_attention.txt', 'a') as f:
				# 		for rating in  rating_sort:
				# 			f.write(str(rating)+'\n')

				r = []

				for i in item_sort:
					if i in pos_item_test:
						r.append(1)
					else:
						r.append(0)

				hr_1 = self.hr_at_k(r, 1)
				hr_3 = self.hr_at_k(r, 3)
				hr_5 = self.hr_at_k(r, 5)
				hr_7 = self.hr_at_k(r, 7)
				hr_10 = self.hr_at_k(r, 10)
				ndcg_1 = self.ndcg_at_k(r, 1)
				ndcg_3 = self.ndcg_at_k(r, 3)
				ndcg_5 = self.ndcg_at_k(r, 5)
				ndcg_7 = self.ndcg_at_k(r, 7)
				ndcg_10 = self.ndcg_at_k(r, 10)
				HR = HR + np.array([hr_1, hr_3, hr_5, hr_7, hr_10], dtype=np.float)
				NDCG = NDCG + np.array([ndcg_1, ndcg_3, ndcg_5, ndcg_7, ndcg_10], dtype=np.float)

			HR = HR / num_item_test
			NDCG = NDCG / num_item_test

			print('HR: '+str(HR))
			print('NDCG: '+str(NDCG))
			# print(type(HR[0]))

			
			with open('results/100_'+self.dataset_name+'_Record_HR_cold_%s_attention.txt'%(self.sparsity),'a') as f:
				f.write('Testing cold Result(dim=%d)(%d epochs): hr@1:%f  hr@3:%f  hr@5:%f  hr@7:%f  hr@10:%f\n'%(self.n_z,(epoch+1),HR[0],HR[1],HR[2],HR[3],HR[4]))
				# f.write(str(HR)+'\n')
			with open('results/100_'+self.dataset_name+'_Record_NDCG_cold_%s_attention.txt'%(self.sparsity), 'a') as f:
				f.write('Testing cold Result(dim=%d)(%d epochs): ndcg@1:%f ndcg@3:%f ndcg@5:%f ndcg@7:%f ndcg@10:%f\n'%(self.n_z,(epoch+1),NDCG[0],NDCG[1],NDCG[2],NDCG[3],NDCG[4]))
			


			'''
			self.user_mean_distribution, self.user_cov_distribution, self.item_mean_distribution, self.item_cov_distribution = \
					sess.run([u_mean, u_cov, i_mean, i_cov], feed_dict={
						self.user_idx: [idx for idx in range(self.num_users)],
						self.item_pos_idx: [idx for idx in range(self.num_items)],
						self.item_neg_idx: []
						})
			self.R = np.zeros([self.num_users, self.num_items])
			'''


	def hr_at_k(self, r, k):
		r = np.asfarray(r)[:k]
		return np.sum(r)


	def dcg_at_k(self, r, k, method=1):
		"""Score is discounted cumulative gain (dcg)
		Relevance is positive real values.  Can use binary
		as the previous methods.
		Returns:
			Discounted cumulative gain
		"""
		r = np.asfarray(r)[:k]
		if r.size:
			if method == 0:
				return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
			elif method == 1:
				return np.sum(r / np.log2(np.arange(2, r.size + 2)))
			else:
				raise ValueError('method must be 0 or 1.')
		return 0.


	def ndcg_at_k(self, r, k, method=1):
		"""Score is normalized discounted cumulative gain (ndcg)
		Relevance is positive real values.  Can use binary
		as the previous methods.
		Returns:
			Normalized discounted cumulative gain
		"""
		dcg_max = self.dcg_at_k(sorted(r, reverse=True), k, method)
		if not dcg_max:
			return 0.
		return self.dcg_at_k(r, k, method) / dcg_max


	def activate(self, linear, name):
		if name == 'sigmoid':
			return tf.nn.sigmoid(linear, name='encoded')
		elif name == 'softmax':
			return tf.nn.softmax(linear, name='encoded')
		elif name == 'linear':
			return linear
		elif name == 'tanh':
			return tf.nn.tanh(linear, name='encoded')
		elif name == 'relu':
			return tf.nn.relu(linear, name='encoded')


	def load_model(self, sess, weight_path):
		# logging.info("Loading weights from " + weight_path)
		self.saver.restore(sess, weight_path)


	def get_pretrain_weights(self, x):
		# load weights in the pretrain model and restore it
		### Variable Scope
		with tf.variable_scope("inference"):
			self.rec = {'W1': tf.get_variable("W1", [self.input_dim, self.dims[0]], 
				initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
			'b1': tf.get_variable("b1", [self.dims[0]], 
				initializer=tf.constant_initializer(0.0), dtype=tf.float32),
			'W2': tf.get_variable("W2", [self.dims[0], self.dims[1]], 
				initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
			'b2': tf.get_variable("b2", [self.dims[1]], 
				initializer=tf.constant_initializer(0.0), dtype=tf.float32),
			'W_z_mean': tf.get_variable("W_z_mean", [self.dims[1], self.n_z], 
				initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
			'b_z_mean': tf.get_variable("b_z_mean", [self.n_z], 
				initializer=tf.constant_initializer(0.0), dtype=tf.float32),
			'W_z_log_sigma': tf.get_variable("W_z_log_sigma", [self.dims[1], self.n_z], 
				initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
			'b_z_log_sigma': tf.get_variable("b_z_log_sigma", [self.n_z], 
				initializer=tf.constant_initializer(0.0), dtype=tf.float32)}
		self.weights += [self.rec['W1'], self.rec['b1'], self.rec['W2'], self.rec['b2'], self.rec['W_z_mean'],
						self.rec['b_z_mean'], self.rec['W_z_log_sigma'], self.rec['b_z_log_sigma']]

		with tf.variable_scope("generation"):
			self.gen = {'W2': tf.get_variable("W2", [self.n_z, self.dims[1]], 
					initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
				'b2': tf.get_variable("b2", [self.dims[1]], 
					initializer=tf.constant_initializer(0.0), dtype=tf.float32),
				'W1': tf.transpose(self.rec['W2']),
				'b1': self.rec['b1'],
				'W_x': tf.transpose(self.rec['W1']),
				'b_x': tf.get_variable("b_x", [self.input_dim], 
					initializer=tf.constant_initializer(0.0), dtype=tf.float32)}

		self.weights += [self.gen['W2'], self.gen['b2'], self.gen['b_x']]

		# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
		# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6) # allocate certain gpu memory
		# config = tf.ConfigProto(gpu_options=gpu_options)
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True

		# initialize the weights
		self.saver = tf.train.Saver(self.weights)

		with tf.Session(config=config) as sess:
			self.load_model(sess, self.pretrain_path)


			# self.reg_loss += tf.nn.l2_loss(rec['W1']) + tf.nn.l2_loss(rec['W2'])
			h1 = self.activate(
				tf.matmul(x, self.rec['W1']) + self.rec['b1'], self.activations[0])
			h2 = self.activate(
				tf.matmul(h1, self.rec['W2']) + self.rec['b2'], self.activations[1])
			z_mean = tf.matmul(h2, self.rec['W_z_mean']) + self.rec['b_z_mean']
			z_log_sigma_sq = tf.matmul(h2, self.rec['W_z_log_sigma']) + self.rec['b_z_log_sigma']

			# eps = tf.random_normal((self.batch_size, self.n_z), 0, 1, 
			# 	seed=0, dtype=tf.float32)
			eps = tf.random_normal((self.num_items, self.n_z), 0, 1, 
				seed=0, dtype=tf.float32)
			self.z = z_mean + tf.sqrt(tf.maximum(tf.exp(z_log_sigma_sq), 1e-10)) * eps

			# self.reg_loss += tf.nn.l2_loss(gen['W1']) + tf.nn.l2_loss(gen['W_x'])
			# h2 = self.activate(
			# 	tf.matmul(self.z, gen['W2']) + gen['b2'], self.activations[1])
			# h1 = self.activate(
			# 	tf.matmul(h2, gen['W1']) + gen['b1'], self.activations[0])
			# x_recon = tf.matmul(h1, gen['W_x']) + gen['b_x']
			
			z_mean = sess.run(z_mean)
			z_log_sigma_sq = sess.run(z_log_sigma_sq)

		return z_mean, z_log_sigma_sq