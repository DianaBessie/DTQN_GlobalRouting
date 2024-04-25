
#!/usr/bin/env python
import keras, tensorflow as tf, numpy as np, sys, copy, argparse, random
import matplotlib.pyplot as plt
import math
import os

np.random.seed(10701) # 设置了 NumPy 随机数生成器的种子。
tf.set_random_seed(10701) # 设置了 TensorFlow 随机数生成器的种子。
random.seed(10701) # 设置了 Python 内置的随机数生成器的种子。

class QNetwork():

	# This class essentially defines the network architecture. 
	# The network should take in state of the world as an input, 
	# and output Q values of the actions available to the agent as the output. 

	def __init__(self, environment_name, networkname, trianable):
		# Define your network architecture here. It is also a good idea to define any training operations 
		# and optimizers here, initialize your variables, or alternately compile your model here.
		if environment_name == 'grid':
			self.nObservation = 12    # 观察空间大小
			self.nAction = 6    # 动作空间大小
			self.learning_rate = 0.0001     # 学习率
			self.architecture = [32, 64, 32]   # 神经网络层的维度

		kernel_init = tf.random_uniform_initializer(-0.5, 0.5)  # 权重
		bias_init = tf.constant_initializer(0)  # 偏置
		self.input = tf.placeholder(tf.float32, shape=[None, self.nObservation], name='input')   # 定义了网络的输入层
		with tf.variable_scope(networkname):  # 创建了三个隐藏层和一个输出层
			layer1 = tf.layers.dense(self.input, self.architecture[0], tf.nn.relu, kernel_initializer=kernel_init, bias_initializer=bias_init, name='layer1', trainable=trianable)
			layer2 = tf.layers.dense(layer1, self.architecture[1], tf.nn.relu, kernel_initializer=kernel_init, bias_initializer=bias_init, name='layer2', trainable=trianable)
			layer3 = tf.layers.dense(layer2, self.architecture[2], tf.nn.relu, kernel_initializer=kernel_init, bias_initializer=bias_init, name='layer3', trainable=trianable)
			self.output = tf.layers.dense(layer3, self.nAction, kernel_initializer=kernel_init, bias_initializer=bias_init, name='output', trainable=trianable)

		self.targetQ = tf.placeholder(tf.float32, shape=[None, self.nAction], name='target')  # 定义了目标Q值的占位符，用于训练时提供真实的Q值
		if trianable == True:
			self.loss = tf.losses.mean_squared_error(self.targetQ, self.output)  # 损失函数
			self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)  # 最小化损失

		with tf.variable_scope(networkname, reuse=True):   # 获取之前定义的网络层中的权重和偏置变量
			self.w1 = tf.get_variable('layer1/kernel')   # 从TensorFlow的计算图中第一层全连接层的权重
			self.b1 = tf.get_variable('layer1/bias')
			self.w2 = tf.get_variable('layer2/kernel')
			self.b2 = tf.get_variable('layer2/bias')
			self.w3 = tf.get_variable('layer3/kernel')
			self.b3 = tf.get_variable('layer3/bias')
			self.w4 = tf.get_variable('output/kernel')
			self.b4 = tf.get_variable('output/bias')


class Replay_Memory():  # 用于实现强化学习中的经验回放机制

	def __init__(self, memory_size=50000, burn_in=10000):

		# The memory essentially stores transitions recorder from the agent
		# taking actions in the environment.

		# Burn in episodes define the number of episodes that are written into the memory from the 
		# randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced. 
		# A simple (if not the most efficient) was to implement the memory is as a list of transitions.
		self.memory = []
		self.is_burn_in = False
		self.memory_max = memory_size
		self.burn_in = burn_in

	def sample_batch(self, batch_size=32):
		# This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples. 
		# You will feed this to your model to train.
		index = np.random.randint(len(self.memory), size=batch_size)
		batch = [self.memory[i] for i in index]
		return batch

	def append(self, transition):  # 向存储中添加一个新的过渡。如果存储达到最大容量，它会自动移除最早的过渡以保持存储大小不变。
		# Appends transition to the memory.
		self.memory.append(transition)
		if len(self.memory) > self.memory_max:
			self.memory.pop(0)



class DQN_Agent():
	# DQN（Deep Q-Network）算法中，通常使用两个网络：评估网络（evaluation network）和目标网络（target network）。
    # 评估网络负责根据当前策略为每个动作生成Q值，并根据这些Q值选择动作。
	# 目标网络则用于计算目标Q值，即执行动作后预期获得的累积奖励的估计。
	# 使用两个网络的目的是为了稳定学习过程，减少Q值估计中的偏差和方差，因为评估网络的频繁更新可能导致学习目标不断变化，从而使训练过程不稳定。
	# 目标网络的参数定期从评估网络复制过来，但更新频率较低。
	# In this class, we will implement functions to do the following. 
	# (1) Create an instance of the Q Network class.
	# (2) Create a function that constructs a policy from the Q values predicted by the Q Network. 
	#		(a) Epsilon Greedy Policy.
	# 		(b) Greedy Policy. 
	# (3) Create a function to train the Q Network, by interacting with the environment.
	# (4) Create a function to test the Q Network's performance on the environment.
	# (5) Create a function for Experience Replay.
	
	def __init__(self, environment_name, sess, gridgraph, render=False):

		# Create an instance of the network itself, as well as the memory. 
		# Here is also a good place to set environmental parameters,
		# as well as training parameters - number of episodes / iterations, etc.
		self.epsilon = 0.05
		
		if environment_name == 'grid':
			self.gamma = 0.95
		self.max_episodes = 200 #20000 #200
		self.batch_size = 32
		self.render = render

		self.qNetwork = QNetwork(environment_name, 'q', trianable=True)   # 评估网络
		self.tNetwork = QNetwork(environment_name, 't', trianable=False)  # 目标网络
		self.replay = Replay_Memory()

		self.gridgraph = gridgraph  # 同步目标网络与动作网络的权重和偏置参数与行动网络的

		self.as_w1 = tf.assign(self.tNetwork.w1, self.qNetwork.w1)
		self.as_b1 = tf.assign(self.tNetwork.b1, self.qNetwork.b1)
		self.as_w2 = tf.assign(self.tNetwork.w2, self.qNetwork.w2)
		self.as_b2 = tf.assign(self.tNetwork.b2, self.qNetwork.b2)
		self.as_w3 = tf.assign(self.tNetwork.w3, self.qNetwork.w3)
		self.as_b3 = tf.assign(self.tNetwork.b3, self.qNetwork.b3)
		self.as_w4 = tf.assign(self.tNetwork.w4, self.qNetwork.w4)
		self.as_b4 = tf.assign(self.tNetwork.b4, self.qNetwork.b4)


		self.init = tf.global_variables_initializer()

		self.sess = sess
		# tf.summary.FileWriter("logs/", self.sess.graph)
		self.sess.run(self.init)
		self.saver = tf.train.Saver(max_to_keep=20)  # 保存和恢复模型的参数

	def epsilon_greedy_policy(self, q_values):
		# Creating epsilon greedy probabilities to sample from.
		rnd = np.random.rand()  # 生成一个介于0到1之间的随机浮点数
		if rnd <= self.epsilon:
			return np.random.randint(len(q_values))
		else:
			return np.argmax(q_values)

	def greedy_policy(self, q_values):
		# Creating greedy policy for test time. 
		return np.argmax(q_values)

	def network_assign(self):
		# pass the weights of evaluation network to target network
		self.sess.run([self.as_w1, self.as_b1, self.as_w2, self.as_b2, self.as_w3, self.as_b3, self.as_w4, self.as_b4])

	def train(self,twoPinNum,twoPinNumEachNet,netSort,savepath,model_file=None):
		# ! savepath: "../model_(train/test)"
		# ! if model_file = None, training; if given, testing
    	# ! if testing using training function, comment burn_in in Router.py


		# In this function, we will train our network. 
		# If training without experience replay_memory, then you will interact with the environment 
		# in this function, while also updating your network parameters. 

		# If you are using a replay memory, you should interact with environment here, and store these 
		# transitions to memory, while also updating your model.

		# the model will be saved to ../model/
		# the training/testing curve will be saved as a .npz file in ../data/
		if model_file is not None:
			self.saver.restore(self.sess, model_file)  # 提供了model_file参数，则使用TensorFlow的Saver对象恢复之前保存的模型，以便进行测试或进一步训练

        # 记录训练过程中的奖励信息和测试奖励信息，以及记录测试情节的信息。
		reward_log = []
		test_reward_log = []
		test_episode = []
		# if not self.replay.is_burn_in:
		# 	self.burn_in_memory()
		solution_combo = []

        # 初始化了一些用于存储解决方案组合和奖励数据的列表
		reward_plot_combo = []
		reward_plot_combo_pure = []
		for episode in np.arange(self.max_episodes*len(self.gridgraph.twopin_combo)):
        # 对于每个可能的两终端网络，算法将尝试执行最多self.max_episodes次迭代，以训练模型或评估其性能。

			# n_node = len([n.name for n in tf.get_default_graph().as_graph_def().node])
			# print("No of nodes: ", n_node, "\n")

			# print('Route:',self.gridgraph.route)
			solution_combo.append(self.gridgraph.route)  # 将当前的路由结果添加到solution_combo列表中。

			state, reward_plot, is_best = self.gridgraph.reset()  # 重置环境到初始状态，并获取初始状态、奖励以及是否是最佳状态的标志。
			reward_plot_pure = reward_plot-self.gridgraph.posTwoPinNum*100  # 计算纯粹的奖励值，通过从原始奖励中减去一个基于两终端网络数量的固定值。
			# print('reward_plot-self.gridgraph.posTwoPinNum*100',reward_plot-self.gridgraph.posTwoPinNum*100)

            # 每当episode可被twoPinNum整除时，将当前奖励和纯粹奖励分别添加到reward_plot_combo和reward_plot_combo_pure列表中。
			if (episode) % twoPinNum == 0:
				reward_plot_combo.append(reward_plot)
				reward_plot_combo_pure.append(reward_plot_pure)
			is_terminal = False
			rewardi = 0.0
			if episode % 100 == 0:  # 每100个迭代更新一次网络参数。
				self.network_assign()

			rewardfortwopin = 0
			while not is_terminal:
				observation = self.gridgraph.state2obsv()  # 获取当前环境状态的观察值
				q_values = self.sess.run(self.qNetwork.output, feed_dict={self.qNetwork.input: observation})  # 使用观察值通过Q网络获取动作的Q值。
				action = self.epsilon_greedy_policy(q_values)  # 根据epsilon-贪婪策略选择一个动作
				# print(action)
				nextstate, reward, is_terminal, debug = self.gridgraph.step(action)  # 执行选定的动作，环境返回下一个状态、本次动作的奖励、是否终止的标志和调试信息。
				# print(nextstate)
				observation_next = self.gridgraph.state2obsv()  # 生成下一个状态的观察值。
				self.replay.append([observation, action, reward, observation_next, is_terminal])  # 将当前的转换（当前观察、动作、奖励、下一个观察、终止标志）添加到重放缓冲区。
				state = nextstate
				rewardi = rewardi+reward
				rewardfortwopin = rewardfortwopin + reward  # 更新当前状态和累积的奖励。

				batch = self.replay.sample_batch(self.batch_size)  # 从经验回放缓冲区中抽样一个批次的转换，为每个转换提取观察值、动作、奖励、下一个观察值和是否终止的标志。
				batch_observation = np.squeeze(np.array([trans[0] for trans in batch]))
				batch_action = np.array([trans[1] for trans in batch])
				batch_reward = np.array([trans[2] for trans in batch])
				batch_observation_next = np.squeeze(np.array([trans[3] for trans in batch]))
				batch_is_terminal = np.array([trans[4] for trans in batch])
				q_batch = self.sess.run(self.qNetwork.output, feed_dict={self.qNetwork.input: batch_observation})  # 使用评估网络（qNetwork）和当前观察值
				# （batch_observation）计算当前状态下每个动作的Q值。
				q_batch_next = self.sess.run(self.tNetwork.output, feed_dict={self.tNetwork.input: batch_observation_next})  # 使用目标网络（tNetwork）和下一个状态的观察值
				# （batch_observation_next）计算下一个状态下每个动作的Q值。
				y_batch = batch_reward+self.gamma*(1-batch_is_terminal)*np.max(q_batch_next, axis=1)  # 根据即时奖励（batch_reward）、折扣因子（gamma）、下一个状态的最大Q值
				# （np.max(q_batch_next, axis=1)），以及是否为终止状态（1-batch_is_terminal）来计算目标Q值。这个目标Q值用于训练评估网络，使其预测的Q值接近实际的期望值。

				targetQ = q_batch.copy()  # 复制了当前状态下的Q值数组
				targetQ[np.arange(self.batch_size), batch_action] = y_batch  # 对于批次中的每个动作，使用y_batch中计算出的目标Q值更新targetQ数组。
				_, train_error = self.sess.run([self.qNetwork.opt, self.qNetwork.loss], feed_dict={self.qNetwork.input: batch_observation, self.qNetwork.targetQ: targetQ})
			    # 执行训练步骤：通过传递当前观察值和目标Q值到网络，执行一次优化步骤(self.qNetwork.opt)并计算当前批次的训练误差(self.qNetwork.loss)。
			reward_log.append(rewardi)  # comment in test; do not save model test

			self.gridgraph.instantrewardcombo.append(rewardfortwopin)
		# 将特定两终端布线任务的累积奖励（存储在变量rewardfortwopin中）添加到gridgraph对象的成员列表instantrewardcombo中。
		# 这一操作可能用于跟踪大问题中每个两终端布线任务相关的即时奖励，可能用于分析或优化目的。

			# print(episode, rewardi)
			
			# if is_best == 1:
			# 	print('self.gridgraph.route',self.gridgraph.route)
		# 		print('Save model')
		# # 		test_reward = self.test()
		# # 		test_reward_log.append(test_reward/20.0)
		# # 		test_episode.append(episode)
		# 		save_path = self.saver.save(self.sess, "{}/model_{}.ckpt".format(savepath,episode))
		# 		print("Model saved in path: %s" % savepath)
			### Change made
			# if rewardi >= 0:
				# print(self.gridgraph.route)
				# solution_combo.append(self.gridgraph.route)

		# solution = solution_combo[-twoPinNum:]	
		score = self.gridgraph.best_reward	# 从gridgraph对象中获取最佳奖励best_reward并存储
		solution = self.gridgraph.best_route[-twoPinNum:]  # 提取与最佳路由相关的最后twoPinNum数量的布线解决方案并存储
		
		solutionDRL = []  # 为每个netSort中的网络初始化一个空列表solutionDRL

		for i in range(len(netSort)):
			solutionDRL.append([])

		print('twoPinNum',twoPinNum)
		print('solution',solution)

        # 组织并分配路由解决方案到不同的网络中。
		if self.gridgraph.posTwoPinNum  == twoPinNum:  # 当gridgraph对象中记录的两终端网络数posTwoPinNum等于twoPinNum时
			dumpPointer = 0
			for i in range(len(netSort)):  # 遍历netSort数组（代表网络排序）
				netToDump = netSort[i]
				for j in range(twoPinNumEachNet[netToDump]):  # 遍历twoPinNumEachNet（每个网络的两终端数）
					# for k in range(len(solution[dumpPointer])):
					solutionDRL[netToDump].append(solution[dumpPointer])  # 将最佳布线解决方案solution分配到对应的网络中
					dumpPointer = dumpPointer + 1  # 追踪solution数组中当前考虑的元素。
		# print('best reward: ', score)
		# print('solutionDRL: ',solutionDRL,'\n')
		else:
			solutionDRL = solution  # 如果不满足条件，则直接将solution赋值给solutionDRL

		## Generte solution

		# print ('solution_combo: ',solution_combo)


		#
		# print(test_reward_log)
		# train_episode = np.arange(self.max_episodes)
		# np.savez('../data/training_log.npz', test_episode=test_episode, test_reward_log=test_reward_log,
		# 		 reward_log=reward_log, train_episode=train_episode)

		self.sess.close()  # 关闭TensorFlow会话
		tf.reset_default_graph()  # 重置默认的TensorFlow图

		return solutionDRL,reward_plot_combo,reward_plot_combo_pure,solution,self.gridgraph.posTwoPinNum
	   # 返回：solutionDRL（DRL算法找到的最终解决方案）、reward_plot_combo（奖励数据集合）、reward_plot_combo_pure（纯净奖励数据集合）、
	# solution（最佳路由解决方案）和self.gridgraph.posTwoPinNum（处理的两终端问题数量）。



	def test(self, model_file=None, no=20, stat=False):
		# Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
		# Here you need to interact with the environment, irrespective of whether you are using a memory. 
		
		# uncomment this line below for videos
		# self.env = gym.wrappers.Monitor(self.env, "recordings", video_callable=lambda episode_id: True)
		if model_file is not None:  # 如果提供了模型文件
			self.saver.restore(self.sess, model_file)  # 加载这个模型
		reward_list = []
		cum_reward = 0.0
		for episode in np.arange(no):  # 对环境执行一定数量（no）的情节（episodes）
			episode_reward = 0.0
			state = self.gridgraph.reset()
			is_terminal = False
			while not is_terminal:
				observation = self.gridgraph.state2obsv()
				q_values = self.sess.run(self.qNetwork.output, feed_dict={self.qNetwork.input: observation})
				action = self.greedy_policy(q_values)
				nextstate, reward, is_terminal, debug = self.gridgraph.step(action)
				state = nextstate
				episode_reward = episode_reward+reward
				cum_reward = cum_reward+reward
			reward_list.append(episode_reward)
		if stat:
			return cum_reward, reward_list
		else:
			return cum_reward
        # 在每个情节中，不断地获取当前状态的观察值，使用贪婪策略（greedy_policy）基于Q网络（qNetwork）的输出选择动作，并执行动作来获取奖励，直到情节终止。这一过程中累积的奖励用于评估模型的性能。

	def burn_in_memory(self):
		# Initialize your replay memory with a burn_in number of episodes / transitions.
		print('Start burn in...')
		state = self.gridgraph.reset()
		for i in np.arange(self.replay.burn_in):
			if i % 2000 == 0:
				print('burn in {} samples'.format(i))
			observation = self.gridgraph.state2obsv()
			action = self.gridgraph.sample()
			nextstate, reward, is_terminal, debug = self.gridgraph.step(action)
			observation_next = self.gridgraph.state2obsv()
			self.replay.append([observation, action, reward, observation_next, is_terminal])
			if is_terminal:
				# print(self.gridgraph.current_step)
				state = self.gridgraph.reset()
			else:
				state = nextstate
		self.replay.is_burn_in = True
		print('Burn in finished.')
    # 为经验回放缓存进行预填充，这个过程通常称为“烧录”（burn-in）。在正式开始训练之前，通过与环境进行交互来收集一定数量
	# （由self.replay.burn_in指定）的初步经验数据。对于每一次交互，它随机选择一个动作（通过self.gridgraph.sample()），
	# 执行该动作并观察结果（下一个状态、奖励、是否结束），然后将这些数据作为一个转换（observation, action, reward, next observation, is_terminal）
	# 保存到回放缓存中。如果达到终止状态，则重置环境。

    # 使用特定的搜索算法预填充经验回放缓冲区
	def burn_in_memory_search(self,observationCombo,actionCombo,rewardCombo,
        observation_nextCombo,is_terminalCombo): # Burn-in with search
		print('Start burn in with search algorithm...')
		for i in range(len(observationCombo)):
			observation = observationCombo[i]
			action = actionCombo[i]
			reward = rewardCombo[i]
			observation_next = observation_nextCombo[i]
			is_terminal = is_terminalCombo[i]

			
			self.replay.append([observation, action, reward, observation_next, is_terminal])

		self.replay.is_burn_in = True
		print('Burn in with search algorithm finished.')

def parse_arguments():
	parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')   # 定义接受的参数和它们的类型
	parser.add_argument('--env',dest='env',type=str)     # 环境名称
	parser.add_argument('--render',dest='render',type=int,default=0)    # 是否渲染环境，类型为整数，默认为0（不渲染）
	parser.add_argument('--train',dest='train',type=int,default=1)      # 是否进行训练，类型为整数，默认为1（进行训练）
	parser.add_argument('--test',dest='test',type=int,default=0)        # 是否进行测试，类型为整数，默认为0（不进行测试）
	parser.add_argument('--lookahead',dest='lookahead',type=int,default=0)     # 是否启用前瞻，类型为整数，默认为0（不启用）
	parser.add_argument('--test_final',dest='test_final',type=int,default=0)     # 是否进行最终测试，类型为整数，默认为0（不进行最终测试）
	parser.add_argument('--model_no',dest='model_file_no',type=str)     # 模型编号，类型为字符串
	return parser.parse_args()     # 解析这些参数，并返回解析后的参数对象


def main(args):

	args = parse_arguments()
	environment_name = args.env

	# Setting the session to allow growth, so it doesn't allocate all GPU memory. 
	gpu_ops = tf.GPUOptions(allow_growth=True)
	config = tf.ConfigProto(gpu_options=gpu_ops)
	sess = tf.Session(config=config)

	# Setting this as the default tensorflow session. 
	keras.backend.tensorflow_backend.set_session(sess)

	# You want to create an instance of the DQN_Agent class here, and then train / test it. 
	model_path = '../model/'
	data_path = '../data/'
	if not os.path.exists(model_path):
		os.makedirs(model_path)
	if not os.path.exists(data_path):
		os.makedirs(data_path)
	agent = DQN_Agent(environment_name, sess, render=args.render)
	if args.train == 1:
		agent.train()
	if args.test == 1:
		print(agent.test(model_file="../model/model_{}.ckpt".format(args.model_file_no))/20.0)
	sess.close()


if __name__ == '__main__':
	main(sys.argv)

