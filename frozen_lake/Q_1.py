import random, math, gym
from gym import wrappers
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class environment:

	def __init__(self,problem):
		self.problem = problem
		self.env = gym.make(problem)
		#env = wrappers.Monitor(env, '/tmp/cartpole-experiment-1',force=True)
	
	def run(self, agent):
		s = self.env.reset()
		r_all = 0

		while True:
			self.env.render()
			a = agent.act(s)
			s_, r, done, info = self.env.step(a)

			if done:
				s_ = None
			
			agent.add_memory( (s, a, r, s_) )
			agent.find_batch_to_train()
			
			s = s_
			r_all += r
			
			if done:
				break
			print("Total reward:", r_all)

#--------------------------------------------agent--------------------------------------------
MEMORY_CAPACITY = 100000
BATCH_SIZE = 64

GAMMA = 0.99

MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.001      # speed of decay

class agent:
	steps = 0
	epsilon = MAX_EPSILON

	def __init__(self, n_state, n_action):
		self.n_state  = n_state
		self.n_action = n_action

		self.model_train   = model(n_state, n_action, is_training = True)
		self.model_predict = model(n_state, n_action, is_training = False)
		self.memory = memory(MEMORY_CAPACITY)



	def act(self, state):
		if random.random() < self.epsilon:
			return random.randint(0, self.n_action-1)
		else:
			return np.argmax(self.model_predict.predict(state))

	def add_memory(self, sample):	# in (s, a, r, s_) format
		self.memory.add(sample) 

		# slowly decrease Epsilon based on our eperience
		self.steps += 1
		self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

	def find_batch_to_train(self):
		batch = self.memory.sample(BATCH_SIZE)
		batch_len = len(batch)

		no_state = np.zeros(self.n_state)

		states = np.array([ o[0] for o in batch ])
		states_ = np.array([ (no_state if o[3] is None else o[3]) for o in batch ])

		p  = self.model_predict.predict(states)
		p_ = self.model_predict.predict(states_)

		x = np.zeros((batch_len, self.n_state))
		y = np.zeros((batch_len, self.n_action))
        	
		for i in range(batch_len):
			o = batch[i]
			s = o[0]; a = o[1]; r = o[2]; s_ = o[3]
            
			t = p[i]
			if s_ is None:
				t[a] = r
			else:
				t[a] = r + GAMMA * np.amax(p_[i])

			x[i] = s
			y[i] = t

		self.model_train.train(x, y)


class memory:
	samples = []

	def __init__(self, capacity):
		self.capacity = capacity

	def add(self, sample):
		self.samples.append(sample)        

		if len(self.samples) > self.capacity:
			self.samples.pop(0)

	def sample(self, n):
		n = min(n, len(self.samples))
		return random.sample(self.samples, n)


class model:
	def __init__(self, n_state, n_action, is_training):
		self.n_state = n_state
		self.n_action = n_action

		#createmodel

		init = tf.global_variables_initializer()

		input_s = tf.placeholder(shape=[1,n_state],dtype=tf.float32)
		w1 = tf.Variable(tf.random_normal([n_state,n_action],0,0.01))
		b1 = tf.Variable(tf.random_normal([1,n_action],0,0.01))
		Qout = tf.add(tf.matmul(input_s,w1), b1)
		predict = tf.argmax(Qout,1)

		#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
		nextQ = tf.placeholder(shape=[1,n_action],dtype=tf.float32)
		loss = tf.reduce_sum(tf.square(nextQ - Qout))

		if not is_training:
			return

		trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
		updateModel = trainer.minimize(loss)

	def train(self,x,y, epoch=1):
		return
	def predict(self,state):
			with tf.Session() as sess:
				sess.run(init)
				#-----------------------------------------------------------here






if __name__ == "__main__":

	problem = 'FrozenLake-v0'
	env 	= environment(problem)

	n_state  = env.env.observation_space.n
	n_action = env.env.action_space.n

	agent_ = agent(n_state, n_action)

	try:
		while True:
			env.run(agent_)
	finally:
		print('end')