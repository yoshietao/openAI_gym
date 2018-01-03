import random, math, gym
from gym import wrappers
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class environment:

	def __init__(self,problem):
		self.problem = problem
		self.env = gym.make(problem)
		self.env = wrappers.Monitor(self.env, '/tmp/Q_2',force=True)

	def run(self, agent, sess, m1, m2, ith):
		s = self.env.reset()
		#self.env.render()
		r_all = 0

		while True:

			#for i in range(16):
			#	print np.argmax(sess.run(m.A2, feed_dict = {m.input_s:np.identity(16)[i:i+1]}))
			
			a_1 = agent.act(s, sess, m1)
			a_2 = agent.act(s, sess, m2)
			
			x = random.randint(0,1)

			if x:
				s_, r, done, info = self.env.step(a_1)
				if done:
					s_ = None
				agent.add_memory( (s, a_1, r, s_, x) )
			else:
				s_, r, done, info = self.env.step(a_2)
				if done:
					s_ = None
				agent.add_memory( (s, a_2, r, s_, x) )
			
			agent.find_batch_to_train(sess,m1, m2)
			
			s = s_
			r_all += r
			#print("Total reward:", r_all)

			if done:
				return r_all

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
		self.memory = memory(MEMORY_CAPACITY)



	def act(self, state, sess, m):
		if random.random() < self.epsilon:
			return random.randint(0, self.n_action-1)
		else:
			#print np.identity(self.n_state)[state:state+1]
			Q_ = sess.run(m.A2, feed_dict = {m.input_s:np.identity(self.n_state)[state:state+1]})
			return np.argmax(Q_[0])

	def add_memory(self, sample):	# in (s, a, r, s_) format
		self.memory.add(sample) 

		# slowly decrease Epsilon based on our eperience
		self.steps = self.steps+ 1
		self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)
		#print('epsilon',self.epsilon)

	def assign_epsilon(self,epsi):
		self.epsilon = epsi

	def find_batch_to_train(self, sess, m1, m2):
		#batch = self.memory.sample(BATCH_SIZE)
		batches = self.memory.batches(BATCH_SIZE)
		batches_len = len(batches)
		batch_len = len(batches[0])
		#print ('batchlen',batch_len)

		no_state = np.zeros(self.n_state)

		x_1 = []
		x_2 = []
		y_1 = []
		y_2 = []

		for j in range(batches_len):
			batch = batches[j]
			for i in range(batch_len):
				o = batch[i]
				state  = o[0]
				state_ = no_state if o[3] is None else o[3]
				pp = sess.run(m1.A2, feed_dict = {m1.input_s:np.identity(self.n_state)[state:state+1]}) if o[4] is True else sess.run(m2.A2, feed_dict = {m2.input_s:np.identity(self.n_state)[state:state+1]})
				try:
					if o[4]:
						pp_ = sess.run(m2.A2, feed_dict = {m2.input_s:np.identity(self.n_state)[state_:state_+1]})
					else:
						pp_ = sess.run(m1.A2, feed_dict = {m1.input_s:np.identity(self.n_state)[state_:state_+1]})
				except:
					True

				s = o[0]; a = o[1]; r = o[2]; s_ = o[3]
				t = pp[0]
				pp0a=pp[0][a]
				if s_ is None:
					t[a] = r
				else:
					t[a] = r + GAMMA * max(pp_[0])

				sams_count = 0
				if abs(pp0a-t[a])<0.01:
					print pp0a,t[a]
					True
				else:
					sams_count+=1
					if o[4]:
						x_1.append(np.identity(self.n_state)[s:s+1])
						y_1.append(t)
					else:
						x_2.append(np.identity(self.n_state)[s:s+1])
						y_2.append(t)

		#print np.array(y_1),np.array(y_2)
		print sams_count
		sess.run(m1.updateModel,feed_dict = {m1.input_s:np.array(x_1).reshape(-1,16), m1.nextQ:np.array(y_1).reshape(-1,4)})
		sess.run(m2.updateModel,feed_dict = {m2.input_s:np.array(x_2).reshape(-1,16), m2.nextQ:np.array(y_2).reshape(-1,4)})	


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
	def batches(self,n):
		batches_ = []
		n = min(n, len(self.samples))
		batches_.append(self.samples[-n:])
		return batches_


class model:
	def __init__(self, n_state, n_action):

		self.n_state = n_state
		self.n_action = n_action

		self.input_s = tf.placeholder(shape=[None,n_state],dtype=tf.float32)
		self.w1 = tf.Variable(tf.random_normal([n_state,16],0,0.2))
		#self.Qout = tf.matmul(self.input_s,self.w1)
		self.b1 = tf.Variable(tf.random_normal([1,16],0,0.2))
		self.h1 = tf.add(tf.matmul(self.input_s,self.w1), self.b1)
		self.A1 = tf.nn.relu(self.h1)
		
		self.w2 = tf.Variable(tf.random_normal([16,n_action],0,0.2))
		self.b2 = tf.Variable(tf.random_uniform([1,n_action],0,0.2))
		self.h2 = tf.add(tf.matmul(self.A1,self.w2),self.b2)
		self.A2 = self.h2
		#self.h2 = tf.matmul(self.A1,self.w2)
		
		#self.Qout = tf.nn.softmax(self.Qo)

		#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
		self.nextQ = tf.placeholder(shape=[None,n_action],dtype=tf.float32)
		self.loss = tf.reduce_sum(tf.square(self.nextQ - self.A2))
		#self.loss = tf.losses.sigmoid_cross_entropy(multi_class_labels = nextQ, logits = self.A2)
		#self.loss = tf.losses.huber_loss(labels = self.nextQ, predictions = self.A2, delta = )

		#if not is_training:
		#	return

		self.trainer = tf.train.GradientDescentOptimizer(learning_rate=0.0025)
		#GradientDescentOptimizer
		self.updateModel = self.trainer.minimize(self.loss)


		self.init = tf.initialize_all_variables()

		@property
		def init(self):
			return self.init


if __name__ == "__main__":

	problem = 'FrozenLake-v0'
	env 	= environment(problem)

	n_state  = env.env.observation_space.n
	n_action = env.env.action_space.n
	episode  = 1000

	agent_ = agent(n_state, n_action)

	r_all  = 0
	R = []
	plt.axis([0,episode,0,0.5])
	plt.ion()

	with tf.Session() as sess:
		model_1 = model(n_state,n_action)
		model_2 = model(n_state,n_action)
		
		sess.run(model_1.init)
		sess.run(model_2.init)
		
		for i in range(episode):
			r_ = env.run(agent_, sess, model_1,model_2,i)

			r_all += r_
			R.append(r_all/(i+1))
			plt.scatter(i,r_all/(i+1))
			plt.pause(0.05)

		for i in range(n_state):
			print np.argmax(sess.run(model_1.A2, feed_dict = {model_1.input_s:np.identity(n_state)[i:i+1]}))
			
		print ('ratio = ',r_all/episode)
		while True:
			plt.pause(0.05)
		#plt.plot(R)
		#plt.show()













