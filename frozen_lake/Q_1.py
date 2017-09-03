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
	
	def run(self, agent, sess, m):
		s = self.env.reset()
		#print ('s',s)
		r_all = 0

		while True:
			self.env.render()
			a = agent.act(s, sess, m)
			s_, r, done, info = self.env.step(a)

			if done:
				s_ = None
				self.env.render()
			
			agent.add_memory( (s, a, r, s_) )
			agent.find_batch_to_train(sess,m)
			
			s = s_
			r_all += r
			print("Total reward:", r_all)

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

		#self.model_train   = model(n_state, n_action, is_training = True)
		#self.model_predict = model(n_state, n_action, is_training = False)
		self.memory = memory(MEMORY_CAPACITY)



	def act(self, state, sess, m):
		if random.random() < self.epsilon:
			return random.randint(0, self.n_action-1)
		else:
			#print np.identity(self.n_state)[state:state+1]
			Q_ = sess.run(m.Qout, feed_dict = {m.input_s:np.identity(self.n_state)[state:state+1]})
			return np.argmax(Q_)

	def add_memory(self, sample):	# in (s, a, r, s_) format
		self.memory.add(sample) 

		# slowly decrease Epsilon based on our eperience
		self.steps += 1
		self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)
		#print('epsilon',self.epsilon)

	def find_batch_to_train(self, sess, m):
		batch = self.memory.sample(BATCH_SIZE)
		batch_len = len(batch)
		#print ('batchlen',batch_len)

		no_state = np.zeros(self.n_state)
		'''
		states = np.array([ o[0] for o in batch ])
		states_ = np.array([ (no_state if o[3] is None else o[3]) for o in batch ])
		p = sess.run(m.Qout, feed_dict = {m.input_s:np.identity(self.n_state)[states_:states_+1]})
		p_ = sess.run(m.Qout, feed_dict = {m.input_s:np.identity(self.n_state)[states_:states_+1]})

		'''
		p  = []
		p_ = []
		for o in batch:
			state  = o[0]
			state_ = no_state if o[3] is None else o[3]
			pp = sess.run(m.Qout, feed_dict = {m.input_s:np.identity(self.n_state)[state:state+1]})
			pp_ = no_state if o[3] is None else sess.run(m.Qout, feed_dict = {m.input_s:np.identity(self.n_state)[state_:state_+1]})
			p.append(pp)
			p_.append(pp_)

		#print max(p_[0][0]) #(1,n_action)
		
		x = np.zeros((batch_len, self.n_state))
		y = np.zeros((batch_len, self.n_action))
        	
		for i in range(batch_len):
			o = batch[i]
			#print ('o',o)
			s = o[0]; a = o[1]; r = o[2]; s_ = o[3]
            
			t = p[i][0]
			if s_ is None:
				t[a] = r
			else:
				t[a] = r + GAMMA * max(p_[i][0])

			x[i] = np.identity(self.n_state)[s:s+1]
			y[i] = t

		#----train(x, y, sess)
		#print('x,y',x,y)
		sess.run(m.updateModel,feed_dict = {m.input_s:x, m.nextQ:y})


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
	def __init__(self, n_state, n_action, sess):

		self.n_state = n_state
		self.n_action = n_action

		self.input_s = tf.placeholder(shape=[None,n_state],dtype=tf.float32)
		self.w1 = tf.Variable(tf.random_uniform([n_state,n_action],0,0.01))
		self.Qout = tf.matmul(self.input_s,self.w1)
		#self.b1 = tf.Variable(tf.random_uniform([1,n_action],0,0.01))
		#self.Qo = tf.add(tf.matmul(self.input_s,self.w1), self.b1)
		#self.Qout = tf.nn.softmax(self.Qo)
		self.predict = tf.argmax(self.Qout,1)

		#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
		self.nextQ = tf.placeholder(shape=[None,n_action],dtype=tf.float32)
		self.loss = tf.reduce_sum(tf.square(self.nextQ - self.Qout))

		#if not is_training:
		#	return

		self.trainer = tf.train.GradientDescentOptimizer(learning_rate=0.00025)
		self.updateModel = self.trainer.minimize(self.loss)


		self.init = tf.initialize_all_variables()
		
		#sess.run(init)

		@property
		def init(self):
			return self.init



if __name__ == "__main__":

	problem = 'FrozenLake-v0'
	env 	= environment(problem)

	n_state  = env.env.observation_space.n
	n_action = env.env.action_space.n

	agent_ = agent(n_state, n_action)

	r_all  = 0
	
	with tf.Session() as sess:
		model_ = model(n_state,n_action, sess)
		sess.run(model_.init)
		for i in range(100):
			r_ = env.run(agent_, sess, model_)
			r_all += r_
		print ('ratio = ',r_all/100)













