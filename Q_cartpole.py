import random, math, gym
from gym import wrappers
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------------------------
counter = {
	-1:-1,
	0:2,
	1:3,
	2:0,
	3:1
}

class environment:

	def __init__(self,problem):
		self.problem = problem
		self.env = gym.make(problem)
		#self.env = wrappers.Monitor(self.env, '/tmp/Q_2',force=True)
	
	def run(self, agent, sess, m):
		s = self.env.reset()
		#print s.shape
		self.env.render()
		#a_prev = -1
		#print ('s',s)
		r_all = 0

		while True:
			a = agent.act(s.reshape(-1,4), sess, m)#, a_prev)
			s_, r, done, info = self.env.step(a)

			if done:
				#print 'done'
				s_ = None
			
			agent.add_memory( (s.reshape(-1,4), a, r, s_ if s_ is None else s_.reshape(-1,4)) )
			agent.find_batch_to_train(sess,m)
			
			#a_prev = a
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

		#self.model_train   = model(n_state, n_action, is_training = True)
		#self.model_predict = model(n_state, n_action, is_training = False)
		self.memory = memory(MEMORY_CAPACITY)



	def act(self, state, sess, m):#, a_prev):
		if random.random() < self.epsilon:
			#print('-------random move------')
			return random.randint(0, self.n_action-1)
		else:
			#print np.identity(self.n_state)[state:state+1]
			Q_ = sess.run(m.A2, feed_dict = {m.input_s:state})
			#print np.argmax(Q_[0]),a_prev
			#if np.argmax(Q_[0]) == counter[a_prev]:
			#	Q_[0][np.argmax(Q_)] = 0
			return np.argmax(Q_[0])

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

		no_state = np.zeros(self.n_state).reshape(-1,self.n_state)
		'''
		states = np.array([ o[0] for o in batch ])
		states_ = np.array([ (no_state if o[3] is None else o[3]) for o in batch ])
		p = sess.run(m.Qout, feed_dict = {m.input_s:np.identity(self.n_state)[states_:states_+1]})
		p_ = sess.run(m.Qout, feed_dict = {m.input_s:np.identity(self.n_state)[states_:states_+1]})

		'''
		p  = []
		p_ = []
		for o in batch:
			#print o
			state  = o[0]
			state_ = no_state if o[3] is None else o[3]
			pp = sess.run(m.A2, feed_dict = {m.input_s:state})
			pp_ = sess.run(m.A2, feed_dict = {m.input_s:state_})
			p.append(pp)
			p_.append(pp_)

		#print max(p_[0][0]) #(1,n_action)
		
		x = np.zeros((batch_len, self.n_state))
		y = np.zeros((batch_len, self.n_action))
        	
		for i in range(batch_len):
			o = batch[i]
			#print o
			s = o[0]; a = o[1]; r = o[2]; s_ = o[3]
            
			t = p[i][0]
			if s_ is None:
				t[a] = r
			else:
				t[a] = r + GAMMA * max(p_[i][0])

			x[i] = s
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
		self.w1 = tf.Variable(tf.random_normal([n_state,16],0,0.01))
		#self.Qout = tf.matmul(self.input_s,self.w1)
		self.b1 = tf.Variable(tf.random_normal([1,16],0,0.01))
		self.h1 = tf.add(tf.matmul(self.input_s,self.w1), self.b1)
		self.A1 = tf.nn.relu(self.h1)
		
		self.w2 = tf.Variable(tf.random_normal([16,n_action],0,0.01))
		self.b2 = tf.Variable(tf.random_uniform([1,n_action],0,0.01))
		self.h2 = tf.add(tf.matmul(self.A1,self.w2),self.b2)
		self.A2 = (self.h2)
		#self.h2 = tf.matmul(self.A1,self.w2)
		
		#self.Qout = tf.nn.softmax(self.Qo)

		#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
		self.nextQ = tf.placeholder(shape=[None,n_action],dtype=tf.float32)
		self.loss = tf.reduce_sum(tf.square(self.nextQ - self.A2))

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

	problem = 'CartPole-v0'
	env 	= environment(problem)

	n_state  = env.env.observation_space.shape[0]
	n_action = env.env.action_space.n
	episode  = 300

	agent_ = agent(n_state, n_action)

	r_all  = 0
	R = []
	with tf.Session() as sess:
		model_ = model(n_state,n_action, sess)
		sess.run(model_.init)
		for i in range(episode):
			r_ = env.run(agent_, sess, model_)

			r_all += r_
			R.append(r_all)

		#for i in range(n_state):
		#	print np.argmax(sess.run(model_.A2, feed_dict = {model_.input_s:i}))
			
		print ('ratio = ',r_all/episode)
		plt.plot(R)
		plt.show()













