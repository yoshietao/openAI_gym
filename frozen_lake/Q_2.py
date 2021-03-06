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

NUM_EPOCH = 1

class environment:

	def __init__(self,problem):
		self.problem = problem
		self.env = gym.make(problem)
		self.env = wrappers.Monitor(self.env, '/tmp/Q_2',force=True)

	def is_goal(self, s, s_):
		#if s == 14:
		if s_ is None:
			return True if s is 14 else False
		else:
			return False
	def run(self, agent, sess, m1, m2):
		s = self.env.reset()
		#self.env.render()
		a_prev = -1
		#print ('s',s)
		r_all = 0

		while True:

			#for i in range(16):
			#	print np.argmax(sess.run(m.A2, feed_dict = {m.input_s:np.identity(16)[i:i+1]}))
			
			a_1 = agent.act(s, sess, m1, a_prev)
			a_2 = agent.act(s, sess, m2, a_prev)
			
			x = random.randint(0,1)

			if x:
				s_, r, done, info = self.env.step(a_1)
				if done:
					s_ = None
				#print self.is_goal(s,s_)
				agent.add_memory( (s, a_1, r, s_, x), self.is_goal(s,s_) )
				a_prev = a_1
				#self.env.render()
			else:
				s_, r, done, info = self.env.step(a_2)
				if done:
					s_ = None
				agent.add_memory( (s, a_2, r, s_, x), self.is_goal(s,s_) )
				a_prev = a_2
				#self.env.render()

			
			for i in range(NUM_EPOCH):
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

		#self.model_train   = model(n_state, n_action, is_training = True)
		#self.model_predict = model(n_state, n_action, is_training = False)
		self.memory = memory(MEMORY_CAPACITY)



	def act(self, state, sess, m, a_prev):
		if random.random() < self.epsilon:
			#print('-------random move------')
			return random.randint(0, self.n_action-1)
		else:
			#print np.identity(self.n_state)[state:state+1]
			Q_ = sess.run(m.A2, feed_dict = {m.input_s:np.identity(self.n_state)[state:state+1]})
			#print np.argmax(Q_[0]),a_prev
			#if np.argmax(Q_[0]) == counter[a_prev]:
			#	Q_[0][np.argmax(Q_)] = 0
			return np.argmax(Q_[0])

	def add_memory(self, sample, goal):	# in (s, a, r, s_) format
		self.memory.add(sample) 

		# slowly decrease Epsilon based on our eperience
		#try:
		#	self.steps = self.steps+ 1 if goal is False else self.steps*2
		#except:
		#	True
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
		'''
		states = np.array([ o[0] for o in batch ])
		states_ = np.array([ (no_state if o[3] is None else o[3]) for o in batch ])
		p = sess.run(m.Qout, feed_dict = {m.input_s:np.identity(self.n_state)[states_:states_+1]})
		p_ = sess.run(m.Qout, feed_dict = {m.input_s:np.identity(self.n_state)[states_:states_+1]})

		'''
		#x = np.zeros((batches_len,batch_len, self.n_state))
		#y = np.zeros((batches_len,batch_len, self.n_action))
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

				#pp_ = no_state if o[3] is None else sess.run(m.A2, feed_dict = {m.input_s:np.identity(self.n_state)[state_:state_+1]})

				s = o[0]; a = o[1]; r = o[2]; s_ = o[3]
				t = pp[0]
				if s_ is None:
					t[a] = r
				#	if s == 14:
				#		t[a] = r
				#	else:
				#		t[a] = r-1
				else:
					t[a] = r + GAMMA * max(pp_[0])

				if o[4]:
					x_1.append(np.identity(self.n_state)[s:s+1])
					y_1.append(t)
				else:
					x_2.append(np.identity(self.n_state)[s:s+1])
					y_2.append(t)
				
				#x[j][i] = np.identity(self.n_state)[s:s+1]
				#y[j][i] = t	

		#----train(x, y, sess)
		#print('x,y',x,y)
		#print np.array(y_1),np.array(y_2)
		for j in range(len(y_1)):
			sess.run(m1.updateModel,feed_dict = {m1.input_s:x_1[j].reshape(-1,16), m1.nextQ:y_1[j].reshape(-1,4)})
		for j in range(len(y_2)):
			sess.run(m2.updateModel,feed_dict = {m2.input_s:x_2[j].reshape(-1,16), m2.nextQ:y_2[j].reshape(-1,4)})	


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
		#for i in range(len(self.samples)/n):
		#	batches_.append(self.samples[i*n+len(self.samples)%n:(i+1)*n+len(self.samples)%n])
		#batches_.append(self.samples[-(self.samples%n):])
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
		self.A2 = (self.h2)
		#self.h2 = tf.matmul(self.A1,self.w2)
		
		#self.Qout = tf.nn.softmax(self.Qo)

		#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
		self.nextQ = tf.placeholder(shape=[None,n_action],dtype=tf.float32)
		self.loss = tf.reduce_sum(tf.square(self.nextQ - self.A2))
		#self.loss = tf.losses.huber_loss(labels = self.nextQ, predictions = self.A2, delta = )

		#if not is_training:
		#	return

		self.trainer = tf.train.GradientDescentOptimizer(learning_rate=0.0025)
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
			r_ = env.run(agent_, sess, model_1,model_2)

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













