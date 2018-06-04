import numpy as np

class radviz_optimization:
	def __init__(self, data):
		self.data = data
		self.normalize_data()
		self.d_sums = np.sum(data, axis = 1)
		self.N, self.M = np.shape(data)
		self.thetas = np.random.random(self.M) * 2 * np.pi
		#self.thetas = np.arange(self.M) * 2 * np.pi/self.M
		self.update_anchors()

	def partial(self, k):
		''' Calculates partial derivative of the centroid objective w.r.t. the kth parameter. '''
		a_s = self.data[:,k] / self.d_sums
		c_s = (self.data_cos_scaled_sum - self.data_cos_scaled[:,k]) / self.d_sums
		s_s = (self.data_sin_scaled_sum - self.data_sin_scaled[:,k]) / self.d_sums
		Ga = a_s - np.mean(a_s)
		Gc = c_s - np.mean(c_s)
		Gs = s_s - np.mean(s_s)
		partial_derivative = 0
		bottom = np.sqrt((((Ga * self.cos_thetas[k]) + Gc) ** 2) + (((Ga * self.sin_thetas[k]) + Gs) ** 2))
		top = (Ga * ((Gs * self.cos_thetas[k]) - (Gc * self.sin_thetas[k])))
		partial_derivative = np.sum(top / bottom)
		if partial_derivative == np.nan:
			print("Partial derivative undefined at k =", k)
			return np.random.random() * 1e-5
		return partial_derivative

	def gradient(self):
		''' Calculates gradient of centroid objective w.r.t. the parameters.
			The first parameter is fixed at 0, so its partial derivative is always 0. '''
		self.data_sin_scaled = self.data * self.sin_thetas
		self.data_sin_scaled_sum = np.sum(self.data_sin_scaled[:,1:], axis=1)
		self.data_cos_scaled = self.data * self.cos_thetas
		self.data_cos_scaled_sum = np.sum(self.data_cos_scaled, axis=1)
		return np.concatenate([[0.0], [self.partial(k) for k in range(1, self.M)]])

	def rectify(self):
		''' This rectifies the parameters, so that there is only one unique reflection. '''
		self.thetas = np.mod(self.thetas, 2 * np.pi)
		for t in self.thetas:
			if t == 0: continue
			elif t > np.pi:
				self.thetas = np.mod(- self.thetas, 2 * np.pi)
			else: return
		return

	def optimize(self, max_iterations = 200, momentum = 0.9, step_size = 0.01, halting_threshold = 1e-5, printing=False):
		''' Performs gradient ascent. '''
		step_size *= self.M
		self.thetas[0] = 0
		self.update_anchors()
		delta = np.zeros(self.M)
		moment = np.zeros(self.M)
		for iteration in range(max_iterations):
			moment *= momentum
			moment += (1 - momentum) * self.gradient()
			delta = moment * step_size
			self.thetas += delta
			self.update_anchors()
			if printing:
				print("\n\nIteration: ", iteration, "\nScore: ", self.centroid_score(), "\nGradient Magnitude: ", np.linalg.norm(delta))
			if(np.linalg.norm(delta) < halting_threshold):
				break
		self.rectify()
		self.update_anchors()
		return self.thetas

	def update_anchors(self):
		''' Calculates the sin and cos of the parameters. '''
		self.sin_thetas = np.sin(self.thetas)
		self.cos_thetas = np.cos(self.thetas)
		return

	def normalize_data(self):
		''' Normalizes the data such that, for any column, min(column) = 0, max(column) = 1. '''
		mins = np.min(self.data, axis=0)
		maxs = np.max(self.data, axis=0)
		self.data = (self.data - mins) / [1 if x==0 else x for x in (maxs - mins)]
		return

	def centroid_score(self):
		''' Returns the score as defined by the centroid objective. '''
		view = self.get_view()
		return np.sum([np.linalg.norm(r) for r in (view - np.mean(view, axis=0))])

	def get_thetas(self):
		''' Returns the parameters. '''
		return self.thetas

	def get_anchors(self):
		''' Returns the [x,y] coordinates of the anchors.  This is an Mx2 matrix. '''
		return np.transpose([self.cos_thetas, self.sin_thetas])

	def get_view(self):
		''' Returns the resulting [x,y] coordinates of the records. This is a Nx2 matrix. '''
		view = []
		for i in range(self.N):
			x = np.dot(self.cos_thetas, self.data[i]) / self.d_sums[i]
			y = np.dot(self.sin_thetas, self.data[i]) / self.d_sums[i]
			view.append([x,y])
		return np.vstack(view)