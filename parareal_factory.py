import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.integrate import solve_ivp
import time

from ode import VDP, Brusselator, Oregonator

class Propagator():

	def __init__(self, ode, u0, t_interval, integrator='BDF', tol=1.e-10):
		self.ode = ode
		self.u0 = u0
		self.dim = u0.shape[0]
		[self.ti, self.tf] = t_interval
		self.integrator = integrator
		self.tol = tol

		start = time.time()
		self.ivp = solve_ivp(ode.f, [self.ti, self.tf], u0, jac = ode.jac, method = integrator, atol=tol, rtol=tol, dense_output=True)
		end = time.time()

		self.cost = {
			'cpu_time': end-start,
			't_steps': len(self.ivp.t.tolist()),
			'n_rhs':   self.ivp.nfev, # number of evaluations of rhs
			'n_jac':   self.ivp.njev, # Number of evaluations of the Jacobian
			'nlu':     self.ivp.nlu,  # Number of lu decompositions
		}

	def eval(self, t):
		return self.ivp.sol(t) 

	def plot(self, color='black'):
		if self.dim ==2:
			t_vec = self.ivp.t
			u_vec = self.eval(t_vec)
			plt.plot(u_vec[0,:], u_vec[1,:], color=color)
		else:
			raise ValueError('Dimension of problem is '+str(self.dim)+'. Not supported for visualization.')

	def compute_cost(self, type='sequential'):
		return self.cost

class Piecewise_Propagator():

	def __init__(self, ode, u0, t_span, u_span, integrator='BDF', tol=1.e-10):
		self.ode = ode
		self.u0 = u0
		self.dim = u0.shape[0]
		self.integrator = integrator
		self.tol = tol

		self.t_span = t_span
		self.interval_span = [[ti, tf] for (ti,tf) in zip(t_span[:-1], t_span[1:])]		
		self.u_span = np.asarray(u_span) # Column j has solution at time t_j
		self.propagator_span = list()

		for i, t_interval in enumerate(self.interval_span):
			u_init = u_span[:,i]
			self.propagator_span.append(Propagator(ode, u_init, t_interval, integrator=integrator, tol=tol))

		self.cost = None

	def eval(self, t):
		if isinstance(t, float):
			if t == self.t_span[-1]:
				return self.propagator_span[-1].eval(t)
			else:
				idx_test = [int(t>=ti and t<tf) for (ti,tf) in self.interval_span]
				idx = np.argmax(idx_test)
				return self.propagator_span[idx].eval(t)
		else:
			return np.array([self.eval(ti) for ti in t.tolist()]).T

	def compute_cost(self, type='sequential'):
		cl = list()
		for prop in self.propagator_span:
			cl.append(prop.cost)

		keys = self.propagator_span[0].cost.keys()
		cost = {key: 0. for key in keys}

		if type=='sequential':
			for key in keys:
				cost[key] = sum(c[key] for c in cl)
		else:
			for key in keys:
				cost[key] = max(c[key] for c in cl)

		self.cost = cost
		return cost

	def plot(self):
		color_list = cm.tab20c(np.linspace(0, 1, len(self.t_span-1)))
		for i, prop in enumerate(self.propagator_span):
			t_vec = prop.ivp.t
			u_vec = self.eval(t_vec)
			plt.plot(u_vec[0,:], u_vec[1,:], color=color_list[i])

class Parareal_Algorithm():
	def __init__(self, ode, u0, t_interval, N, integrator_g='RK23', tol_g = 1.e-2, integrator_f='BDF', tol_f=1.e-2):
		self.ode   = ode
		self.u0    = u0
		[self.ti, self.tf] = t_interval
		self.N = N # Number of processors

		self.integrator_g = integrator_g
		self.tol_g = tol_g

		self.integrator_f = integrator_f
		self.tol_f = tol_f

		self.exact = Propagator(self.ode, self.u0, t_interval, integrator=self.integrator_f, tol=1.e-12)

		self.pl = None
		self.fl = None
		self.gl = None

		self.cost = {
			'cpu_time': 0.,
			't_steps':  0.,
			'n_rhs':    0., # number of evaluations of rhs
			'n_jac':    0., # Number of evaluations of the Jacobian
			'nlu':      0.,  # Number of lu decompositions
		}

	def run(self, eps=1.e-6, kmax=3):

		# List of coarse, fine, parareal prop for each parareal iter k.
		# List of t_span
		gl = list()
		fl = list()
		pl = list()
		tl = list()

		for k in range(kmax):

			print('Iteration k='+str(k))

			if k == 0:
				# Sequential propagation of coarse solver
				g0 = Propagator(self.ode, self.u0, [self.ti, self.tf], integrator=self.integrator_g, tol=self.tol_g)
				gl.append(g0)
				pl.append(g0)

				# First guess for macro-intervals
				t_span = np.linspace(self.ti, self.tf, num=self.N+1, endpoint=True)
				tl.append(t_span)
				
			else:
				# Fine propagator (k-1)
				self.tol_f = 10**(-k-1) # TODO: Change this
				fkprev = Piecewise_Propagator(self.ode, self.u0, tl[-1], pl[-1].eval(tl[-1]), integrator=self.integrator_f, tol=self.tol_f)
				fl.append(fkprev)

				# Update t_span with last fine propagation
				t_span = self.balance_tasks(fl[-1])
				tl.append(t_span)

				# Parareal solution
				pk_u_span = [ self.u0 ]
				for i, t_interval in enumerate(zip(t_span[:-1], t_span[1:])):
					gi = Propagator(self.ode, pk_u_span[-1], t_interval, integrator=self.integrator_g, tol=self.tol_g)
					pk_u_span.append( gi.eval(t_interval[-1]) + fkprev.eval(t_interval[-1]) - gl[-1].eval(t_interval[-1]) )

				pk_u_span = np.asarray(pk_u_span).T
				gk = Piecewise_Propagator(self.ode, self.u0, tl[-1], pk_u_span, integrator=self.integrator_g, tol=self.tol_g)
				gl.append(gk)

				pk = Piecewise_Propagator(self.ode, self.u0, tl[-1], pk_u_span, integrator=self.integrator_f, tol=self.tol_f)
				pl.append(pk)

		self.pl = pl
		self.fl = fl
		self.gl = gl

		return pl, fl, gl

	def balance_tasks(self, parareal_sol):
		tl = list()
		for i, prop in enumerate(parareal_sol.propagator_span):
			tl = tl + prop.ivp.t[:-1].tolist()
		tl = tl + [self.tf]

		which_idxs = lambda m, n: np.rint(np.linspace(1,n, min(m,n))-1).astype(int)
		t_span = np.array(tl)[which_idxs(self.N+1,len(tl))]
		return t_span

	def compute_cost(self):
		# Add cost of all coarse propagations
		cgl = list()
		for g in self.gl:
			cgl.append(g.compute_cost())

		keys = self.gl[0].cost.keys()
		cost_g = {key: 0. for key in keys}
		for key in keys:
			cost_g[key] = sum(c[key] for c in cgl)

		return cost_g

