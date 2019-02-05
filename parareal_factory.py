"""
    Copyright (c) 2019 Olga Mula
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.integrate import solve_ivp
import time
import warnings

from ode import VDP, Brusselator, Oregonator

class Propagator():

	def __init__(self, ode, u0, t_interval, integrator='BDF', tol=1.e-12):
		self.ode = ode
		self.u0 = u0
		self.dim = u0.shape[0]
		[self.ti, self.tf] = t_interval
		self.integrator = integrator
		self.tol = tol

		start = time.time()
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore")
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
			raise ValueError('Dimension of problem is {}. Not supported for visualization.'.format(self.dim))

	def compute_cost(self, type='sequential'):
		return self.cost

class Piecewise_Propagator():

	def __init__(self, ode, u0, t_span, u_span, integrator='BDF', tol=1.e-12):
		self.ode = ode
		self.u0 = u0
		self.dim = u0.shape[0]
		self.integrator = integrator
		self.tol = tol

		self.t_span = t_span
		self.interval_span = [[ti, tf] for (ti,tf) in zip(t_span[:-1],
		                                                  t_span[1:])]
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
				idx_test = [int(t>=ti and t<tf) for (ti,tf)
							in self.interval_span]
				idx = np.argmax(idx_test)
				return self.propagator_span[idx].eval(t)
		else:
			return np.array([self.eval(ti) for ti in t.tolist()]).T

	def compute_cost(self, type='sequential'):
		cl = [prop.cost for prop in self.propagator_span]

		if type=='sequential':
			aggregate = sum
		else:
			aggregate = max

		keys = self.propagator_span[0].cost.keys()
		cost = {key: aggregate(c[key] for c in cl) for key in keys}

		self.cost = cost
		return cost

	def plot(self):
		color_list = cm.tab20c(np.linspace(0, 1, len(self.t_span-1)))
		for i, prop in enumerate(self.propagator_span):
			t_vec = prop.ivp.t
			u_vec = self.eval(t_vec)
			plt.plot(u_vec[0,:], u_vec[1,:], color=color_list[i])

class Parareal_Algorithm():
	def __init__(self, ode, u0, t_interval, N, integrator_g='RK45', tol_g = 1.e-1, integrator_f='RK45', tol_f=1.e-12):
		self.ode   = ode
		self.u0    = u0
		[self.ti, self.tf] = t_interval
		self.N = N # Number of processors

		self.integrator_g = integrator_g
		self.tol_g = tol_g

		self.integrator_f = integrator_f
		self.tol_f = tol_f

		self.exact = Propagator(self.ode, self.u0, t_interval, integrator=self.integrator_f, tol=1.e-12)

		self.tl = None
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

	def run(self, eps=1.e-6, kmax=5):

		print('Running Adaptive Parareal Algorithm.')
		print('====================================')
		print('Number of processors = '+str(self.N))
		print('Target accuracy = '+str(eps))
		print(self.tol_g, self.tol_f)


		# List of coarse, fine, parareal prop for each parareal iter k.
		# List of t_span
		gl = list()
		fl = list()
		pl = list()
		tl = list()

		for k in range(kmax):

			print('Iteration k='+str(k))

			if k == 0:
				# First guess for macro-intervals
				t_span = np.linspace(self.ti, self.tf, num=self.N+1, endpoint=True)
				tl.append(t_span)

				# Sequential propagation of coarse solver
				u_span_g = [ self.u0 ]
				for i, t_interval in enumerate(zip(t_span[:-1], t_span[1:])):
					gi = Propagator(self.ode, u_span_g[-1], t_interval, integrator=self.integrator_g, tol=self.tol_g)
					u_span_g.append( gi.eval(t_interval[-1]) )
				u_span_g = np.asarray(u_span_g).T
				g0 = Piecewise_Propagator(self.ode, self.u0, tl[-1], u_span_g, integrator=self.integrator_g, tol=self.tol_g)

				gl.append(g0)
				pl.append(g0)

			else:
				# Fine propagator (k-1)
				self.tol_f = self.tol_g/ 10**(k+1) # TODO: Change this
				fkprev = Piecewise_Propagator(self.ode, self.u0, tl[-1], pl[-1].eval(tl[-1]), integrator=self.integrator_f, tol=self.tol_f)
				fl.append(fkprev)

				# Update t_span with last fine propagation
				t_span = np.linspace(self.ti, self.tf, num=self.N+1, endpoint=True)
				# t_span = self.balance_tasks(fl[-1])
				tl.append(t_span)

				# Parareal solution
				pk_u_span = [ self.u0 ]
				for i, t_interval in enumerate(zip(t_span[:-1], t_span[1:])):
					gi = Propagator(self.ode, pk_u_span[-1], t_interval, integrator=self.integrator_g, tol=self.tol_g)
					f = fkprev.propagator_span[i].eval(t_interval[-1])
					g = gl[-1].propagator_span[i].eval(t_interval[-1])
					pk_u_span.append( gi.eval(t_interval[-1]) + f - g )

				pk_u_span = np.asarray(pk_u_span).T
				gk = Piecewise_Propagator(self.ode, self.u0, tl[-1], pk_u_span, integrator=self.integrator_g, tol=self.tol_g)
				gl.append(gk)

				# pk = Piecewise_Propagator(self.ode, self.u0, tl[-1], pk_u_span, integrator=self.integrator_f, tol=self.tol_f)
				pk = Piecewise_Propagator(self.ode, self.u0, tl[-1], pk_u_span, integrator=self.integrator_f, tol=1.e-10)
				pl.append(pk)

		print('End parareal iterations.\n')

		self.tl = tl
		self.pl = pl
		self.fl = fl
		self.gl = gl

		return pl, fl, gl

	def balance_tasks(self, parareal_sol):
		tl = list()
		for prop in parareal_sol.propagator_span:
			tl.extend(prop.ivp.t[:-1].tolist())
		tl.append(self.tf)

		which_idxs = lambda m, n: np.rint(np.linspace(1,n, min(m,n))-1) \
		                            .astype(int)
		t_span = np.array(tl)[which_idxs(self.N+1,len(tl))]
		return t_span

	def compute_cost(self):
		keys = self.gl[0].propagator_span[0].cost.keys()
		# Add cost of all coarse propagations
		cgl = [g.compute_cost(type='sequential') for g in self.gl]
		cost_g = {key: sum(c[key] for c in cgl) for key in keys}

		# Add cost of fine propagators
		cfl = [f.compute_cost(type='parallel') for f in self.fl]
		cost_f = {key: sum(c[key] for c in cfl) for key in keys}

		# Total cost of parareal algorithm
		cl = [cost_g, cost_f]
		cost_parareal = {key: sum(c[key] for c in cl) for key in keys}

		# Cost exact solver
		cost_exact = self.exact.cost

		return cost_g, cost_f, cost_parareal, cost_exact

	def compute_error(self):

		err = list()
		t = self.exact.ivp.t

		for k, pk in enumerate(self.pl):
			# t = self.tl[k]
			u_pk = pk.eval(t)
			u_exact =  self.exact.eval(t)
			err.append(np.linalg.norm(u_pk - u_exact, axis=0))

		return err





