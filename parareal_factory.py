"""
    Copyright (c) 2019 Olga Mula
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.special import factorial
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
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

class SolverHelper():

	def __init__(self, ode, u0, t_interval, integrator='RK45', integrator_e='LSODA', tol_e=1.e-13, type_norm='linf'):

		self.ode   = ode
		self.u0    = u0
		[self.ti, self.tf] = t_interval
		self.integrator = integrator
		self.integrator_e = integrator_e
		self.tol_e = tol_e
		self.exact = Propagator(self.ode, self.u0, [self.ti, self.tf], integrator=self.integrator_e, tol=tol_e)

		self.type_norm = type_norm
		self.tol_span, self.eps_span, self.cost_span, self.tol_to_eps = \
			self.compute_tol_to_eps(self.integrator, self.type_norm)

	def compute_tol_to_eps(self, integrator, type_norm, n_tol=20):

		if type_norm=='l2':
			aggregate = sum
			self.type_norm = 'l2'
		elif type_norm=='linf':
			aggregate = max
			self.type_norm = 'linf'
		else:
			raise ValueError('Norm '+type_norm+' not supported in tol_to_eps.')

		
		t = self.exact.ivp.t
		u_exact = self.exact.eval(t)
		tol_span = np.logspace(np.log10(self.tol_e), 0., num=n_tol, base=10, endpoint=True)

		eps_span = list()
		cost_span = list()
		for tol in tol_span:
			p = Propagator(self.ode, self.u0, [self.ti, self.tf], integrator=integrator, tol=tol)
			u = p.eval(t)

			cost_span.append(p.compute_cost())
			eps_span.append(aggregate(np.linalg.norm(u-u_exact, axis=0)))

		cs = CubicSpline(tol_span, eps_span)

		return tol_span, eps_span, cost_span, CubicSpline(tol_span, eps_span)




class Parareal_Algorithm():
	def __init__(self, ode, u0, t_interval, N, integrator_g='RK45',
		tol_g = 1.e-1, integrator_f='LSODA', tol_f=1.e-12, tol_e=1.e-20):

		self.ode   = ode
		self.u0    = u0
		[self.ti, self.tf] = t_interval
		self.N = N # Number of processors

		self.integrator_g = integrator_g
		self.integrator_f = integrator_f

		self.sh_g = SolverHelper(ode, u0, t_interval, integrator=integrator_g, integrator_e = integrator_f, tol_e=1.e-13, type_norm='linf')
		self.sh_f = SolverHelper(ode, u0, t_interval, integrator=integrator_f, integrator_e = integrator_f, tol_e=1.e-13, type_norm='linf')

		self.tol_g = tol_g
		self.tol_f = tol_f
		self.tol_e = tol_e

		self.exact = self.sh_f.exact

		self.tl = None
		self.pl = None
		self.fl = None
		self.gl = None

		self.cost = {
			'cpu_time': 0.,
			't_steps':  0.,
			'n_rhs':    0., # number of evaluations of rhs
			'n_jac':    0., # Number of evaluations of the Jacobian
			'nlu':      0., # Number of lu decompositions
		}

	def run(self, eps=1.e-6, kmax=1):

		print('Running Adaptive Parareal Algorithm.')
		print('====================================')
		print('Number of processors = '+str(self.N))
		print('Target accuracy = '+str(eps))


		# List of coarse, fine, parareal prop for each parareal iter k.
		# List of t_span
		gl = list()
		fl = list()
		pl = list()
		tl = list()
		eps_g = self.estimate_eps_g()
		print('eps_g = '+str(eps_g))

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
				nuk = self.estimate_nu_k(pl[-1])
				print('nu_k = '+str(nuk))
				self.tol_f = eps_g/2 **(k+2)/ (nuk * factorial(k+1))
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

	def estimate_nu_k(self, parareal_prop_k):
		t = parareal_prop_k.t_span
		u_k = parareal_prop_k.u_span
		u_exact = self.exact.eval(t)
		return np.max(1 + np.linalg.norm(u_k, axis=0)) / np.max(1 + np.linalg.norm(u_exact, axis=0))

	def estimate_eps_g(self):
		t = self.exact.ivp.t[3:]
		e = self.exact.eval(t)
		g = Propagator(self.ode, self.u0, [self.ti,self.tf], integrator=self.integrator_g, tol=self.tol_g).eval(t)

		return np.max(
			np.linalg.norm(e-g, axis=0)/(t * (1+np.linalg.norm(e, axis=0))))


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

	def plot_eps_to_key(self, folder_name, key='cpu_time'):

		if key not in self.cost.keys():
			raise ValueError('Invalid key in plot_eps_to_cost')

		# Fine solver
		plt.figure()
		plt.style.use('classic')
		cost_cpu = [ c[key] for c in self.sh_g.cost_span ]
		plt.loglog(self.sh_f.eps_span, cost_cpu, label=key)
		plt.gca().yaxis.grid(True)
		plt.xlabel(r'$\varepsilon$', fontsize=20)
		plt.legend()
		plt.savefig(folder_name+'eps_to_'+key+'_f.pdf')

		# Coarse solver
		plt.figure()
		plt.style.use('classic')
		cost_cpu = [ c[key] for c in self.sh_g.cost_span ]
		plt.loglog(self.sh_g.eps_span, cost_cpu, label=key)
		plt.gca().yaxis.grid(True)
		plt.xlabel(r'$\varepsilon$', fontsize=20)
		plt.legend()
		plt.savefig(folder_name+'eps_to_'+key+'_g.pdf')

	def plot_tol_to_eps(self, folder_name):

		plt.figure()
		plt.semilogx(self.sh_f.tol_span, self.sh_f.eps_span, 'o', label='computed')
		tol_span_fine = np.logspace(np.log10(self.sh_f.tol_e), 0., num=100, base=10, endpoint=True)
		plt.loglog(tol_span_fine, self.sh_f.tol_to_eps(tol_span_fine), label='interp. spline')
		plt.legend()
		plt.savefig(folder_name+'tol_to_eps_f.pdf')

		plt.figure()
		plt.semilogx(self.sh_g.tol_span, self.sh_g.eps_span, 'o', label='computed')
		tol_span_fine = np.logspace(np.log10(self.sh_g.tol_e), 0., num=100, base=10, endpoint=True)
		plt.loglog(tol_span_fine, self.sh_g.tol_to_eps(tol_span_fine), label='interp. spline')
		plt.legend()
		plt.savefig(folder_name+'tol_to_eps_g.pdf')






