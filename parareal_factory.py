import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.integrate import solve_ivp

from ode import VDP, Brusselator, Oregonator

class Parareal_Solution():

	def __init__(self, ode, u0, t_span, u_span = None, integrator='BDF', tol=1.e-10):
		self.ode = ode
		self.u0 = u0
		self.tol = tol

		self.t_span = t_span
		self.interval_span = [[ti, tf] for (ti,tf) in zip(t_span[:-1], t_span[1:])]

		self.ivp_span = list()

		if u_span is None:
			ti = t_span[0]
			tf = t_span[-1]
			ivp = solve_ivp(ode.f, [ti, tf], u0, jac = ode.jac, method = integrator, atol=tol, rtol=tol, dense_output=True)
			for i, t_interval in enumerate(self.interval_span): 
				self.ivp_span.append(ivp)
			self.u_span = ivp.sol(self.t_span).T
		else:
			self.u_span = np.asarray(u_span)
			for i, t_interval in enumerate(self.interval_span):
				u_init = u_span[i,:]
				self.ivp_span.append(solve_ivp(ode.f, t_interval, u_init, jac = ode.jac, method = integrator, atol=tol, rtol=tol, dense_output=True))

	def eval(self, t):
		if isinstance(t, float):
			if t == self.t_span[-1]:
				return self.ivp_span[-1].sol(t)
			else:
				idx_test = [int(t>=ti and t<tf) for (ti,tf) in self.interval_span]
				idx = np.argmax(idx_test)
				return self.ivp_span[idx].sol(t)
		else:
			return np.array([self.eval(ti) for ti in t.tolist()])

	def plot(self):
		color_list = cm.tab20c(np.linspace(0, 1, len(self.t_span-1)))
		for i, ivp in enumerate(self.ivp_span):
			t_vec = ivp.t
			u_vec = self.eval(t_vec).T
			plt.plot(u_vec[0,:], u_vec[1,:], color=color_list[i])


class Parareal_Algorithm():
	def __init__(self, ode, u0, t_interval, N, integrator_g='RK23', tol_g = 1.e-2, integrator_f='BDF', tol_f=1.e-2):
		self.ode   = ode
		self.u0    = u0
		[self.ti, self.tf] = t_interval
		self.N = N

		self.integrator_g = integrator_g
		self.tol_g = tol_g

		self.integrator_f = integrator_f
		self.tol_f = tol_f

		t_span = np.linspace(self.ti, self.tf, num=self.N+1, endpoint=True)
		self.exact = Parareal_Solution(self.ode, self.u0, t_span, u_span=None, integrator=self.integrator_f, tol=1.e-12)

	def run(self, eps=1.e-6, kmax=12):

		g_span = list()
		f_span = list()
		p_span = list()
		t_span_l = list()

		for k in range(kmax):
			print('Iteration k='+str(k))
			if k == 0:
				t_span = np.linspace(self.ti, self.tf, num=self.N+1, endpoint=True)
				g0 = Parareal_Solution(self.ode, self.u0, t_span, u_span=None, integrator=self.integrator_g, tol=self.tol_g)

				t_span_l.append(t_span)
				g_span.append(g0)
				p_span.append(g0)
			else:
				# Fine propagator (k-1)
				self.tol_f = 10**(-k-1)
				fkprev = Parareal_Solution(self.ode, self.u0, t_span_l[-1], u_span=p_span[-1].eval(t_span_l[-1]), integrator=self.integrator_f, tol=self.tol_f)
				f_span.append(fkprev)

				# Update t_span with last fine propagation
				t_span = self.balance_tasks(f_span[-1])
				t_span_l.append(t_span)

				# Coarse propagator (k). Uses updates t_span
				gk = Parareal_Solution(self.ode, self.u0, t_span_l[-1], \
					u_span=p_span[-1].eval(t_span_l[-1]), integrator=self.integrator_g, tol=self.tol_g)
				g_span.append(gk)

				# Parareal solution
				pk_u_span = g_span[-1].eval(t_span_l[-1]) + fkprev.eval(t_span_l[-1]) - g_span[-2].eval(t_span_l[-1])
				pk = Parareal_Solution(self.ode, self.u0, t_span_l[-1], u_span=pk_u_span, integrator=self.integrator_f, tol=self.tol_f)
				p_span.append(pk)

		return p_span, f_span, g_span

	def balance_tasks(self, parareal_sol):
		tl = list()
		for i, ivp in enumerate(parareal_sol.ivp_span):
			tl = tl + ivp.t[:-1].tolist()
		tl = tl + [self.tf]

		which_idxs = lambda m, n: np.rint(np.linspace(1,n, min(m,n))-1).astype(int)
		t_span = np.array(tl)[which_idxs(self.N+1,len(tl))]
		return t_span

def propagate(ode, u0, ti, tf, method, atol=1.e-8, rtol=1.e-13, **kwargs):

    start = time.time()
    solver = solve_ivp(ode.f, [ti, tf], u0, jac = ode.jac, method = method, atol=atol, rtol=rtol, dense_output=True)
    end = time.time()

    out = dict()
    out['cpu_time'] = end-start

    return solver, out
