"""
	Copyright (c) 2019 Olga Mula
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import argparse

from ode import VDP, Brusselator, Oregonator
from parareal_factory import Propagator, Piecewise_Propagator, Parareal_Algorithm

def summary_run(eps, p, pl, fl, gl, folder_name):

	if not os.path.exists(folder_name):
		os.makedirs(folder_name)

	# Error
	# =====
	print('Convergence Analysis.')
	print('=====================')
	err = p.compute_error()

	plt.figure()
	for k, e in enumerate(err):
		t = p.exact.ivp.t
		plt.semilogy(t, e, 'x', label='k='+str(k))
	plt.legend()
	plt.savefig(folder_name+'conv-parareal-detail.pdf')
	plt.close()

	plt.figure()
	e = list()
	t = np.linspace(ti, tf, num=N+1, endpoint=True)
	for para in p.pl:
		e.append(np.max(para.eval(t)-p.exact.eval(t)))
	plt.semilogy(e, 'o-')
	plt.legend()
	plt.savefig(folder_name+'conv-parareal.pdf')
	plt.close()


	# Evaluate cost
	# =============
	print()
	print('Cost Parareal.')
	print('==============')
	cost_g, cost_f, cost_parareal, cost_seq_fine = p.compute_cost(eps)
	print('Sequential fine propagator: ', cost_seq_fine)
	print('Parareal: ', cost_parareal)

	print()
	print('Cost: Save some figures.')
	print('========================')
	print('- Figure: tol to eps')
	print('- Figure: eps to cpu_time and t_steps.')
	print('- Figure: t_steps per macro-interval')
	
	p.plot_tol_to_eps(folder_name)
	p.plot_eps_to_key(folder_name, key='cpu_time')
	p.plot_eps_to_key(folder_name, key='t_steps')
	p.plot_cost_detail(folder_name, key='t_steps')

	# Figures
	# =======
	# Exact solution
	plt.figure()
	p.exact.plot()
	plt.xlim(ode.ylim())
	plt.ylim(ode.ylim())
	plt.savefig(folder_name+'exact.pdf')
	plt.close()

	# Parareal solution
	for k, sol in enumerate(pl):
		plt.figure()
		sol.plot()
		plt.xlim(ode.ylim())
		plt.ylim(ode.ylim())
		plt.savefig(folder_name+'para-sol-k-'+str(k)+'.pdf')
		plt.close()

# Dictionary of available odes
ode_dict = {'VDP': VDP, 'Brusselator': Brusselator, 'Oregonator':Oregonator}

# MAIN PROGRAM
# ============
parser = argparse.ArgumentParser()
parser.add_argument('-ode', '--ode_name', help='{VDP, Brusselator, Oregonator}')
parser.add_argument('-id','--id', help='Job ID for output folder')
args = parser.parse_args()

# Folder management
# -----------------
folder_name = ''
if args.id is not None:
	folder_name = args.ode_name + '/' + args.id + '/'
else:
	folder_name = args.ode_name + '/default/'

if not os.path.exists(folder_name):
	os.makedirs(folder_name)

# Set ode problem
# ---------------
ODE = None
if args.ode_name in ode_dict:
	ODE = ode_dict[args.ode_name]
else:
	ODE = ode_dict['VDP']

if ODE.name() == 'VDP':
	mu = 1.e6
	u0 = np.array([2, 0])
	[ti, tf] = [0.0, 11]
	method = 'LSODA'
	ode = ODE(mu)
elif ODE.name() == 'Brusselator':
	A = 1.
	B = 3.
	u0 = np.array([0., 1.])
	[ti, tf] = [0.0, 10.]
	method = 'LSODA'
	ode = ODE(A, B)
elif ODE.name() == 'Oregonator':
	u0 = np.array([1., 2., 3.])
	[ti, tf] = [0.0, 100]
	method = 'BDF'
	ode = ODE()

print(ode.info())

# Parareal algorithm
# ==================
N = 10					# Number of macro-intervals <=> Number of processors
eps = 1.e-10			# Target accuracy
integrator_f = 'Radau'	# Fine integrator
integrator_g = 'RK45' 	# Coarse integrator
eps_g = 1.e-1			# Accuracy coarse integrator
balance_tasks_cp = False# Balance tasks in classical parareal
balance_tasks_ap = True	# Balance tasks in adaptive parareal
compute_sh = True		# Store eps-to-tol abacus

# Create object of class Parareal_Algorithm
p = Parareal_Algorithm(ode, u0, [ti, tf], N, integrator_g=integrator_g, integrator_f=integrator_f, eps_g = eps_g, compute_sh=compute_sh)

# Run classical parareal
pl, fl, gl, k_classic = p.run(eps, adaptive=False, balance_tasks=balance_tasks_cp)
summary_run(eps, p, pl, fl, gl, folder_name+'/non-adaptive/')

# Run adaptive parareal
pl, fl, gl, k_adaptive = p.run(eps, adaptive=True, balance_tasks=balance_tasks_ap, kth = k_classic)
summary_run(eps, p, pl, fl, gl, folder_name+'/adaptive/')
