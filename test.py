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

	cg = cost_g['total']
	cf = cost_f['total']
	cseq = cost_seq_fine['total']
	np.savez(folder_name+'cost.npz', cost_g=cg, cost_f=cf, cost_seq_fine=cseq)
	# To recover:
	# d = load('cost.npz')
	# cf = d['cost_f'].item()  ---> Returns int with cost in nb time steps
	print('Sequential fine propagator: ', cost_seq_fine, cseq)
	print('Coarse:', cost_g, cg)
	print('Fine:', cost_f, cf)
	print('Parareal: ', cost_parareal, cg+cf)

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

# MAIN PROGRAM
# ============

# Dictionary of available odes
ode_dict = {'VDP': VDP, 'Brusselator': Brusselator, 'Oregonator': Oregonator}

# Receive args
# ------------
parser = argparse.ArgumentParser()
parser.add_argument(
	'-ode', '--ode_name', default='Brusselator',
	help='{VDP,Brusselator,Oregonator}')
parser.add_argument(
	'-N', '--N', type=int, default=10, help='Number of procs')
parser.add_argument(
	'-T', '--T', type=float, default=10, help='Final time')
parser.add_argument(
	'-eps', '--eps', type=float, default=1.e-6, help='Final target accuracy')
parser.add_argument(
	'-eps_g', '--eps_g', type=float, default=1.e-1, help='Accuracy coarse solver')
parser.add_argument(
	'-compute_sh', '--compute_sh', type=bool, default=True, help='Compute abacus')
parser.add_argument(
	'-id','--id', help='Job ID for output folder')
args = parser.parse_args()

# Set ode problem
# ---------------
if args.ode_name not in ode_dict:
	raise Exception('ODE type '+args.ode_name+' not supported')

ODE = ode_dict[args.ode_name]
ode = ODE()
print(ode.info())

# Folder management
# -----------------
T_formatted = '{:d}'.format(int(args.T))
N_formatted = '{:d}'.format(args.N)
eps_formatted = '{:.1e}'.format(args.eps)

folder_name = args.ode_name + '/T_'+ T_formatted + '-N_'+ N_formatted + '-eps_'+ eps_formatted + '/'
folder_name_sh = args.ode_name + '/T_'+T_formatted

if not os.path.exists(folder_name):
	os.makedirs(folder_name)

# Parareal algorithm
# ==================
ti = 0.
tf = args.T         # Final time
N     = args.N	    # Nb macro-intervals <=> Nb procs
eps   = args.eps    # Target accuracy
eps_g = args.eps_g  # Accuracy coarse integ
integrator_f = 'Radau'	# Fine integrator
integrator_g = 'RK45' 	# Coarse integrator
balance_tasks_cp = False# Balance tasks in classical parareal
balance_tasks_ap = True	# Balance tasks in adaptive parareal
compute_sh = args.compute_sh # Store eps-to-tol abacus

# Create object of class Parareal_Algorithm
p = Parareal_Algorithm(ode, ode.u0, [ti, tf], N, integrator_g=integrator_g, integrator_f=integrator_f, eps_g = eps_g, compute_sh=compute_sh)

# Run classical parareal
pl, fl, gl, k_classic = p.run(eps, adaptive=False, balance_tasks=balance_tasks_cp)
summary_run(eps, p, pl, fl, gl, folder_name+'/non-adaptive/')

# Run adaptive parareal
pl, fl, gl, k_adaptive = p.run(eps, adaptive=True, balance_tasks=balance_tasks_ap, kth = k_classic-1)
summary_run(eps, p, pl, fl, gl, folder_name+'/adaptive/')
