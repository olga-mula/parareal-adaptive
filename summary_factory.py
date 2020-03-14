"""
	Copyright (c) 2019 Olga Mula, Paris Dauphine University
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

from parareal_factory import Propagator, Piecewise_Propagator, Parareal_Algorithm

def summary_run(eps, p, pl, fl, gl, folder_name):
  """
    Store data regarding the run and make pictures (convergence analysis, cost of propagators).
    Everything is stored in folder_name.
  """

	if not os.path.exists(folder_name):
		os.makedirs(folder_name)

	# Error
	# =====
	print('Convergence Analysis.')
	print('=====================')
	err = p.compute_error()

	values = range(7)
	cm = plt.get_cmap('jet') 
	cNorm  = colors.Normalize(vmin=0, vmax=values[-1])
	scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
	plt.figure()
	for k, e in enumerate(err):
		t = p.exact.ivp.t
		plt.semilogy(t, e, 'x', label='k='+str(k), color=scalarMap.to_rgba(values[k]))
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