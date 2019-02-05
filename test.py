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
    mu = .1
    u0 = np.array([2, 0])
    [ti, tf] = [0.0, 7]
    method = 'LSODA'
    ode = ODE(mu)
elif ODE.name() == 'Brusselator':
    A = 1.
    B = 3.
    u0 = np.array([0., 1.])
    [ti, tf] = [0.0, 10]
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
N = 10
p = Parareal_Algorithm(ode, u0, [ti, tf], N)
pl, fl, gl = p.run()

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
plt.savefig(folder_name+'conv-parareal.pdf')

# Evaluate cost
# =============
print('Cost.')
print('=====')
cost_g, cost_f, cost_parareal, cost_exact = p.compute_cost()
print('Exact propagator: ', cost_exact)
print('Adaptive Parareal: ', cost_parareal)

# Figures
# =======
# Exact solution
plt.figure()
p.exact.plot()
plt.xlim(ode.ylim())
plt.ylim(ode.ylim())
plt.savefig(folder_name+'exact.pdf')

# Parareal solution
for k, sol in enumerate(pl):
    plt.figure()
    sol.plot()
    plt.xlim(ode.ylim())
    plt.ylim(ode.ylim())
    plt.savefig(folder_name+'para-sol-k-'+str(k)+'.pdf')

