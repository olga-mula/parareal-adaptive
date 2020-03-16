"""
SPDX-FileCopyrightText: 2019 Olga Mula (Paris Dauphine University)

SPDX-License-Identifier: GPL-3.0-or-later

This routine solves an ODE with the adaptive parareal algorithm.
We compare its performance with the classical parareal algorithm.
"""

import os
import argparse
from ode import ode_dict
from parareal_factory import Propagator, Piecewise_Propagator, Parareal_Algorithm
from summary_factory import summary_run

# MAIN PROGRAM
# ============

# Receive args
# ------------
parser = argparse.ArgumentParser()
parser.add_argument(
	'-ode', '--ode', default='Brusselator',
	help='{{{}}}'.format(', '.join(sorted(ode_dict.keys()))))
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
	'-integrator_f', '--integrator_f', type=bool, default='Radau', help='Fine integrator: {RK45, RK23, DOP853, Radau, LSODA}')
  parser.add_argument(
	'-integrator_g', '--integrator_g', type=bool, default='RK45', help='Fine integrator: {RK45, RK23, DOP853, Radau, LSODA}')
parser.add_argument(
	'-id','--id', help='Job ID for output folder')
args = parser.parse_args()

# Set ode problem
# ---------------
if args.ode not in ode_dict:
	raise Exception('ODE type '+args.ode+' not supported')
if args.integrator_f not in ['RK45', 'RK23', 'DOP853', 'Radau', 'LSODA']:
	raise Exception('Fine integrator '+args.integrator_f+' not supported')
if args.integrator_g not in ['RK45', 'RK23', 'DOP853', 'Radau', 'LSODA']:
	raise Exception('Fine integrator '+args.integrator_g+' not supported')

ODE = ode_dict[args.ode]
ode = ODE()
print(ode.info())

# Folder management
# -----------------
T_formatted = '{:d}'.format(int(args.T))
N_formatted = '{:d}'.format(args.N)
eps_formatted = '{:.1e}'.format(args.eps)

folder_name = args.ode + '/T_'+ T_formatted + '-N_'+ N_formatted + '-eps_'+ eps_formatted
folder_name_sh = args.ode + '/T_'+T_formatted

if not os.path.exists(folder_name):
	os.makedirs(folder_name)

# Parareal algorithm.
# Some parameters are set via script arguments
# ============================================
ti = 0.
tf = args.T                  # Final time
N     = args.N	             # Nb macro-intervals <=> Nb procs
eps   = args.eps             # Target accuracy
eps_g = args.eps_g           # Accuracy coarse integ
integrator_f = args.integrator_f	     # Fine integrator
integrator_g = args.integrator_g 	     # Coarse integrator
balance_tasks_cp = False     # Balance tasks in classical parareal
balance_tasks_ap = True	     # Balance tasks in adaptive parareal
compute_sh = args.compute_sh # Store eps-to-tol abacus

# Create object of class Parareal_Algorithm
p = Parareal_Algorithm(ode, ode.u0, [ti, tf], N, integrator_g=integrator_g, integrator_f=integrator_f, eps_g = eps_g, compute_sh=compute_sh)

# Run classical parareal and store information on preformance
pl, fl, gl, k_classic = p.run(eps, adaptive=False, balance_tasks=balance_tasks_cp, kmax=15)
summary_run(eps, k_classic, p, pl, fl, gl, folder_name+'/non-adaptive/')

# Run adaptive parareal and store information on preformance
pl, fl, gl, k_adaptive = p.run(eps, adaptive=True, balance_tasks=balance_tasks_ap, kth = k_classic-1, kmax=15)
summary_run(eps, k_adaptive, p, pl, fl, gl, folder_name+'/adaptive/')
