import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import argparse

from ode import VDP, Brusselator, Oregonator
from parareal_factory import Propagator, Piecewise_Propagator, Parareal_Algorithm

# from __future__ import print_function
# import numpy as np
# from scipy.integrate import ode, solve_ivp
# import time
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
# import matplotlib as mpl
# import matplotlib.pyplot as plt


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
    mu = 1.
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
    [ti, tf] = [0.0, 500]
    method = 'BDF'
    ode = ODE()

print(ode.info())

# Test solution
# =============
N = 10
p = Parareal_Algorithm(ode, u0, [ti, tf], N)
pl, fl, gl = p.run()
c = p.compute_cost()
print(c)


for k, sol in enumerate(pl):
    plt.figure()
    sol.plot()
    # p.exact.plot()
    plt.ylim(ode.ylim())
    plt.savefig(folder_name+'iter-k-'+str(k)+'.pdf')