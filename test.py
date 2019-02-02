import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import argparse

from ode import VDP, Brusselator, Oregonator
from parareal_factory import Parareal_Solution, Parareal_Algorithm

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
p_span, f_span, g_span = p.run()

for k, sol in enumerate(p_span):
    plt.figure()
    sol.plot()
    p.exact.plot()
    plt.ylim(-3., 6.)
    plt.savefig(folder_name+'iter-k-'+str(k)+'.pdf')



# # Define problem. (Here we use the diffusion checkerboard problem)
# # ==============
# problem = Problem(l=3)
# hilbert = problem.ambientSpace()


# class Van_der_Pool():

#     def __init__(self, *args):
#         self.mu = args[0]

#     def f(self, t, u):
#         return np.array([ u[1], self.mu*(1 - u[0]*u[0])*u[1] - u[0] ])

#     def jac(self, t, u):
#         j = np.empty((2, 2))
#         j[0, 0] = 0.0
#         j[0, 1] = 1.0
#         j[1, 0] = -self.mu*2*u[0]*u[1] - 1
#         j[1, 1] = self.mu*(1 - u[0]*u[0])
#         return np.array([[0., 1.],[-self.mu*2*u[0]*u[1]-1., self.mu*(1 - u[0]*u[0])]])

# class Brusselator():

#     def __init__(self, *args):
#         self.A = float(args[0])
#         self.B = float(args[1])

#     def f(self, t, u):
#         x = u[0]
#         y = u[1]
#         return np.asarray([self.A+(x**2)*y-(self.B+1.)*x, self.B*x - (x**2)*y])

#     def jac(self, t, u):
#         x = u[0]
#         y = u[1]
#         j = np.empty((2, 2))
#         j[0, 0] = 2*x*y-(self.B+1.)
#         j[0, 1] = x**2
#         j[1, 0] = self.B - 2*x*y
#         j[1, 1] = x**2
#         return j

# class Oregonator():

#     def __init__(self, *args):
#         pass

#     def f(self, t, u):
#         f = np.zeros(3)
#         f[0] = 77.27*(u[1]+u[0]*(1.-8.375e-6*u[0]-u[1]))
#         f[1] = (1./77.27)*(u[2]-(1+u[0])*u[1])
#         f[2] = 0.161*(u[0]-u[2])
#         return f

#     def jac(self, t, u):
#         j = np.empty((3, 3))
#         j[0, 0] = 77.27*( (1.-8.375e-6*u[0]-u[1]) -8.375e-6*u[0])
#         j[0, 1] = 77.27*( 1.-u[0] )
#         j[0, 2] = 0.
#         j[1, 0] = (1./77.27)*(-u[1])
#         j[1, 1] = (1./77.27)*(-(1+u[0]))
#         j[1, 2] = (1./77.27)
#         j[2, 0] = 0.161
#         j[2, 1] = 0.
#         j[2, 2] = -0.161
#         return j 


# def propagate(ode, u0, ti, tf, method, atol=1.e-8, rtol=1.e-13, **kwargs):

#     start = time.time()
#     solver = solve_ivp(ode.f, [ti, tf], u0, jac = ode.jac, method = method, atol=atol, rtol=rtol, dense_output=True)
#     end = time.time()

#     out = dict()
#     out['cpu_time'] = end-start

#     return solver, out
        

# ode_name = 'Van_der_Pool'

# if ode_name == 'Van_der_Pool':
#     mu = 1.0
#     u0 = np.array([2, 0])
#     [ti, tf] = [0.0, 500]
#     method = 'LSODA'
#     ode = Van_der_Pool(mu)
    
# elif ode_name == 'Brusselator':
#     A = 1.
#     B = 3.
#     u0 = np.array([0., 1.])
#     [ti, tf] = [0.0, 500]
#     method = 'LSODA'
#     ode = Brusselator(A, B)
# elif ode_name == 'Oregonator':
#     u0 = np.array([1., 2., 3.])
#     [ti, tf] = [0.0, 500]
#     method = 'BDF'
#     ode = Oregonator()


# # Reference solver
# tol_ref = 1.e-11
# sol_ref, out = propagate(ode, u0, ti, tf, method, atol=tol_ref, rtol=tol_ref)
# t = sol_ref.t
# uref = sol_ref.y

# # Accuracy study
# cpu_time = list()
# err = list()
# tol_list = np.logspace(-11, -5, num=10)
# print(tol_list)

# err_t = list()
# U = list()

# for tol in tol_list:
#     solver, out = propagate(ode, u0, ti, tf, method, atol=tol, rtol=tol/10)
#     cpu_time.append(out['cpu_time'])

#     # L2 Error in time
#     u = solver.sol(t)
    
#     # Store info
#     U.append(u)
#     err_t.append(np.linalg.norm(u-uref, axis=0))
#     err.append(np.max(err_t[-1]))
    
# # Plots
# # =====

# # Ref solution
# fig = plt.figure()
# if ode_name == 'Oregonator':
#     ax = Axes3D(fig)
#     ax.scatter(uref[0], uref[1], uref[2])
# else:
#     plt.plot(uref[0], uref[1], label='ref')
#     idx = 3
#     plt.plot(U[idx][0], U[idx][1], label='err='+str(err[idx]))
#     plt.legend()
# plt.savefig('sol.pdf')


# # err_t
# plt.figure()
# for i, e in enumerate(err_t):
#     plt.semilogy(t, e, 'x-',label=str(tol_list[i]))
# plt.legend()
# plt.savefig('err-time.pdf')

# # max_t err_t
# plt.figure()
# plt.loglog(tol_list, err)
# plt.savefig('err-max.pdf')

# # tol VS cpu_time
# plt.figure()
# plt.semilogx(err, cpu_time)
# plt.savefig('err-vs-cpu-time.pdf')