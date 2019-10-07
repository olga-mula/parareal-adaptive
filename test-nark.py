import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import ode, solve_ivp

from ode import Brusselator
from nonadaptive_rungekutta import NonAdaptiveRK23, NonAdaptiveRK45

A = 1.
B = 3.
ti = 0.
tf = 100.
u0 = np.array([0., 1.])
tol = 1.e-1

ode = Brusselator(A, B)

ivp = solve_ivp(ode.f, [ti, tf], u0, jac=ode.jac, method=NonAdaptiveRK23, atol=tol, rtol=tol, dense_output=True)

t = np.linspace(ti, tf, num=200, endpoint=True)
u=ivp.sol(t)

t_ivp = ivp.t
u_ivp = ivp.y

print(ivp.t.shape)

ivp_e1 = solve_ivp(ode.f, [ti, tf], u0, jac=ode.jac, method='Radau', atol=1.e-10, rtol=1.e-10, dense_output=True)
print(ivp_e1.t.shape)

for i in range(10):
	ivp_e = solve_ivp(ode.f, [i, i+1], u0, jac=ode.jac, method='Radau', atol=1.e-10, rtol=1.e-10, dense_output=True)
	print(ivp_e.t.shape)



plt.figure()
plt.plot(u_ivp[0,:], u_ivp[1,:])
plt.savefig('test-nark.pdf')