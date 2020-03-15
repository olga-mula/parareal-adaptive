"""
SPDX-FileCopyrightText: 2019 Olga Mula (Paris Dauphine University)

SPDX-License-Identifier: GPL-3.0-or-later
"""

import numpy as np
from scipy.integrate import ode, solve_ivp
import time


class VDP():
  """Van der Pol oscilator"""

  def __init__(self, u0=np.array([2, 0]), mu=4):
    """ODE of the type du/dt = f(t,u)

    Arguments:
      u0: initial condition
      mu: parameter of the Van der Pol equation:
        - mu close to 0 -> nonstiff equation
        - mu > 0.1       -> stiff equation
    """
    self.mu = mu
    self.u0 = u0

  @staticmethod
  def name():
    return 'VDP'

  def info(self):
    """Print info about ODE"""
    s = 'Problem\n'
    s += '=========\n'
    s += 'Van der Pol oscillator with mu='+str(self.mu)
    s += '\n'
    return s

  def xlim(self):
    """Limits abscissae (for plotting purposes)"""
    return [-2., 2.]

  def ylim(self):
    """Limits ordinates (for plotting purposes)"""
    return [-6.5, 6.5]

  def f(self, t, u):
    """du/dt = f(t,u)"""
    return np.array([ u[1], self.mu*(1 - u[0]*u[0])*u[1] - u[0] ])

  def jac(self, t, u):
    """Jacobian of f(t,u)"""
    j = np.empty((2, 2))
    j[0, 0] = 0.0
    j[0, 1] = 1.0
    j[1, 0] = -self.mu*2*u[0]*u[1] - 1
    j[1, 1] = self.mu*(1 - u[0]*u[0])
    return np.array([[0., 1.],[-self.mu*2*u[0]*u[1]-1., self.mu*(1 - u[0]*u[0])]])

class Brusselator():
  """Brusselator system"""

  def __init__(self, u0=np.array([0., 1.]), A=1., B=3.):
    """ODE of the type du/dt = f(t,u)

    Arguments:
      u0: initial condition
      A, B: parameters of the Brusselator equation
    """
    self.u0 = u0
    self.A  = A
    self.B  = B

  @staticmethod
  def name():
    return 'Brusselator'

  def info(self):
    s = 'Problem\n'
    s += '=========\n'
    s += 'Brusselator with A='+str(self.A)+'; B='+str(self.B)
    s += '\n'
    return s

  def xlim(self):
    """Limits abscissae (for plotting purposes)"""
    return [-0.5, 4.]

  def ylim(self):
    """Limits ordinates (for plotting purposes)"""
    return [-0.5, 5.5]

  def f(self, t, u):
    """du/dt = f(t,u)"""
    x = u[0]
    y = u[1]
    return np.asarray([self.A+(x**2)*y-(self.B+1.)*x, self.B*x - (x**2)*y])

  def jac(self, t, u):
    """Jacobian of f(t,u)"""
    x = u[0]
    y = u[1]
    j = np.empty((2, 2))
    j[0, 0] = 2*x*y-(self.B+1.)
    j[0, 1] = x**2
    j[1, 0] = self.B - 2*x*y
    j[1, 1] = x**2
    return j


# Dictionary of available odes
ode_dict = {'VDP': VDP, 'Brusselator': Brusselator}