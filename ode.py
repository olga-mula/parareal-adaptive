"""
SPDX-FileCopyrightText: 2019 Olga Mula (Paris Dauphine University)

SPDX-License-Identifier: GPL-3.0-or-later
"""

import numpy as np
from scipy.integrate import ode, solve_ivp
import time
import warnings


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

class Oregonator():
  """Oregonator"""
  def __init__(self, u0 = np.array([1.,1.,1.]), e1=0.00992063492063492, e2=1.984126984126984e-05, q=7.619047619047618e-05, g=1.):
    """ODE of the type du/dt = f(t,u)

    The default parameters give limiting cycle.
    """
    self.u0 = u0
    self.e1 = e1
    self.e2 = e2
    self.q  = q
    self.g  = g

  @staticmethod
  def name():
    return 'Oregonator'

  def info(self):
    s = 'Problem\n'
    s += '=========\n'
    s += 'Oregonator'
    s += '\n'
    return s

  def xlim(self):
    """Limits abscissae (for plotting purposes)"""
    return [-4, 4.]

  def ylim(self):
    """Limits ordinates (for plotting purposes)"""
    return [-4, 4]

  def zlim(self):
    """Limits ordinates (for plotting purposes)"""
    return [-4, 4]

  def info(self):
    return 'Oregonator'

  def f(self, t, u):
    f = np.zeros(3)
    f[0] = 1./self.e1*(-u[0]*u[1] + self.q*u[1] + u[0] - u[0]**2)
    f[1] = 1./self.e2*(-self.q*u[1]-u[0]*u[1]+self.g*u[2])
    f[2] = u[0]-u[2]
    return f

  def jac(self, t, u):
    j = np.empty((3, 3))
    j[0, 0] = 1./self.e1*(-u[1] + 1 - 2*u[0])
    j[0, 1] = 1./self.e1*(-u[0] + self.q)
    j[0, 2] = 0
    j[1, 0] = 1./self.e2*(-u[1])
    j[1, 1] = 1./self.e2*(-self.q-u[0])
    j[1, 2] = 1./self.e2*(self.g)
    j[2, 0] = 1
    j[2, 1] = 0
    j[2, 2] = -1
    return j

class SEIR():
  """SEIR

  Model of the spread of coronavirus COVID-19 in Wuhan proposed the paper:

    A conceptual model for the coronavirus disease 2019 (COVID-19) outbreak in Wuhan, China with individual reaction and governmental action
    
    by Qianying Lin, Shi Zhao, Daozhou Gao, Yijun Lou, Shu Yang, Salihu S. Musa, Maggie H. Wang, Yongli Cai, Weiming Wang, Lin Yang, Daihai He,

    published in the International Journal of Infectious Diseases 93 (2020) 211–216.

  REMARK: The present implementation follows the model described in the paper to the best of the author's ability to understand the data provided.

  Unknowns:
    - S: Susceptible
    - E: Exposed
    - I: Infectious
    - R: Recovered 
    - N: Total population
    - D: Risk perception
    - C: Number of cumulative cases (reported and not reported)
  """
  def __init__(self, action='I'):
    """ODE of the type du/dt = f(t,u)

    Argument:
      action: keys:
        - 'N': Models naive action against the virus: no individual nor governmental reaction.
        - 'I': Individual reaction but no governmental reaction
        - 'I+G': Individual and governmental reaction

    The parameter action sets the following parameters:
      - kappa: individual reaction (0=no reaction, >0: individuals react to the threat)
      - alpha1: parameter on governmental action between January 23-29, 2020
      - alpha2: parameter on governmental action after January 29, 2020
      - mu: emigration rate before closing of Wuhan on January 23, 2020 (set to zero after that date)
    """
    N0 = 14e6 # Initial population of Wuhan
    if action=='N':
      """Naive action against the virus: no individual nor governmental reaction"""
      kappa=0; alpha1=0; alpha2=0; mu=0
    elif action=='I':
      """Individual reaction but no governmental reaction"""
      kappa=1117.3; alpha1=0; alpha2=0; mu=0
    elif action=='I+G':
      """Individual and governmental reaction"""
      kappa=1117.3; alpha1=0.4249; alpha2=0.8478; mu=0.0205
    else:
      warnings.warn('SEIR ODE. action='+action+' not supported. Proceed with action=I+G', UserWarning)
      kappa=1117.3; alpha1=0.4249; alpha2=0.8478; mu=0.0205
      
    self.u0 = np.array([0.9*N0, 0, 0, 0, N0, 0, 0]) # [S, E, I, R, N, D, C]
    self.F = lambda t: 10*(t>=0)
    self.alpha = lambda t: alpha1*(t>=23 and t<=29) + alpha2*(t>29) # Governmental action
    self.kappa = kappa # Individual reaction
    self.mu = lambda t: mu*(t<=23)  # Closing of Wuhan (governmental action)
    self.sigma = 1./3
    self.gamma = 1./5
    self.d = 0.2
    self.l = 1./11.2
    self.beta0 = lambda t: 1.68*(t<=23)+0.59444*(t>23)

  @staticmethod
  def name():
    return 'SEIR'

  def info(self):
    s = 'Problem\n'
    s += '=========\n'
    s += 'SEIR model for spread of COVID-19 in Wuhan'
    s += '\n'
    return s

  def info(self):
    return 'SEIR model for spread of COVID-19 in Wuhan'

  def beta(self, t, u):
    return self.beta0(t)*(1-self.alpha(t))*(1-u[5]/u[4])**self.kappa

  def jac_beta(self, t, u):
    j = np.zeros(7)
    j[4] = self.beta0(t)*(1-self.alpha(t))*self.kappa*u[5]/(u[4]**2)*(1-u[5]/u[4])**(self.kappa-1)
    j[5] = self.beta0(t)*(1-self.alpha(t))*self.kappa*(-1./u[4])*(1-u[5]/u[4])**(self.kappa-1)
    return j

  def f(self, t, u):
    f = np.zeros(7)
    f[0] = -self.beta0(t)*self.F(t)*u[0]/u[4]-self.beta(t, u)*u[0]*u[2]/u[4]-self.mu(t)*u[0]
    f[1] = self.beta0(t)*self.F(t)*u[0]/u[4]+self.beta(t, u)*u[0]*u[2]/u[4]-(self.sigma+self.mu(t))*u[1]
    f[2] = self.sigma*u[1] - (self.gamma+self.mu(t))*u[2]
    f[3] = self.gamma*u[2]-self.mu(t)*u[3]
    f[4] = -self.mu(t)*u[4]
    f[5] = self.d*self.gamma*u[2] - self.l*u[5]
    f[6] = self.sigma*u[1]
    return f

  def jac(self, t, u):
    jb = self.jac_beta(t, u)

    j = np.zeros((7, 7))
    j[0, 0] = -self.beta0(t)*self.F(t)/u[4]-self.beta(t, u)*u[2]/u[4]-self.mu(t)
    j[0, 1] = 0
    j[0, 2] = -self.beta(t, u)*u[0]/u[4]
    j[0, 3] = 0
    j[0, 4] = self.beta0(t)*self.F(t)*u[0]/(u[4]**2)+self.beta(t, u)*u[0]*u[2]/(u[4]**2)-jb[4]*self.F(t)*u[0]/u[4]-jb[4]*u[0]*u[2]/u[4]
    j[0, 5] = -jb[5]*self.F(t)*u[0]/u[4]-jb[5]*u[0]*u[2]/u[4]
    j[0, 6] = 0

    j[1, 0] = self.beta0(t)*self.F(t)/u[4]+self.beta(t, u)*u[2]/u[4]
    j[1, 1] = -(self.sigma+self.mu(t))
    j[1, 2] = self.beta(t, u)*u[0]/u[4]
    j[1, 3] = 0
    j[1, 4] = -self.beta0(t)*self.F(t)*u[0]/(u[4]**2)-self.beta(t, u)*u[0]*u[2]/(u[4]**2)+jb[4]*self.F(t)*u[0]/u[4]+jb[4]*u[0]*u[2]/u[4]
    j[1, 5] = jb[5]*self.F(t)*u[0]/u[4]+jb[5]*u[0]*u[2]/u[4]
    j[1, 6] = 0

    j[2, 0] = 0
    j[2, 1] = self.sigma
    j[2, 2] = - (self.gamma+self.mu(t))
    j[2, 3] = 0
    j[2, 4] = 0
    j[2, 5] = 0
    j[2, 6] = 0

    j[3, 0] = 0
    j[3, 1] = 0
    j[3, 2] = self.gamma
    j[3, 3] = -self.mu(t)
    j[3, 4] = -jb[4]*u[3]
    j[3, 5] = -jb[5]*u[3]
    j[3, 6] = 0

    j[4, 0] = 0
    j[4, 1] = 0
    j[4, 2] = 0
    j[4, 3] = 0
    j[4, 4] = -self.mu(t)-jb[4]*u[4]
    j[4, 5] = -jb[5]*u[4]
    j[4, 6] = 0

    j[5, 0] = 0
    j[5, 1] = 0
    j[5, 2] = self.d*self.gamma
    j[5, 3] = 0
    j[5, 4] = 0
    j[5, 5] = - self.l
    j[5, 6] = 0

    j[6, 0] = 0
    j[6, 1] = self.sigma
    j[6, 2] = 0
    j[6, 3] = 0
    j[6, 4] = 0
    j[6, 5] = 0
    j[6, 6] = 0

    return j

# Dictionary of available odes
ode_dict = {'VDP': VDP, 'Brusselator': Brusselator, 'Oregonator': Oregonator, 'SEIR': SEIR}
dict_action_SIR = {'N': 'N', 'I': 'I', 'I+G': 'I+G'}