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

    published in the International Journal of Infectious Diseases 93 (2020) 211–216

    https://doi.org/10.1016/j.ijid.2020.02.058

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


class SEEIUR():
    
  """
    Source:Predicting the number of reported and unreported cases for the COVID-19 epidemic in South Korea, Italy, France and Germany.
    
    Magal, P. and Webb, G.
    
    https://doi.org/10.1101/2020.03.21.20040154.

  Unknowns:
    - S: Susceptible
    - E1: Exposed non-Infctious
    - E2: Exposed Infctious
    - I: Infected 
    - U: Unreported
    - R: Removed
  """
  def __init__(self, action='I'):
    """ODE of the type du/dt = f(t,u)

    Argument:
      action: keys: (not implemted atm for this case)
        - 'N': Models naive action against the virus: no individual nor governmental reaction.
        - 'I': Individual reaction but no governmental reaction
        - 'I+G': Individual and governmental reaction

    The parameter action sets the following parameters:
      
    """
          
    if action=='N':
      """Naive action against the virus: no individual nor governmental reaction"""
      sigma = 1./1.5; beta  = 0.45; delta = 1./3.7; gamma1 = 1./7; gamma2 = 1./7; nu = 0.1
    elif action=='I':
      """Individual reaction but no governmental reaction"""
      sigma = 1./1.5; beta  = 0.45; delta = 1./3.7; gamma1 = 1./7; gamma2 = 1./7; nu = 0.1
    elif action=='I+G':
      """Individual and governmental reaction"""
      sigma = 1./1.5; beta  = 0.45; delta = 1./3.7; gamma1 = 1./7; gamma2 = 1./7; nu = 0.1
    else:
      warnings.warn('SEEIUR ODE. action='+action+' not supported. Proceed with action=I+G', UserWarning)
      sigma = 1./1.5; beta  = 0.45; delta = 1./3.7; gamma1 = 1./7; gamma2 = 1./7; nu = 0.1

    N0 = 14e6
    I0 = nu*N0/1000
    U0 = (1-nu)*N0/1000
    R0 = 0
    E20= 8.*U0/5.
    E10= 8.*(E20+U0)/5.
    S0 = N0-E10-E20-I0-U0
      
    self.u0 = np.array([S0, E10, E20, I0, U0, R0])
    self.beta = lambda t: beta
    self.delta = lambda t: delta
    self.sigma = lambda t: sigma
    self.nu = lambda t: nu
    self.gamma1 = lambda t: gamma1
    self.gamma2 = lambda t: gamma2
    self.N = lambda t: N0

  @staticmethod
  def name():
    return 'SEEIUR'

  def info(self):
    s = 'Problem\n'
    s += '=========\n'
    s += 'SEEIUR model for spread of COVID-19 in Wuhan'
    s += '\n'
    return s

  def info(self):
    return 'SEEIUR model for spread of COVID-19 in Wuhan'

  def f(self, t, u):
    S,E1,E2,I,U,R = u
    f = np.zeros(6)
    
    f[0]  = -self.beta(t)*S*(E2+U+I)/self.N(t)
    f[1]  = self.beta(t)*S*(E2+U+I)/self.N(t) - self.delta(t)*E1
    f[2]  = self.delta(t)*E1 - self.sigma(t)*E2
    f[3]  = self.nu(t)*self.sigma(t)*E2 - self.gamma1(t)*I
    f[4]  = (1-self.nu(t))*self.sigma(t)*E2 - self.gamma2(t)*U
    f[5]  = self.gamma1(t)*I + self.gamma2(t)*U
    return f

  def jac(self, t, u):
    S,E1,E2,I,U,R = u
    j = np.zeros((6,6))

    j[0,0] = -self.beta(t)*(E2+U+I)/self.N(t)
    j[0,1] = 0.
    j[0,2] = -self.beta(t)*S/self.N(t)
    j[0,3] = -self.beta(t)*S/self.N(t)
    j[0,4] = -self.beta(t)*S/self.N(t)
    j[0,5] = 0.
    
    j[1,0] = self.beta(t)*(E2+U+I)/self.N(t)
    j[1,1] = -self.delta(t)
    j[1,2] = self.beta(t)*S/self.N(t)
    j[1,3] = self.beta(t)*S/self.N(t)
    j[1,4] = self.beta(t)*S/self.N(t)
    j[1,5] = 0.
    
    j[2,0] = 0.
    j[2,1] = self.delta(t)
    j[2,2] = -self.sigma(t)
    j[2,3] = 0.
    j[2,4] = 0.
    j[2,5] = 0.
    
    j[3,0] = 0.
    j[3,1] = 0.
    j[3,2] = self.nu(t)*self.sigma(t)
    j[3,3] = -self.gamma1(t)
    j[3,4] = 0.
    j[3,5] = 0.
    
    j[4,0] = 0.
    j[4,1] = 0.
    j[4,2] = (1-self.nu(t))*self.sigma(t)
    j[4,3] = 0.
    j[4,4] = -self.gamma2(t)
    j[4,5] = 0.

    
    j[5,0] = 0.
    j[5,1] = 0.
    j[5,2] = 0.
    j[5,3] = self.gamma1(t)
    j[5,4] = self.gamma2(t)
    j[5,5] = 0.

    return j
    
class SE5IHCRD():
    
  """SE5IHCRD

  Source: Expected impact of lockdown in Île-de-France and possible exit strategies
    
  Laura Di Domenico, Giulia Pullano, Chiara E. Sabbatini, Pierre-Yves Boëlle, Vittoria Colizza

  https://www.epicx-lab.com/uploads/9/6/9/4/9694133/inserm-covid-19_report_lockdown_idf-20200412.pdf

  Unknowns:
    
    - S: Susceptible
    - E: Exposed non-Infctious
    - Ip: Infected and pre-symptomatic (already infectious)
    - Ia: Infected and a-symptomatic (but infectious)
    - Ips: Infected and paucisymptomatic
    - Ims: Infected with mild symptoms
    - Iss: Infected with severe symptoms
    - H: Hospitalized
    - C: Intensive Care Unit
    - R: Removed
    - D: Dead
    
  """
  def __init__(self, action='I'):
    """ODE of the type du/dt = f(t,u)

    Argument:
      action: keys: (not implemted atm for this case)
        - 'N': Models naive action against the virus: no individual nor governmental reaction.
        - 'I': Individual reaction but no governmental reaction
        - 'I+G': Individual and governmental reaction

    The parameter action sets the following parameters:
      
    """
          
    if action=='N':
      """Naive action against the virus: no individual nor governmental reaction"""
      # Model parameters 
      betaIp = 0.51; betaIa = 0.51; betaIps = 0.51; betaIms = 0.; betaIss = 0.; betaH = 0.; betaC = 0.
      epsil  = 1/3.7; mup = 1/1.5; pa = 0.5; mu = 1/2.3; pps = 0.2; pms = 0.7; pss = 0.1; pC = 0.36
      lamCR = 0.05; lamCD = 0.0074; lamHR = 0.072; lamHD = 0.0042
    elif action=='I':
      """Individual reaction but no governmental reaction"""
      betaIp = 0.51; betaIa = 0.51; betaIps = 0.51; betaIms = 0.; betaIss = 0.; betaH = 0.; betaC = 0.
      epsil  = 1/3.7; mup = 1/1.5; pa = 0.5; mu = 1/2.3; pps = 0.2; pms = 0.7; pss = 0.1; pC = 0.36
      lamCR = 0.05; lamCD = 0.0074; lamHR = 0.072; lamHD = 0.0042
    elif action=='I+G':
      """Individual and governmental reaction"""
      betaIp = 0.51; betaIa = 0.51; betaIps = 0.51; betaIms = 0.; betaIss = 0.; betaH = 0.; betaC = 0.
      epsil  = 1/3.7; mup = 1/1.5; pa = 0.5; mu = 1/2.3; pps = 0.2; pms = 0.7; pss = 0.1; pC = 0.36
      lamCR = 0.05; lamCD = 0.0074; lamHR = 0.072; lamHD = 0.0042
    else:
      warnings.warn('SE5IHCRD ODE. action='+action+' not supported. Proceed with action=I+G', UserWarning)
      betaIp = 0.51; betaIa = 0.51; betaIps = 0.51; betaIms = 0.; betaIss = 0.; betaH = 0.; betaC = 0.
      epsil  = 1/3.7; mup = 1/1.5; pa = 0.5; mu = 1/2.3; pps = 0.2; pms = 0.7; pss = 0.1; pC = 0.36
      lamCR = 0.05; lamCD = 0.0074; lamHR = 0.072; lamHD = 0.0042

    N0 = 14e6
    E0 = 1000.
    Ip0 = 10000.
    Ia0 = 1000.
    Ips0 = 1000. 
    Ims0 = 1000. 
    Iss0 = 1000. 
    H0 = 1000.  
    C0 = 1000. 
    R0 = 1000. 
    D0 = 1000.  
    S0 = N0 - Ip0 - Ia0 - Ips0 - Ims0 - Iss0 - H0 - C0 - R0 - D0
      
    self.u0 = np.array([S0, E0, Ip0, Ia0, Ips0, Ims0, Iss0, C0, H0, R0, D0])
    self.betaIp = lambda t: betaIp
    self.betaIa = lambda t: betaIa
    self.betaIps = lambda t: betaIps
    self.betaIms = lambda t: betaIms
    self.betaIss = lambda t: betaIss
    self.betaH = lambda t: betaH
    self.betaC = lambda t: betaC    
    self.epsil = lambda t: epsil
    self.mup = lambda t: mup 
    self.pa = lambda t: pa 
    self.mu = lambda t: mu 
    self.pps = lambda t: pps 
    self.pms = lambda t: pms 
    self.pss = lambda t: pss 
    self.pC = lambda t: pC 
    self.lamCR = lambda t: lamCR 
    self.lamCD = lambda t: lamCD
    self.lamHR = lambda t: lamHR
    self.lamHD = lambda t: lamHD
    self.N = lambda t: N0

  @staticmethod
  def name():
    return 'SE5IHCRD'

  def info(self):
    s = 'Problem\n'
    s += '=========\n'
    s += 'SE5IHCRD model for spread of COVID-19 in Wuhan'
    s += '\n'
    return s

  def info(self):
    return 'SE5IHCRD model for spread of COVID-19 in Wuhan'

  def f(self, t, u):
    S,E,Ip,Ia,Ips,Ims,Iss,C,H,R,D =  u 
    f = np.zeros(11)

    f[0] = - (S/self.N(t))*( self.betaIp(t)*Ip + self.betaIa(t)*Ia + self.betaIps(t)*Ips + self.betaIms(t)*Ims + self.betaIss(t)*Iss + self.betaH(t)*H + self.betaC(t)*C)
    f[1] =   (S/self.N(t))*( self.betaIp(t)*Ip + self.betaIa(t)*Ia + self.betaIps(t)*Ips + self.betaIms(t)*Ims + self.betaIss(t)*Iss + self.betaH(t)*H + self.betaC(t)*C) - self.epsil(t)*E
    f[2] = self.epsil(t)*E - self.mup(t)*Ip 
    f[3] = self.pa(t)*self.mup(t)*Ip - self.mu(t)*Ia 
    f[4] = self.pps(t)*(1-self.pa(t))*self.mup(t)*Ip - self.mu(t)*Ips 
    f[5] = self.pms(t)*(1-self.pa(t))*self.mup(t)*Ip - self.mu(t)*Ims 
    f[6] = self.pss(t)*(1-self.pa(t))*self.mup(t)*Ip - self.mu(t)*Iss 
    f[7]  = self.pC(t)*self.mu(t)*Iss- (self.lamCR(t)+self.lamCD(t))*C
    f[8]  = (1-self.pC(t))*self.mu(t)*Iss - (self.lamHR(t)+self.lamHD(t))*H
    f[9] = self.lamCR(t)*C + self.lamHR(t)*H 
    f[10] = self.lamCD(t)*C + self.lamHD(t)*H
    
    return f

  def jac(self, t, u):
    S,E,Ip,Ia,Ips,Ims,Iss,C,H,R,D =  u 
    j = np.zeros((11,11))

    # derivattives of dSdt 
    j[0,0] = -(self.betaIp(t)*Ip + self.betaIa(t)*Ia + self.betaIps(t)*Ips + self.betaIms(t)*Ims + self.betaIss(t)*Iss + self.betaH(t)*H + self.betaC(t)*C)/self.N(t)
    j[0,1] = 0.
    j[0,2] = - (S/self.N(t))*(self.betaIp(t)) 
    j[0,3] = - (S/self.N(t))*(self.betaIa(t)) 
    j[0,4] = - (S/self.N(t))*(self.betaIps(t)) 
    j[0,5] = - (S/self.N(t))*(self.betaIms(t)) 
    j[0,6] = - (S/self.N(t))*(self.betaIss(t)) 
    j[0,7] = - (S/self.N(t))*(self.betaC(t)) 
    j[0,8] = - (S/self.N(t))*(self.betaH(t)) 
    j[0,9] = 0.
    j[0,10]= 0.
    
    # derivattives of dEdt
    j[1,0] = (self.betaIp(t)*Ip + self.betaIa(t)*Ia + self.betaIps(t)*Ips + self.betaIms(t)*Ims + self.betaIss(t)*Iss + self.betaH(t)*H + self.betaC(t)*C)/self.N(t)
    j[1,1] = -self.epsil(t)
    j[1,2] = (S/self.N(t))*(self.betaIp(t)) 
    j[1,3] = (S/self.N(t))*(self.betaIa(t)) 
    j[1,4] = (S/self.N(t))*(self.betaIps(t)) 
    j[1,5] = (S/self.N(t))*(self.betaIms(t)) 
    j[1,6] = (S/self.N(t))*(self.betaIss(t)) 
    j[1,7] = (S/self.N(t))*(self.betaC(t)) 
    j[1,8] = (S/self.N(t))*(self.betaH(t))
    j[1,9] = 0.
    j[1,10]= 0.
    
    # derivattives of dIpdt
    j[2,0] = 0.                  
    j[2,1] = self.epsil(t)            
    j[2,2] = -self.mup(t)
    j[2,3] = 0.
    j[2,4] = 0.
    j[2,5] = 0.
    j[2,6] = 0.
    j[2,7] = 0.
    j[2,8] = 0.
    j[2,9] = 0.
    j[2,10]= 0.
    
    # derivattives of dIadt 
    j[3,0] = 0.
    j[3,1] = 0.
    j[3,2] = self.pa(t)*self.mup(t) 
    j[3,3] = -self.mu(t) 
    j[3,4] = 0.
    j[3,5] = 0.
    j[3,6] = 0.
    j[3,7] = 0.
    j[3,8] = 0.
    j[3,9] = 0.
    j[3,10]= 0.
    
    
    # derivattives of dIpsdt
    j[4,0] = 0.
    j[4,1] = 0.
    j[4,2] = self.pps(t)*(1-self.pa(t))*self.mup(t) 
    j[4,3] = 0.
    j[4,4] = -self.mu(t) 
    j[4,5] = 0.
    j[4,6] = 0.
    j[4,7] = 0.
    j[4,8] = 0.
    j[4,9] = 0.
    j[4,10]= 0.
    
    
    # derivattives of dImsdt
    j[5,0] = 0.
    j[5,1] = 0.
    j[5,2] = self.pms(t)*(1-self.pa(t))*self.mup(t) 
    j[5,3] = 0.
    j[5,4] = 0.
    j[5,5] = -self.mu(t) 
    j[5,6] = 0.
    j[5,7] = 0.
    j[5,8] = 0.
    j[5,9] = 0.
    j[5,10]= 0.
    
    # derivattives of dIssdt
    j[6,0] = 0.
    j[6,1] = 0.
    j[6,2] = self.pss(t)*(1-self.pa(t))*self.mup(t) 
    j[6,3] = 0.
    j[6,4] = 0.
    j[6,5] = 0.
    j[6,6] = -self.mu(t) 
    j[6,7] = 0.
    j[6,8] = 0.
    j[6,9] = 0.
    j[6,10]= 0.
    
    
    # derivattives of dICdt
    j[7,0] = 0.
    j[7,1] = 0.
    j[7,2] = 0.
    j[7,3] = 0.
    j[7,4] = 0.
    j[7,5] = 0.
    j[7,6] = self.pC(t)*self.mu(t)
    j[7,7] = -(self.lamCR(t) + self.lamCD(t))
    j[7,8] = 0.
    j[7,9] = 0.
    j[7,10]= 0.
    
    
    # derivattives of dIHdt
    j[8,0] = 0.
    j[8,1] = 0.
    j[8,2] = 0.
    j[8,3] = 0.
    j[8,4] = 0.
    j[8,5] = 0.
    j[8,6] = (1-self.pC(t))*self.mu(t)
    j[8,7] = 0.
    j[8,8] = -(self.lamHR(t) + self.lamHD(t))
    j[8,9] = 0.
    j[8,10]= 0.
    
    # derivattives of dIRdt
    j[9,0] = 0.
    j[9,1] = 0.
    j[9,2] = 0.
    j[9,3] = 0.
    j[9,4] = 0.
    j[9,5] = 0.
    j[9,6] = 0.
    j[9,7] = self.lamCR(t) 
    j[9,8] = self.lamHR(t) 
    j[9,9] = 0.
    j[9,10]= 0.
    
    # derivattives of dIDdt
    j[10,0] = 0.
    j[10,1] = 0.
    j[10,2] = 0.
    j[10,3] = 0.
    j[10,4] = 0.
    j[10,5] = 0.
    j[10,6] = 0.
    j[10,7] = self.lamCD(t) 
    j[10,8] = self.lamHD(t)
    j[10,9] = 0.
    j[10,10]= 0.
    
    return j
    
    

# Dictionary of available odes
ode_dict = {'VDP': VDP, 'Brusselator': Brusselator, 'Oregonator': Oregonator, 'SEIR': SEIR, 'SEEIUR': SEEIUR, 'SE5IHCRD':SE5IHCRD}
dict_action_SIR = {'N': 'N', 'I': 'I', 'I+G': 'I+G'}