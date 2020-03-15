"""
SPDX-FileCopyrightText: 2019 Olga Mula (Paris Dauphine University)

SPDX-License-Identifier: GPL-3.0-or-later
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.special import factorial
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
import time
import warnings
import pickle
import os

COEF_EXACT = 3

class Propagator():
  """ODE Propagator

  Solve an initial value problem in a time interval when the object is initialized.
  """


  def __init__(self, ode, u0, t_interval, integrator='BDF', tol=1.e-12):
    """Create a Propagator
    
    Arguments:
      ode: object of ODE type
      u0: initial condition
      t_interval: time interval of the form [ti, tf]
      integrator: type of integrator (must be supported by solve_ivp from scipy.integrate)
      tol: accuracy of internal parameters atol and rtol from solve_ivp
    """
    self.ode = ode
    self.u0 = u0
    self.dim = u0.shape[0]
    [self.ti, self.tf] = t_interval
    self.integrator = integrator
    self.tol = tol

    start = time.time()
    with warnings.catch_warnings():
      warnings.filterwarnings("ignore")
      self.ivp = solve_ivp(ode.f, [self.ti, self.tf], u0, jac = ode.jac, method = integrator, atol=tol, rtol=tol, dense_output=True)
    end = time.time()

    self.cost = {
      'cpu_time': end-start,
      't_steps': len(self.ivp.t.tolist()),
      'n_rhs':   self.ivp.nfev, # number of evaluations of rhs
      'n_jac':   self.ivp.njev, # Number of evaluations of the Jacobian
      'nlu':     self.ivp.nlu,  # Number of lu decompositions
      'total':   len(self.ivp.t.tolist())+ \
        self.ivp.nfev + self.ivp.njev + self.ivp.nlu
    }

  def eval(self, t):
    """Return solution of ODE at time t."""
    return self.ivp.sol(t)

  def plot(self, color='black'):
    """Plot dynamics in the full interval."""
    if self.dim ==2:
      t_vec = self.ivp.t
      u_vec = self.eval(t_vec)
      plt.plot(u_vec[0,:], u_vec[1,:], color=color)
    else:
      raise ValueError('Dimension of problem is {}. Not supported for visualization.'.format(self.dim))

  def compute_cost(self, type='sequential'):
    """Compute propagator's cost."""
    return self.cost

class Piecewise_Propagator():
  """ODE Piecewise_Propagator

  Solve an initial value problem in a set of time intervals when the object is initialized.
  """

  def __init__(self, ode, u0, t_span, u_span, integrator='BDF', tol=1.e-12):
    """Create a Piecewise_Propagator
    
    Arguments:
      ode: object of ODE type
      u0: initial condition
      t_span: union of subintervals of the form [[ti0, tf0], [ti1, tf1], ...]
      u_span: intermediate initial conditions for each subintervals
      integrator: type of integrator (must be supported by solve_ivp from scipy.integrate)
      tol: accuracy of internal parameters atol and rtol from solve_ivp
    """
    self.ode = ode
    self.u0 = u0
    self.dim = u0.shape[0]
    self.integrator = integrator
    self.tol = tol

    self.t_span = t_span
    self.interval_span = [[ti, tf] for (ti,tf) in zip(t_span[:-1],
                                                      t_span[1:])]
    self.u_span = np.asarray(u_span) # Column j has solution at time t_j
    self.propagator_span = list()

    for i, t_interval in enumerate(self.interval_span):
      u_init = u_span[:,i]
      self.propagator_span.append(Propagator(ode, u_init, t_interval, integrator=integrator, tol=tol))

    self.cost = None

  def eval(self, t):
    """Return solution of ODE at time t."""
    if isinstance(t, float):
      if t == self.t_span[-1]:
        return self.propagator_span[-1].eval(t)
      else:
        idx_test = [int(t>=ti and t<tf) for (ti,tf)
              in self.interval_span]
        idx = np.argmax(idx_test)
        return self.propagator_span[idx].eval(t)
    else:
      return np.array([self.eval(ti) for ti in t.tolist()]).T

  def compute_cost(self, type='sequential'):
    """Compute propagator's cost."""
    cl = [prop.cost for prop in self.propagator_span]

    if type=='sequential':
      aggregate = sum
    else:
      aggregate = max

    keys = self.propagator_span[0].cost.keys()
    cost = {key: aggregate(c[key] for c in cl) for key in keys}

    self.cost = cost
    return cost

  def plot(self):
    """Plot dynamics in the full interval."""
    color_list = cm.tab20c(np.linspace(0, 1, len(self.t_span-1)))
    for i, prop in enumerate(self.propagator_span):
      t_vec = prop.ivp.t
      u_vec = self.eval(t_vec)
      plt.plot(u_vec[0,:], u_vec[1,:], color=color_list[i])

class SolverHelper():
  """SolverHelper of an ODE Propagator.

  Examine global accuracy of a Propagator in the l^2 or l^infinity sense over a time interval.
  Accuracy depends on the interval parameters atol and rtol of the solve_ivp integrator so
  a parametric study must be done.
  """

  def __init__(self, ode, u0, t_interval, integrator='RK45', integrator_e='LSODA', tol_e=1.e-15, tol_interval=[1.e-15, 1.e-2], type_norm='linf'):
    """Create a SolverHelper
    
    Arguments:
      ode: object of ODE type
      u0: initial condition
      t_interval: time interval of the form [ti, tf]
      integrator: type of integrator (must be supported by solve_ivp from scipy.integrate)
      integrator_e: type of the integrator which is considered as exact(must be supported by solve_ivp from scipy.integrate)
      tol_e: accuracy of internal parameters atol and rtol from solve_ivp for exact integrator
      tol_interval: range of accuracy of internal parameters atol and rtol for selected ODE integrator
      type_norm: examine error in norm type_norm (only 'linf' or 'l2' are supported)
    """

    self.ode   = ode
    self.u0    = u0
    [self.ti, self.tf] = t_interval
    self.integrator = integrator
    self.integrator_e = integrator_e
    [self.tol_min, self.tol_max] = tol_interval
    self.tol_e = tol_e
    self.exact = Propagator(self.ode, self.u0, [self.ti, self.tf], integrator=self.integrator_e, tol=tol_e)

    self.type_norm = type_norm
    self.tol_span, self.eps_span, self.cost_span, self.tol_to_eps = \
      self.compute_tol_to_eps(self.integrator, self.type_norm, n_tol=20)

  def compute_tol_to_eps(self, integrator, type_norm, n_tol=30):
    """Compute callable function relating tol from solve_ivp to global accuracy eps."""

    if type_norm=='l2':
      aggregate = sum
      self.type_norm = 'l2'
    elif type_norm=='linf':
      aggregate = max
      self.type_norm = 'linf'
    else:
      raise ValueError('Norm '+type_norm+' not supported in tol_to_eps.')

    t = self.exact.ivp.t
    u_exact = self.exact.eval(t)
    tol_span = np.logspace(np.log10(self.tol_min), np.log10(self.tol_max), num=n_tol, base=10, endpoint=True)

    eps_span = list()
    cost_span = list()
    for tol in tol_span:
      p = Propagator(self.ode, self.u0, [self.ti, self.tf], integrator=integrator, tol=tol)
      u = p.eval(t)

      cost_span.append(p.compute_cost())
      eps_span.append(aggregate(np.linalg.norm(u-u_exact, axis=0)))

    return tol_span, eps_span, cost_span, CubicSpline(tol_span, eps_span)

  def eps_to_tol(self, eps):
    """Relate tol from solve_ivp to global accuracy eps."""

    tol_span = np.logspace(np.log10(self.tol_e), 0., num=1000, base=10, endpoint=True)
    eps_span = self.tol_to_eps(tol_span)
    idx = np.argmax(eps_span - eps > 0.)
    if idx > 0:
      idx = idx-1
    return tol_span[idx]


class Parareal_Algorithm():
  """Parareal_Algorithm to solve an ODE."""

  def __init__(self, ode, u0, t_interval, N,
    integrator_g='RK45', integrator_f='Radau', eps_g = 1.e-1, compute_sh=True):
    """Create a Parareal_Algorithm
    
    Arguments:
      ode: object of ODE type
      u0: initial condition
      t_interval: time interval of the form [ti, tf]
      integrator_g: type of integrator for coarse propagator (must be supported by solve_ivp from scipy.integrate)
      integrator_f: type of integrator for fine propagator (must be supported by solve_ivp from scipy.integrate)
      integrator_e: type of the integrator which is considered as exact(must be supported by solve_ivp from scipy.integrate)
      eps_g: global accuracy of coarse propagator
      compute_sh: compute solver helper
    """

    self.ode   = ode
    self.u0    = u0
    [self.ti, self.tf] = t_interval
    self.N = N # Number of processors

    self.integrator_g = integrator_g
    self.integrator_f = integrator_f

    T_formatted = '{:d}'.format(int(self.tf))
    if os.path.isfile(ode.name()+'/T_'+T_formatted+'-sh_g.p') and os.path.isfile(ode.name()+'/T_'+T_formatted+'-sh_g.p'):
      # Load SolverHelper
      self.sh_g = pickle.load(open(ode.name()+ '/T_'+T_formatted+'-sh_g.p', 'rb'))
      self.sh_f = pickle.load(open(ode.name()+ '/T_'+T_formatted+'-sh_f.p', 'rb'))
    else: # Compute SolverHelper from scratch
      self.sh_g = SolverHelper(ode, u0, t_interval, integrator=integrator_g, integrator_e = integrator_f, tol_e=1.e-13, tol_interval=[1.e-13, 1.e-2], type_norm='linf')
      self.sh_f = SolverHelper(ode, u0, t_interval, integrator=integrator_f, integrator_e = integrator_f, tol_e=1.e-13, tol_interval=[1.e-13, 1.e-2], type_norm='linf')

      pickle.dump(self.sh_g, open(ode.name()+ '/T_'+T_formatted+'-sh_g.p', 'wb'))
      pickle.dump(self.sh_f, open(ode.name()+ '/T_'+T_formatted+'-sh_f.p', 'wb'))

    self.eps_g = eps_g
    self.tol_g = self.sh_g.eps_to_tol(eps_g)

    self.exact = self.sh_f.exact

    self.tl = None
    self.pl = None
    self.fl = None
    self.gl = None

    self.cost = {
      'cpu_time': 0.,
      't_steps':  0.,
      'n_rhs':    0., # number of evaluations of rhs
      'n_jac':    0., # Number of evaluations of the Jacobian
      'nlu':      0., # Number of lu decompositions
    }

  def run(self, eps, kmax=15, adaptive=True, balance_tasks=False, kth=0):
    """Run a Parareal_Algorithm
    
    Arguments:
      eps: global target accuracy
      kmax: maximum number of iterations
      adaptive: set to False if classical non adaptive parareal. True if adaptive parareal.
      balance_tasks: set to True, it activates task balancing in adaptive parareal.
      kth:
    """

    print()
    print('Running Parareal Algorithm.')
    print('===========================')
    print('Time interval = ['+str(self.ti)+','+str(self.tf)+']')
    print('Number of processors = '+str(self.N))
    print('Adaptivity: '+str(adaptive))
    print('Target accuracy = '+str(eps))


    # List of coarse, fine, parareal prop for each parareal iter k.
    # List of t_span
    gl = list()
    fl = list()
    pl = list()
    tl = list()

    eps_g = self.eps_g
    print('eps_g = '+str(self.eps_g))

    k = 0
    err = 1.
    while err > eps and k < kmax:

      if k == 0:
        # First guess for macro-intervals
        if balance_tasks:
          # Balancing with a priori information
          t_span = self.balance_tasks_with_exact_solver()
        else:
          t_span = np.linspace(self.ti, self.tf, num=self.N+1, endpoint=True)
        tl.append(t_span)

        # Sequential propagation of coarse solver
        u_span_g = [ self.u0 ]
        for i, t_interval in enumerate(zip(t_span[:-1], t_span[1:])):
          gi = Propagator(self.ode, u_span_g[-1], t_interval, integrator=self.integrator_g, tol=self.tol_g)
          u_span_g.append( gi.eval(t_interval[-1]) )
        u_span_g = np.asarray(u_span_g).T
        g0 = Piecewise_Propagator(self.ode, self.u0, tl[-1], u_span_g, integrator=self.integrator_g, tol=self.tol_g)

        gl.append(g0)
        pl.append(g0)

        err = max(np.linalg.norm(
          pl[-1].eval(t_span) - self.exact.eval(t_span), axis=0))
        print('Iteration k='+str(k)+'. Err = '+str(err))

      else:
        # Fine propagator (k-1)
        if adaptive:

          # # Update eps_f following our paper (bounds are too coarse)
          # nuk = self.estimate_nu_k(pl[-1])
          # eps_f = eps_g **(k+2)/ (nuk * factorial(k+1))

          # Update eps_f with an intuitive rule
          if k < kth:
            eps_f = np.logspace(np.log10(self.eps_g), np.log10(eps/COEF_EXACT), num=kth, endpoint=True)[k]
          else:
            eps_f = eps/COEF_EXACT
          tol_f = self.sh_f.eps_to_tol(eps_f)
        else:
          tol_f = self.sh_f.eps_to_tol(eps/COEF_EXACT)

        fkprev = Piecewise_Propagator(self.ode, self.u0, tl[-1], pl[-1].eval(tl[-1]), integrator=self.integrator_f, tol= tol_f)
        fl.append(fkprev)

        # Update t_span
        if balance_tasks:
          # Balancing with a priori information
          t_span = self.balance_tasks_with_exact_solver()

          # Dynamic balancing
          # t_span = self.balance_tasks(fl[-1])
        else:
          t_span = np.linspace(self.ti, self.tf, num=self.N+1, endpoint=True)
        t_span = self.balance_tasks_with_exact_solver()
        tl.append(t_span)

        # Parareal solution
        pk_u_span = [ self.u0 ]
        for i, t_interval in enumerate(zip(t_span[:-1], t_span[1:])):
          gi = Propagator(self.ode, pk_u_span[-1], t_interval, integrator=self.integrator_g, tol=self.tol_g)
          f = fkprev.propagator_span[i].eval(t_interval[-1])
          g = gl[-1].propagator_span[i].eval(t_interval[-1])
          pk_u_span.append( gi.eval(t_interval[-1]) + f - g )

        pk_u_span = np.asarray(pk_u_span).T
        gk = Piecewise_Propagator(self.ode, self.u0, tl[-1], pk_u_span, integrator=self.integrator_g, tol=self.tol_g)
        gl.append(gk)

        # pk = Piecewise_Propagator(self.ode, self.u0, tl[-1], pk_u_span, integrator=self.integrator_f, tol=self.tol_f)
        pk = Piecewise_Propagator(self.ode, self.u0, tl[-1], pk_u_span, integrator=self.integrator_f, tol=tol_f)
        pl.append(pk)

        err = max(np.linalg.norm(
          pl[-1].eval(t_span) - self.exact.eval(t_span), axis=0))
        print('Iteration k='+str(k)+'. Err = '+str(err))

      # Update iteration index
      k += 1

    print('End parareal iterations.\n')

    self.tl = tl
    self.pl = pl
    self.fl = fl
    self.gl = gl

    return pl, fl, gl, k

  def estimate_nu_k(self, parareal_prop_k):
    t = parareal_prop_k.t_span
    u_k = parareal_prop_k.u_span
    u_exact = self.exact.eval(t)
    return np.max(1 + np.linalg.norm(u_k, axis=0)) / np.max(1 + np.linalg.norm(u_exact, axis=0))

  def balance_tasks_with_exact_solver(self):
    which_idxs = lambda m, n: np.rint(
      np.linspace(1,n, min(m,n))-1).astype(int)

    tl = self.exact.ivp.t
    t_span = np.array(tl)[which_idxs(self.N+1,len(tl))]
    return t_span

  def balance_tasks(self, parareal_sol):
    tl = list()
    for prop in parareal_sol.propagator_span:
      tl.extend(prop.ivp.t[:-1].tolist())
    tl.append(self.tf)

    which_idxs = lambda m, n: np.rint(np.linspace(1,n, min(m,n))-1) \
                                .astype(int)
    t_span = np.array(tl)[which_idxs(self.N+1,len(tl))]
    return t_span

  def compute_cost(self, eps):
    keys = self.gl[0].propagator_span[0].cost.keys()
    # Add cost of all coarse propagations
    cgl = [g.compute_cost(type='sequential') for g in self.gl]
    cost_g = {key: sum(c[key] for c in cgl) for key in keys}

    # Add cost of fine propagators
    cfl = [f.compute_cost(type='parallel') for f in self.fl]
    cost_f = {key: sum(c[key] for c in cfl) for key in keys}

    # Total cost of parareal algorithm
    cl = [cost_g, cost_f]
    cost_parareal = {key: sum(c[key] for c in cl) for key in keys}

    # Cost sequential fine solver
    tol = self.sh_f.eps_to_tol(eps/COEF_EXACT)
    cost_seq_fine = Propagator(self.ode, self.u0, [self.ti, self.tf], integrator=self.integrator_f, tol=tol).cost

    return cost_g, cost_f, cost_parareal, cost_seq_fine

  def compute_error(self):

    err = list()
    t = self.exact.ivp.t

    for k, pk in enumerate(self.pl):
      # t = self.tl[k]
      u_pk = pk.eval(t)
      u_exact =  self.exact.eval(t)
      err.append(np.linalg.norm(u_pk - u_exact, axis=0))

    return err

  def plot_cost_detail(self, folder_name, key='t_steps'):
    """
      Plot of cost for each macro-interval
    """

    plt.figure()
    for k, g in enumerate(self.gl):
      cost_g = list()
      for gi in g.propagator_span:
        cost_g.append(gi.cost[key])
      plt.semilogy(cost_g, 'o-', label='coarse, k='+str(k))
    plt.legend()
    plt.savefig(folder_name+'cost_detail_g.pdf')

    plt.figure()
    for k, f in enumerate(self.fl):
      cost_f = list()
      for fi in f.propagator_span:
        cost_f.append(fi.cost[key])
      plt.semilogy(cost_f, 'o-', label='fine, k='+str(k))
    plt.legend()
    plt.savefig(folder_name+'cost_detail_f.pdf')


  def plot_eps_to_key(self, folder_name, key='cpu_time'):

    if key not in self.cost.keys():
      raise ValueError('Invalid key in plot_eps_to_cost')

    # Fine solver
    plt.figure()
    plt.style.use('classic')
    cost_cpu = [ c[key] for c in self.sh_g.cost_span ]
    plt.loglog(self.sh_f.eps_span, cost_cpu, label=key)
    plt.gca().yaxis.grid(True)
    plt.xlabel(r'$\varepsilon$', fontsize=20)
    plt.legend()
    plt.savefig(folder_name+'eps_to_'+key+'_f.pdf')
    plt.close()

    # Coarse solver
    plt.figure()
    plt.style.use('classic')
    cost_cpu = [ c[key] for c in self.sh_g.cost_span ]
    plt.loglog(self.sh_g.eps_span, cost_cpu, label=key)
    plt.gca().yaxis.grid(True)
    plt.xlabel(r'$\varepsilon$', fontsize=20)
    plt.legend()
    plt.savefig(folder_name+'eps_to_'+key+'_g.pdf')
    plt.close()

  def plot_tol_to_eps(self, folder_name):

    plt.figure()
    plt.style.use('classic')
    plt.semilogx(self.sh_f.tol_span, self.sh_f.eps_span, 'o', label='computed')
    tol_span_fine = np.logspace(np.log10(self.sh_f.tol_e), np.log10(self.sh_f.tol_max), num=1000, base=10, endpoint=True)
    plt.loglog(tol_span_fine, self.sh_f.tol_to_eps(tol_span_fine), label='interp. spline')
    plt.xlabel('tolerance param', fontsize=20)
    plt.ylabel(r'$\varepsilon$', fontsize=20)
    plt.legend()
    plt.savefig(folder_name+'tol_to_eps_f.pdf')
    plt.close()

    plt.figure()
    plt.style.use('classic')
    plt.semilogx(self.sh_g.tol_span, self.sh_g.eps_span, 'o', label='computed')
    tol_span_fine = np.logspace(np.log10(self.sh_g.tol_e), 0., num=100, base=10, endpoint=True)
    plt.loglog(tol_span_fine, self.sh_g.tol_to_eps(tol_span_fine), label='interp. spline')
    plt.xlabel('tolerance param', fontsize=20)
    plt.ylabel(r'$\varepsilon$', fontsize=20)
    plt.legend()
    plt.savefig(folder_name+'tol_to_eps_g.pdf')
    plt.close()






