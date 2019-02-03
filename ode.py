import numpy as np
from scipy.integrate import ode, solve_ivp
import time


class VDP():
    """
        Van der Pool oscilator
    """

    def __init__(self, *args):
        self.mu = args[0]

    @staticmethod
    def name():
        return 'VDP'

    def info(self):
        s = 'Problem\n'
        s += '=========\n'
        s += 'Van der Pool oscillator with mu='+str(self.mu)
        s += '\n'
        return s

    def xlim(self):
        return [-2., 2.]

    def ylim(self):
        return [-3.5, 3.5]

    def f(self, t, u):
        return np.array([ u[1], self.mu*(1 - u[0]*u[0])*u[1] - u[0] ])

    def jac(self, t, u):
        j = np.empty((2, 2))
        j[0, 0] = 0.0
        j[0, 1] = 1.0
        j[1, 0] = -self.mu*2*u[0]*u[1] - 1
        j[1, 1] = self.mu*(1 - u[0]*u[0])
        return np.array([[0., 1.],[-self.mu*2*u[0]*u[1]-1., self.mu*(1 - u[0]*u[0])]])

class Brusselator():
    """
        Brusselator
    """

    def __init__(self, *args):
        self.A = float(args[0])
        self.B = float(args[1])

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
        return [-0.5, 4.]

    def ylim(self):
        return [-0.5, 5.5]

    def f(self, t, u):
        x = u[0]
        y = u[1]
        return np.asarray([self.A+(x**2)*y-(self.B+1.)*x, self.B*x - (x**2)*y])

    def jac(self, t, u):
        x = u[0]
        y = u[1]
        j = np.empty((2, 2))
        j[0, 0] = 2*x*y-(self.B+1.)
        j[0, 1] = x**2
        j[1, 0] = self.B - 2*x*y
        j[1, 1] = x**2
        return j

class Oregonator():
    """
        Oregonator
    """

    def __init__(self, *args):
        pass

    @staticmethod
    def name():
        return 'Oregonator'

    def info(self):
        s = 'Problem\n'
        s += '=========\n'
        s += 'Oregonator'
        s += '\n'
        return s

    def info(self):
        return 'Oregonator'

    def f(self, t, u):
        f = np.zeros(3)
        f[0] = 77.27*(u[1]+u[0]*(1.-8.375e-6*u[0]-u[1]))
        f[1] = (1./77.27)*(u[2]-(1+u[0])*u[1])
        f[2] = 0.161*(u[0]-u[2])
        return f

    def jac(self, t, u):
        j = np.empty((3, 3))
        j[0, 0] = 77.27*( (1.-8.375e-6*u[0]-u[1]) -8.375e-6*u[0])
        j[0, 1] = 77.27*( 1.-u[0] )
        j[0, 2] = 0.
        j[1, 0] = (1./77.27)*(-u[1])
        j[1, 1] = (1./77.27)*(-(1+u[0]))
        j[1, 2] = (1./77.27)
        j[2, 0] = 0.161
        j[2, 1] = 0.
        j[2, 2] = -0.161
        return j 


def propagate(ode, u0, ti, tf, method, atol=1.e-8, rtol=1.e-13, **kwargs):

    start = time.time()
    solver = solve_ivp(ode.f, [ti, tf], u0, jac = ode.jac, method = method, atol=atol, rtol=rtol, dense_output=True)
    end = time.time()

    out = dict()
    out['cpu_time'] = end-start

    return solver, out