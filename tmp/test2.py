import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from odeSolveMethods import RungeKutta as RK4
from time import time, sleep
# Logistic function
r = 5

def func(t):
    return 1. / (1 + np.exp(-r * t))

def func2(t):
    return 1. / np.exp(-r * t)

def dfunc(t, X, args = None):
    x,  = X
    return [r * x * (1 - x)]

def dfunc2(t, X):
    x, = X
    return [-r * x * x]

t0 = -10
t1 = 10
dt = 0.01
X0 = func(t0)

integrator = 'vode'

t = [t0]
Xi = [X0]
t1_t = time()
ode = integrate.ode(dfunc)
ode.set_integrator(integrator, atol=1e-8, rtol=1e-8, method='bdf')
ode.set_initial_value(X0, t0)
print('here')
while ode.successful() and ode.t < t1:
    t.append(ode.t)
    print(t[-1])
    Xi.append(ode.integrate(t1, step=True))
t2 = time()
print('time bdf: ', t2-t1_t, len(t))

t = np.array(t)     # Time
X = func2(np.linspace(t0, t1, 100000))         # Solution
print(np.linspace(t0, t1, 100000).shape, X.shape)
Xi = np.array(Xi)   # Numerical

t1_t = time()
Tnke4, Ynke4 = RK4(f=dfunc, args=None, tspan=[t0, t1], u0=X0, t_step=0.01)
t2 = time()
print('time RK4:  ', t2-t1_t, len(Tnke4))





plt.plot(np.linspace(t0, t1, 100000), X, label="analytical", color='g')
plt.plot(t, Xi, label=integrator)
plt.plot(Tnke4, Ynke4, label='RK4')
plt.xlabel("t")
plt.ylabel("x")
plt.title("Solution")
plt.legend(loc=0)
plt.show()