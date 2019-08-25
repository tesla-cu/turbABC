
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from scipy.integrate import odeint
S_periodic = np.zeros(6)
from time import time, sleep
import os
from odeSolveMethods import RungeKuttaFehlberg as RK
from odeSolveMethods import RungeKutta as RK4
import rans
import glob_var as g

folder = './valid_data'
period1_k = np.loadtxt(os.path.join(folder, 'period1_k.txt'))

c = [2.3, 0.67, 2.05, 2.34]

# c = [1., 4/3+0.0001, 2.055, 3.5]
c = [1.0, 4/3+0.5, 2.055, 3.5]
s0 = 3.3
beta = [0.125, 0.25, 0.5, 0.75, 1]

err = np.zeros(5)
t_steps = period1_k[:, 0]/s0
# Periodic shear(five different frequencies)
tspan = [0, 51/s0]
t1 = time()

Tnke, Ynke = RK(f=rans.rans_periodic, tspan=tspan, u0=[1, 1, 0, 0, 0, 0, 0, 0],
                t_step=1, args=(c, s0, beta[0]), TOL=1e-8)
t2 = time()
print("%-6s %-8s time=%-15s " % ('my', 'RKFehlberg', t2-t1), end='')
print("len(tvals) =", len(Tnke))
##############################################
# t1 = time()
# Tnke4, Ynke4 = RK4(f=rans.rans_periodic, tspan=tspan, u0=[1, 1, 0, 0, 0, 0, 0, 0],
#                 t_step=0.1, args=(c, s0, beta[0]), TOL=1e-8)
# t2 = time()
# print("%-6s %-8s time=%-15s " % ('my', 'RK4', t2-t1), end='')
# print("len(tvals) =", len(Tnke4))
##############################################
args = (c, s0, beta[0])
t1 = time()
sol, info = odeint(rans.rans_periodic, [1, 1, 0, 0, 0, 0, 0, 0], t_steps,
                   tfirst=True, args=(args, ), atol=1e-8, printmessg=True,
                   full_output=True)
t2 = time()
print("%-6s %-8s time=%-15s " % ('odeint', ' ', t2-t1), end='')
print("len(tvals) =", len(sol))
print(info['hu'])
print(info['nfe'])
##############################################
plt.plot(t_steps, sol[:, 0])
plt.plot(Tnke, Ynke[:, 0])
# plt.plot(Tnke4, Ynke4[:, 0])
plt.scatter(period1_k[:, 0]/s0, period1_k[:, 1])
plt.ylabel('k')
plt.show()
plt.plot(t_steps, sol[:, 1])
plt.plot(Tnke, Ynke[:, 1])
# plt.plot(Tnke4, Ynke4[:, 1])
plt.ylabel('epsilon')
plt.show()
plt.plot(Tnke, Ynke[:, 5], label='RKFehlberg')
# plt.plot(Tnke4, Ynke4[:, 5], label='RK4')
plt.plot(t_steps, sol[:, 5], label='odeint')

# for name, kwargs in [
#     ('vode', dict(method='adams')),
#                      ('vode', dict(method='bdf')),
#                      ('lsoda', {}),
#                      ('dopri', {})]:
#     t1 = time()
#     ts, ys = [], []
#     solver = ode(rans.rans_periodic, jac=None)
#     solver.set_integrator(name, atol=1e-8, rtol=1e-6, **kwargs)
#     solver.set_f_params((c, s0, beta[0]))
#     solver.set_initial_value([1, 1, 0, 0, 0, 0, 0, 0], 0)
#
#     g.tvals = []
#     i = 0
#     while solver.successful() and solver.t < tspan[-1]:
#         ts.append(solver.t)
#         ys.append(solver.integrate(tspan[-1], step=True))
#         i += 1
#     t2 = time()
#     print("%-6s %-8s time=%-15s " % (name, kwargs.get('method', ''), t2-t1), end='')
#     g.tvals = np.unique(g.tvals)
#     print("len(tvals) =", len(g.tvals))
#     plt.plot(ts, np.array(ys)[:, 5], label='{} {}'.format(name, kwargs.get('method', '')))
plt.legend()
plt.ylabel('a')
plt.show()