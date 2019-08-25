# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 10:15:50 2016

@author: olga
"""
import numpy as np
from scipy.integrate import ode, odeint

TINY = 1e-10
DOFIGA = 1e10


def RungeKutta(f, args, tspan, u0, t_step, TOL=None):
    """
    Solve the initial value problem u' = f(t,u) by Runge-Kutta method.
    tspan -- list of time interval ends
    u0 -- the initial conditions u(t0) = u0,
    t_step -- step by time.
    """

    def increment_4(f, t, u, t_step):
        k1 = t_step * f(t, u, args)
        k2 = t_step * f(t + t_step / 2., u + k1 / 2., args)
        k3 = t_step * f(t + t_step / 2., u + k2 / 2., args)
        k4 = t_step * f(t + t_step, u + k3, args)
        return (k1 + 2. * k2 + 2. * k3 + k4) / 6.
    
    def increment_3(f, t, u, t_step):
        k1 = t_step * f(t, u)
        k2 = t_step * f(t + t_step * 2 / 3., u + k1*2 / 3.)
        k3 = t_step * f(t + t_step * 2 / 3., u - k1 /3 + k2)
        return (2 * k1 + k2 + 2. * k3) / 4.
    
    def increment_Hoin(f, t, u, t_step):
        k1 = t_step * f(t, u)
        k2 = t_step * f(t + t_step / 3., u + k1 / 3.)
        k3 = t_step * f(t + t_step * 2 / 3., u + k2 * 2 / 3)
        return (k1 + 3*k3) / 4.
    
    t = []
    u = []
    t0, tEnd = tspan[0], tspan[-1]
    t.append(t0)
    u.append(u0)
    while abs(tEnd - t0) > TINY:
        t_step = min(t_step, tEnd - t0)
        u0 = u0 + increment_4(f, t0, u0, t_step)
        t0 = t0 + t_step
        t.append(t0)
        u.append(u0)
    return np.array(t), np.array(u)


def RungeKuttaFehlberg(f, args, tspan, u0, t_step, TOL=1e-8):
    """
    Algorithm 5.3  in Numerical analisys 9th Burden Faires
    One popular technique that uses Inequality for error control is the Runge-Kutta-Fehlberg method. This technique uses
    a Runge-Kutta method with local truncation error of order five, to estimate the local error in a Runge-Kutta method
    of order four. (An advantage to this method is that only six evaluations of f are required per step. Arbitrary
    Runge-Kutta methods of orders four and five used together require at least four evaluations of f for the
    fourth-order method and an additional six for the fifth-order method, for a total of at least ten function
    evaluations. So the Runge-Kutta-Fehlberg method has at least a 40% decrease in the number of function evaluations
    over the use of a pair of arbitrary fourth- and fifth-order methods.)

    R - differense between these two methods
    :param f: function on the right side of ODE
    :param args: additioanl arguments for f function, except t and u
    :param tspan: list of time interval ends
    :param u0: the initial conditions u(t0) = u0
    :param t_step: maximum step by time
    :param TOL: tolerance of simulation
    :return: array of time series and array of solution
    """

    t0, tEnd = tspan[0], tspan[-1]
    t = [t0]
    u = [u0]
    t_step_max = t_step
    t_step_min = 1e-8
    counter = 0
    while True:
        k1 = t_step * f(t0, u0, args)
        k2 = t_step * f(t0 + t_step / 4, u0 + k1 / 4, args)
        k3 = t_step * f(t0 + t_step * 3 / 8, u0 + k1 * 3 / 32 + k2 * 9 / 32, args)
        k4 = t_step * f(t0 + t_step * 12 / 13, u0 + k1 * 1932 / 2197 - k2 * 7200 / 2197 + k3 * 7296 / 2197, args)
        k5 = t_step * f(t0 + t_step, u0 + k1 * 439 / 216 - k2 * 8 + k3 * 3680 / 513 - k4 * 845 / 4104, args)
        k6 = t_step * f(t0 + t_step / 2, u0 - k1 * 8 / 27 + k2 * 2 - k3 * 3544 / 2565 + k4 * 1859 / 4104 - k5 * 11 / 40, args)
        R = np.max(1 / t_step * np.abs(1 / 360. * k1 - 128 / 4275 * k3 - 2197 / 75240 * k4 + 1 / 50 * k5 + 2 / 55 * k6))
        if R <= TOL:
            u0 = u0 + (25 / 216 * k1 + 1408 / 2565 * k3 + 2197 / 4104 * k4 - 1 / 5 * k5)
            t0 += t_step
            t.append(t0)
            u.append(u0)
            counter += 1

        # Calculate new t_step
        if R == 0:
            t_step = 4 * t_step
        else:
            delta = 0.84 * (TOL / R) ** 0.25
            if delta <= 0.1:
                t_step = 0.1 * t_step
            elif delta >= 4:
                t_step = 4 * t_step
            else:
                t_step = delta * t_step
        if t_step > t_step_max:
            t_step = t_step_max

        if t0 >= tEnd:
            return np.array(t), np.array(u)
        elif t_step < t_step_min:
            print("Minimum t_step exceeded with C={}".format(args[0]))
            return np.array(t), np.array(u)
        else:
            t_step = min(t_step, tEnd - t0)


def BDF(f, args, tspan, u0, t_step, TOL=1e-8):

    t = [tspan[0]]
    y = [u0]
    solver = ode(f)
    solver.set_integrator('vode', atol=TOL, rtol=1e-6, method='bdf')
    solver.set_f_params(args)
    solver.set_initial_value(u0, tspan[0])

    while solver.successful() and solver.t < tspan[1]:
        t.append(solver.t)
        y.append(solver.integrate(tspan[1], step=True))

    return np.array(t), np.array(y)


def ode_int(f, args, tspan, u0, TOL=1e-8):

    t = np.linspace(tspan[0], tspan[-1], 200)
    y = odeint(f, u0, tspan, args=(args,), atol=TOL)
    return np.array(tspan), np.array(y)