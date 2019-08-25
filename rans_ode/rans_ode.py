import numpy as np
import workfunc_rans
import pyabc.glob_var as g


def rans_impulsive(x, t, c, S):
    """
    :param x:
    :return:
    """
    k = x[0]                # turbulence kinetic energy
    e = x[1]                # dissipation rate
    a = np.array(x[2:])     # anisotropy tensor

    P = -k * (np.sum(a * S) + np.sum(a[3:] * S[3:]))

    alf1 = P / e - 1 + c[0]
    alf2 = c[1] - 4 / 3

    # Governing equations
    dx = np.zeros(8)
    dx[0] = P - e                               # dk / dt
    dx[1] = (c[2] * P - c[3] * e) * e / k       # de / dt
    dx[2:] = -alf1 * e * a / k + alf2 * S       # d a_ij / dt
    return dx


def rans_periodic(x, t, c, s0, beta):
    """

    :param x:
    :return:
    """
    S = workfunc_rans.StrainTensor.periodic_strain(t, s0, beta)    # applied periodic shear

    k = x[0]                # turbulence kinetic energy
    e = x[1]                # dissipation rate
    a = np.array(x[2:])     # anisotropy tensor

    P = -k * (np.sum(a * S) + np.sum(a[3:] * S[3:]))
    alf1 = P / e - 1 + c[0]
    alf2 = c[1] - 4 / 3
    # Governing equations
    dx = np.zeros(8)
    dx[0] = P - e                               # dk / dt
    dx[1] = (c[2] * P - c[3] * e) * e / k       # de / dt
    dx[2:] = -alf1 * e * a / k + alf2 * S       # d a_ij / dt
    return dx


def rans_decay(x, t, c):
    """
    :param t:
    :param x:
    :param args:
    :return:
    """
    k = x[0]                # turbulence kinetic energy
    e = x[1]                # dissipation rate
    a = np.array(x[2:])     # anisotropy tensor

    alf1 = -1 + c[0]
    # Governing equations
    dx = np.zeros(8)
    dx[0] = -e                  # dk / dt     dk / dt = -ka_{ij}S_{ij} - e
    dx[1] = -c[3] * e**2 / k    # de / dt     de / dt =(-kC_{e1}a_{ij}S_{ij}-C_{e2}e)*e/k
    dx[2:] = -alf1 * e * a / k
    return dx


def rans_strain_relax(x, t, c):

    S = g.Strain.strain_relax(t)

    k = x[0]                # turbulence kinetic energy
    e = x[1]                # dissipation rate
    a = np.array(x[2:])

    P = -k * (np.sum(a * S) + np.sum(a[3:] * S[3:]))    # kinetic energy production P = -k * a_{ij} * S_{ij}

    alf1 = (P / e - 1 + c[0])
    alf2 = (c[1] - 4 / 3)

    # Governing equations
    dx = np.zeros(8)
    dx[0] = P - e                           # dk / dt = -ka_{ij}S_{ij} - e
    dx[1] = (c[2] * P - c[3] * e) * e / k   # de / dt = (-kC_{e1}a_{ij}S_{ij}-C_{e2}e) * e / k
    dx[2:] = -alf1 * a * e / k + alf2 * S
    return dx
