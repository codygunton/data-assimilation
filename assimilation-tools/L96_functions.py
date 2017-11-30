import numpy as np
from globals_etc import F


# dx/dt = f(x)
def f(x):
    N = x.shape[0]
    out = [(x[(i+1) % N] - x[(i-2) % N]) * x[(i-1) % N] - x[i] + F
           for i in range(N)]
    return np.array(out)


# the Jacobian of f
def Df(x):
    N = len(x)
    out = []
    for i in range(N):
        row = np.zeros(N)
        indices = [i-2, i-1, i, i+1]
        insert = [-x[(i-1) % N],
                  x[(i+1) % N] - x[(i-2) % N],
                  -1, x[(i-1) % N]]
        np.put(row, indices, insert, mode='wrap')
        out += [row]
    return np.array(out)


def Eu_step(x, h):
    return x + f(x) * h


def M_Eu(x, h, n):
    for i in range(n):
        x = Eu_step(x, h)
    return x


def DM_Eu(x, h, n):  # total derivative of M_Eu
    N = len(x)
    dxidx0 = np.identity(N)
    xi = x
    for i in range(1, n+1):
        dxidx0 = (np.identity(N) + Df(xi)*h) @ dxidx0
        xi = M_Eu(x)
    return dxidx0


def RK_step(x, h):
    k1 = f(x)
    k2 = f(x + h*k1/2)
    k3 = f(x + h*k2/2)
    k4 = f(x + h*k3)
    return x + (h/6)*(k1 + 2*k2 + 2*k3 + k4)


def M_RK(x, h, n):
    for i in range(n):
        x = RK_step(x, h)
    return x


def DM_RK(x, h, n):  # total derivative of M_RK
    # def many redundant calculations if I'm running both RK and DM_RK
    N = len(x)
    dxidx0 = np.identity(N)
    xi = x
    for i in range(1, n+1):
        k1 = f(xi)
        k2 = f(xi + (h/2) * k1)
        k3 = f(xi + (h/2) * k2)
        k4 = f(xi + h * k3)

        dk1dx0 = Df(xi) @ dxidx0
        dk2dx0 = Df(xi + (h/2) * k1) * (h/2) @ dk1dx0
        dk3dx0 = Df(xi + (h/2) * k2) * (h/2) @ dk2dx0
        dk4dx0 = Df(xi + h * k3) * h @ dk3dx0

        xi = x + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        dxidx0 = dxidx0 + (h/6)*(dk1dx0 + 2*dk2dx0 + 2*dk3dx0 + dk4dx0)
    return dxidx0


# for times in Eu_times (1-D array), solve system with init state x0
def L96_Eu(x0, T, h):
    Eu_times = np.arange(0, T, h)
    # outputs a list of length T/h, including initial state
    states = [x0]
    x = x0
    for t in Eu_times:
        x = Eu_step(x, h)
        states += [x]
    return states


# for times in RK_times (1-D array), solve system with init state x0
def L96_RK(x0, T, h):
    RK_times = np.arange(0, T, h)
    # outputs a list of length T/h + 1, including initial state
    states = [x0]
    x = x0
    for t in RK_times:
        x = RK_step(x, h)
        states += [x]
    return states


# # test a method works by graphing is working
# T = 10
# x0 = np.load("/home/cody/initial-position.npy")
# states = L96_Eu(x0, T, h)
# states = np.array(states)

# # plot true first three true states
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot(states[:, 0], states[:, 1], states[:, 2])
# ax.set_xlabel('$x_1$')
# ax.set_ylabel('$x_2$')
# ax.set_zlabel('$x_3$')
# plt.show()

# # plot true state of i-th variable
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# i = 1
# fig = plt.figure()
# # ax = fig.gca(projection='3d')
# times = np.arange(0, T, h)
# plt.plot(times, states[:, i-1])
# # plt.set_xlabel('$x_{0}$'.format(i))
# plt.show()
