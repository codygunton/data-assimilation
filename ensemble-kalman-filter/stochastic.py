import scipy.integrate
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# global parameters
dt = 0.001                  # time step between observations
N = 40                      # number of state variables
F = 8                       # lorenz forcing term

# runge-kutta parameters
h = 0.001                   # time step

x0 = np.zeros([1, N])
x0
x0[0, 19] += 0.1
# x0 = x[-1]

RK_times = np.arange(h, 10.0, h)  # number of time steps


def f(A, t):  # A a numpy array
    return (np.roll(A, 1) - np.roll(A, -2)) * np.roll(A, -1) - A + F


x = scipy.integrate.odeint(f, x0[0], RK_times)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(x[:, 0], x[:, 1], x[:, 2])
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$x_3$')
plt.show()


states = x0
for tn in RK_times:
    xn = states[-1]
    k1 = f(xn, tn)
    k2 = f(xn + h*k1/2, tn + h/2)
    k3 = f(xn + h*k2/2, tn + h/2)
    k4 = f(xn + h*k3, tn + h)
    xnew = xn + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    xnew = np.reshape(xnew, (1, N))
    states = np.concatenate((states, xnew), axis=0)


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(states[:, 0], states[:, 1], states[:, 2])
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$x_3$')
plt.show()


diff = x - states[:-1]
sum(diff[9000]>1)
