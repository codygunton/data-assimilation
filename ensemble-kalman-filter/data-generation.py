import numpy as np
import time
import cProfile

# global parameters
N = 40             # number of state variables
F = 8              # lorenz forcing term


# function defining lorenz system
def f(A, t):  # A a numpy array of shape N
    return (np.roll(A, 1) - np.roll(A, -2)) * np.roll(A, -1) - A + F


def g(A):  # A a numpy array of shape N
    return (np.roll(A, 1) - np.roll(A, -2)) * np.roll(A, -1) - A + F


def g2(A):
    out = []
    for i in range(len(A)):
        out += [(A[(i+1)%N] - A[(i-2)%N]) * A[(i-1)%N] - A[i] + F]
    return np.array(out)

def RK_function(x):
    k1 = g2(x)
    k2 = g2(x + h*k1/2)
    k3 = g2(x + h*k2/2)
    k4 = g2(x + h*k3)
    return x + (h/6)*(k1 + 2*k2 + 2*k3 + k4)


standard_basis = []
for i in range(N):
    e = np.zeros(N)
    np.put(e, i, 1)
    standard_basis += [e]


def RK_matrix(N):
    standard_basis = []
    for i in range(N):
        e = np.zeros(N)
        np.put(e, i, 1)
        standard_basis += [e]
    M_cols = []
    for e in standard_basis:
        M_cols += [RK_function(e)]
    return np.matrix(M_cols).T


M = RK_matrix(N)


# for times in RK_times (1-D array), solve system with initial state x0
# returns an array of shape (len(RK_times), N), N = dim x0
def L96(x0, RK_times):
    states = [x0]
    for tn in RK_times:
        xn = states[-1]
        states += [RK_function(xn)]
    return states


# get good initial position near attractor
h = 0.005                 # time step
np.random.seed(2017)
x0 = np.array([np.random.normal() for i in range(N)])
RK_times = np.arange(h, 100.0, h)  # number of time steps
cProfile.run('states = L96(x0, RK_times)')
states = L96(x0, RK_times)
np.save("starting-data", states[-1])


# get true states to generate synthetic data
h = 0.005                         # time step
x0 = np.load("starting-data.npy")
RK_times = np.arange(h, 10.0, h)  # number of time steps
states = L96(x0, RK_times)

# plot true states
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
states = np.array(states)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(states[:, 0], states[:, 1], states[:, 2])
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$x_3$')
plt.show()

# generate synthetic data
# only observe every 0.1 seconds, so keep only every nth state, n=0.1/h
obs = states[::int(0.1//h)]
# ???so I should use M^n later???
# for each state, keep only obs of odd-numbered variables
obs = np.delete(obs, range(0, N, 2), axis=1)
np.random.seed(2017)
epsilon = np.array([np.random.multivariate_normal(
    np.zeros(N//2), np.identity(N//2)) for o in obs])
obs = obs + epsilon
np.save("synthetic-obs", obs)


# generate initial ensemble
h = 0.005                         # time step
x0 = np.load("starting-data.npy")
RK_times = np.arange(h, 1000.0, h)  # number of time steps
cProfile.run('states = L96(x0, RK_times)')
print("time:", t1-t0)


# generate initial ensemble
h = 0.005                         # time step
x0 = np.load("starting-data.npy")
RK_times = np.arange(h, 100.0, h)  # number of time steps
cProfile.run('states = L96(x0, RK_times)')
print("time:", t1-t0)
