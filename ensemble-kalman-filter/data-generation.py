import numpy as np
import time
import cProfile

# global parameters
N = 40             # number of state variables
F = 8              # lorenz forcing term


# functions defining lorenz system
def f(A):
    out = [(A[(i+1) % N] - A[(i-2) % N]) * A[(i-1) % N] - A[i] + F
           for i in range(len(A))]
    return np.array(out)


def RK_step(x, h):
    k1 = f(x)
    k2 = f(x + h*k1/2)
    k3 = f(x + h*k2/2)
    k4 = f(x + h*k3)
    return x + (h/6)*(k1 + 2*k2 + 2*k3 + k4)


# standard_basis = []
# for i in range(N):
#     e = np.zeros(N)
#     np.put(e, i, 1)
#     standard_basis += [e]


# for times in RK_times (1-D array), solve system with initial state x0
# returns an array of shape (len(RK_times), N), N = dim x0
def L96(x0, T, h):
    RK_times = np.arange(h, T, h)  # number of time steps
    states = [x0]
    for tn in RK_times:
        xn = states[-1]
        states += [RK_step(xn, h)]
    return states


# get good initial position near attractor
T = 100                   # length of simulation
h = 0.005                 # time step

np.random.seed(2017)
x0 = np.array([np.random.normal() for i in range(N)])
states = L96(x0, T, h)
np.save("starting-data", states[-1])


# get true states to generate synthetic data
T = 10
h = 0.005                         # time step
x0 = np.load("starting-data.npy")
cProfile.run('states = L96(x0, T, h)')
states = np.array(states)
np.save("true-states", states)

# plot true states
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(states[:, 0], states[:, 1], states[:, 2])
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$x_3$')
plt.show()

# generate synthetic data
# only observe every 0.1 seconds, so keep only every nth state, n=0.1/h
obs = np.array(states[::int(0.1//h)])
# ???so I should use M^n later???
# for each state, kill obs of even-numbered variables

H = np.zeros(N)
np.put(H, range(1, N, 2), 1)  # we observe odd-indexed variables
R = np.identity(N)

# this is inefficient; could flatten against 0s
# but I want to write in terms of R
np.random.seed(2017) 
epsilon = np.array([np.random.multivariate_normal(np.zeros(N), R)
                    for y in obs])
obs = obs + epsilon
for y in obs:                    # maybe i don't want to do this??
    np.put(y, range(0, N, 2), 0)
np.save("synthetic-obs", obs)


# generate initial ensemble
T = 1000
h = 0.005
x0 = np.load("starting-data.npy")
states = L96(x0, T, h)  # runs in 85s on my laptop

Ne = 40
ens = [states[np.random.randint(len(states))] for i in range(Ne)]
np.save("initial-ens", ens)
