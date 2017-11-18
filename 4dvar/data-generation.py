lsimport numpy as np
import time
import cProfile

# global parameters
N = 40             # number of state variables
F = 8              # lorenz forcing term
h = 0.01           # RK time step


# function defining lorenz system and dervatives, and solver
# input: vector of shape N

# dx/dt = f(x)
def f(x):
    out = [(x[(i+1) % N] - x[(i-2) % N]) * x[(i-1) % N] - x[i] + F
           for i in range(len(x))]
    return np.array(out)


def Euler_step(x, h):
    return x + f(x) * h


# for times in RK_times (1-D array), solve system with initial state x0
def L96_Euler(x0, T, h):
    Euler_times = np.arange(h, T, h)  # number of time steps
    states = [x0]
    for t in Euler_times:
        x = states[-1]
        states += [Euler_step(x, h)]
    return states


# Test that L96_Euler is working
# T = 10
# x0 = np.load("initial-position.npy")
# states = L96_Euler(x0, T, h)
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


# get good initial position near attractor
t0 = time.time()
T = 5000                   # propagate for T seconds
np.random.seed(2017)
x0 = np.array([np.random.normal() for i in range(N)])
states = L96_Euler(x0, T, h)
np.save("initial-position", states[-1])
tf = time.time()
print("Initial position generated. Time: {0}".format(tf-t0))


def generate_states(T):
    # get true states to generate synthetic data
    t0 = time.time()
    x0 = np.load("initial-position.npy")

    states = L96_Euler(x0, T, h)
    states = np.array(states)
    np.save("./true-states/true-states-T{0}".format(T), states)
    tf = time.time()
    print("True states generated. Time: {0}".format(tf-t0))


def generate_data(T):
    # generate synthetic data
    # only observe every obs_gap seconds,
    # so keep only every nth state, n = obs_gap/h
    t0 = time.time()
    obs_gap = 0.2
    n = int(obs_gap // h)
    states = np.load("./true-states/true-states-T{0}.npy".format(T))
    obs = np.array(states[::n])
    obs = list(map(lambda A: np.delete(A, range(0, N, 2)), obs))
    obs = np.array(obs)
    # this is inefficient; could flatten against 0s
    # but I want to write in terms of R
    np.random.seed(2016)
    R = np.identity(N//2)
    epsilon = np.array([np.random.multivariate_normal(np.zeros(N//2), R)
                        for y in obs])
    obs = obs + epsilon
    np.save("./synthetic-observations/synthetic-obs-T{0}"
            .format(T), obs)
    tf = time.time()
    print("Synthetic data generated. Time: {0}".format(tf-t0))


for T in [10, 20, 50, 100, 500, 1000]:
    generate_states(T)
    generate_data(T)


# generate initial ensembles
t0 = time.time()
T = 2000
x0 = np.load("initial-position.npy")
states = L96_Euler(x0, T, h)  # runs in 85s on my laptop
np.save("large-run-for-sampling", states)

for Ne in [20, 40, 100, 200, 500]:
    ens = [states[np.random.randint(len(states))] for i in range(Ne)]
    np.save("./initial-ensembles/initial-ens-{0}".format(Ne), ens)

tf = time.time()
print("Inital ensembles generated. Time: {0}".format(tf-t0))

t0 = time.time()
initial_mean = np.mean(states, axis=0)
initial_cov = (1/(len(states)-1)) * sum([np.outer((e - initial_mean),
                                                  (e - initial_mean))
                                         for e in states])
np.save("initial_mean", initial_mean)
np.save("initial_cov", initial_cov)
t1 = time.time()
print(t1-t0)
