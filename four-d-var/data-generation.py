import numpy as np
import time


# global parameters
N = 40             # number of state variables
F = 8              # lorenz forcing term
h = 0.01           # Euler time step
obs_gap = 0.2      # time between observations
n = int(obs_gap // h)

# function defining lorenz system and dervatives, and solver
# input: vector of shape N


# dx/dt = f(x)
def f(x):
    out = [(x[(i+1) % N] - x[(i-2) % N]) * x[(i-1) % N] - x[i] + F
           for i in range(len(x))]
    return np.array(out)


def Euler_step(x, h):
    return x + f(x) * h


def RK_step(x, h):
    k1 = f(x)
    k2 = f(x + h*k1/2)
    k3 = f(x + h*k2/2)
    k4 = f(x + h*k3)
    return x + (h/6)*(k1 + 2*k2 + 2*k3 + k4)


# for times in Euler_times (1-D array), solve system with init state x0
def L96_Euler(x0, T, h):
    Euler_times = np.arange(h, T, h)
    # outputs a list of length T/h, including initial state
    states = [x0]
    for t in Euler_times:
        x = states[-1]
        states += [Euler_step(x, h)]
    return states


# for times in RK_times (1-D array), solve system with init state x0
def L96_RK(x0, T, h):
    RK_times = np.arange(0, T, h)  ### this used to start at
    # outputs a list of length T/h + 1, including initial state
    states = [x0]
    for t in RK_times:
        x = states[-1]
        states += [RK_step(x, h)]
    return states


# # test a method works by graphing is working
# T = 10
# x0 = np.load("/home/cody/initial-position.npy")
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


### ended on changing arange above; will have to change elsewhere

# get good initial position near attractor
t0 = time.time()
T = 5000                   # propagate for T seconds
np.random.seed(2017)
x0 = np.array([np.random.normal() for i in range(N)])
states = L96_RK(x0, T, h)
np.save("/home/cody/initial-position", states[-1])
tf = time.time()
print("Initial position generated. Time: {0}".format(tf-t0))


# create one long run of true states
t0 = time.time()
x0 = np.load("/home/cody/initial-position.npy")

states = L96_RK(x0, 5000, h)
states = np.array(states)
np.save('/home/cody/all-true-states', states)
tf = time.time()
print("All true states generated. Time: {0}".format(tf-t0))


def generate_states(T):  # should just run once for long and clip...
    # get true states to generate synthetic data
    t0 = time.time()
    states = np.load("/home/cody/all-true-states.npy")
    some_states = states[:int(T/h)+1]
    np.save("./true-states/true-states-T{0}".format(T), some_states)
    tf = time.time()
    print("True states generated. Time: {0}".format(tf-t0))


def generate_data(T):
    # generate synthetic data
    # only observe every obs_gap seconds,
    # so keep only every nth state, n = obs_gap/h
    t0 = time.time()
    states = np.load("./true-states/true-states-T{0}.npy".format(T))
    obs = np.array(states[n+1::n])  # not observing at time 0
    obs = list(map(lambda A: np.delete(A, range(0, N, 2)), obs))
    obs = np.array(obs)

    np.random.seed(2016)
    R = np.identity(N//2)
    eta = np.array([np.random.multivariate_normal(np.zeros(N//2), R)
                        for y in obs])
    obs = obs + eta  #check this?
    np.save("./synthetic-observations/synthetic-obs-T{0}"
            .format(T), obs)
    tf = time.time()
    print("Synthetic data generated. Time: {0}".format(tf-t0))


for T in [10, 20, 50, 100, 200, 300, 500, 1000]:
    generate_states(T)
    generate_data(T)


# # generate initial ensembles
t0 = time.time()
T = 2000
states = np.load('/home/cody/all-true-states.npy')
# x0 = np.load("/home/cody/initial-position.npy")
# states = L96_RK(x0, T, h)
# np.save("/home/cody/large-run-for-sampling", states)
for Ne in [20, 40, 50, 100, 200, 500]:
    ens = [states[np.random.randint(len(states))] for i in range(Ne)]
    np.save("./initial-ensembles/initial-ens-{0}".format(Ne), ens)

tf = time.time()
print("Inital ensembles generated. Time: {0}".format(tf-t0))

# t0 = time.time()
# initial_mean = np.mean(states, axis=0)
# initial_cov = (1/(len(states)-1)) * sum([np.outer((e - initial_mean),
#                                                   (e - initial_mean))
#                                          for e in states])
# np.save("initial_mean", initial_mean)
# np.save("initial_cov", initial_cov)
# t1 = time.time()
# print(t1-t0)
