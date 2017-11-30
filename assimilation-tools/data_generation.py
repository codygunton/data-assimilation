import numpy as np
import time
from globals_etc import N, h, n, R
from method import method
if method == 'Euler':
    from L96_functions import L96_Eu as L96
elif method == 'RK':
    from L96_functions import L96_RK as L96
else:
    print('Invalid method specification.')


# get good initial position near attractor
t0 = time.time()
T = 5000                   # propagate for T seconds
np.random.seed(2017)
x0 = np.array([np.random.normal() for i in range(N)])
states = L96(x0, T, h)
np.save("/home/cody/initial-position", states[-1])
tf = time.time()
print("Initial position generated. Time: {0}".format(tf-t0))


# create one long run of true states
t0 = time.time()
x0 = np.load("/home/cody/initial-position.npy")
T = 5000
states = L96(x0, T, h)
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
    obs = np.array(states[n::n])  # not observing at time 0
    obs = list(map(lambda x: np.delete(x, range(0, N, 2)), obs))
    obs = np.array(obs)

    np.random.seed(2016)
    eta = np.array([np.random.multivariate_normal(np.zeros(N//2), R)
                    for y in obs])
    obs = obs + eta
    np.save("./synthetic-observations/synthetic-obs-T{0}"
            .format(T), obs)
    tf = time.time()
    print("Synthetic data generated. Time: {0}".format(tf-t0))


for T in [10, 20, 50, 100, 200]:
    generate_states(T)
    generate_data(T)


# # generate initial ensembles
t0 = time.time()
states = np.load('/home/cody/all-true-states.npy')
for Ne in [20, 21, 40, 41, 50, 100, 200]:
    choice = np.random.choice(len(states), Ne, replace=False)
    ens = states[choice]
    np.save("./initial-ensembles/initial-ens-{0}".format(Ne), ens)
tf = time.time()
print("Inital ensembles generated. Time: {0}".format(tf-t0))


# # get covariance from long run
t0 = time.time()
states = np.load('/home/cody/all-true-states.npy')
states = states[len(states)//5:]
initial_mean = np.mean(states[], axis=0)
initial_cov = (1/(len(states)-1)) * sum([np.outer((e - initial_mean),
                                                  (e - initial_mean))
                                         for e in states])
np.save("initial_mean", initial_mean)
np.save("initial_cov", initial_cov)
t1 = time.time()
print(t1-t0)
