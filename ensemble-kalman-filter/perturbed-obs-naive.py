import numpy as np
import cProfile
import matplotlib.pyplot as plt


obs = np.load("synthetic-obs.npy")
ens_0 = np.load("initial-ens-40.npy")
true_states = np.load("true-states.npy")
Ne = len(ens_0)
N = 40             # number of state variables
F = 8              # coefficient
R = np.identity(N//2)
H = np.array([np.eye(1, N, i) for i in range(1, N, 2)])
H = np.reshape(H, (20, 40))
h = 0.005


# functions defining lorenz system
def f(A):
    N = len(A)
    out = [(A[(i+1) % N] - A[(i-2) % N]) * A[(i-1) % N] - A[i] + F
           for i in range(N)]
    return np.array(out)


# the function defining one Runge-Kutta step of size h
def M(x, h):
    k1 = f(x)
    k2 = f(x + h*k1/2)
    k3 = f(x + h*k2/2)
    k4 = f(x + h*k3)
    return x + (h/6)*(k1 + 2*k2 + 2*k3 + k4)


# observations happen every 0.01 seconds, which is 0.1//h RK steps;
# we need to iterate M that many times
def Mn(x):  # want to use map below so i assume h is defined
    for i in range(int(0.1/h)):
        x = M(x, h)
    return x


# kalman filter

def main():
    ensembles = [ens_0]  # will remove this element at the end
    mu_as = []
    P_as = []

    # generate perturbed observations
    np.random.seed(2017)
    epsilon = np.array([np.random.multivariate_normal(np.zeros(N//2), R)
                        for y in obs])
    perturbed_obs = obs + epsilon

    # forecast observations and means
    # t for "tilde"
    ty_fs = list(map(Mn, perturbed_obs))
    ty_f_bar = np.mean(ty_fs, axis=0)

    for ty in perturbed_obs:
        # get previous ensemble and analysis mean and covariance
        ens = ensembles[-1]

        # forecasting step
        ens_f = list(map(Mn, ens))
        mu_f = (1/Ne) * sum(ens_f)
        P_f = (1/(Ne-1)) * sum([np.outer((e - mu_f), (e - mu_f).T)
                                for e in ens_f])

        # I think Matti just used R
        Re = (1/(Ne-1)) * sum(  # do I want to do this?
            [np.outer((t - ty_f_bar),
                      (t - ty_f_bar).T) for t in ty_fs])

        K = np.linalg.solve(H @ P_f.T @ H.T + R.T, H @ P_f.T).T

        # form analysis ensemble
        ens_a = map(lambda e: e + K @ (ty - H @ e), ens_f)
        ens_a = np.array(list(ens_a))
        mu_a = mu_f + K @ (ty - H @ mu_f)  # this works too?
        # mu_a = (1/Ne) * sum(ens_a)  # this is what i wrote down
        P_a = (np.identity(N)-K @ H) @ P_f
        ensembles += [ens_a]
        mu_as += [mu_a]
        P_as += [P_a]  # remove this?

    ensembles = ensembles[1:]
    mu_as = np.array(mu_as)
    P_as = np.array(P_as)
    return ensembles, mu_as, P_as


cProfile.run('ensembles, mu_as, P_as = main()')
# 184s when Ne = 200; 447s when N_e = 500

# analyze RMSEs
mu_as = np.array(mu_as)
xts = true_states[range(0, len(true_states), 20)]
RMSEs = list(map(lambda A: np.sqrt(sum(A)/N), (mu_as-xts)**2))
plt.figure(figsize=(12, 6))
times = range(0, 100)
plt.scatter(times, RMSEs, s=10, c='#ff6699')
plt.show()

print("Time-averaged RMSE is {0}".format(np.mean(RMSEs)))

# why f(ens[0]) == f(ens)[0] is FALSE?
