import numpy as np
import matplotlib.pyplot as plt

# global parameters
N = 40                         # number of state variables
F = 8                          # lorenz forcing term
h = 0.01                       # Euler/RK step size

obs_gap = 0.2                  # time between observations
n = int(obs_gap // h)          # number of steps between observations
R = np.identity(N//2)          # covariance matrix for observations
H = np.array([np.eye(1, N, i)
              for i in range(1, N, 2)])  # obs matrix
H = np.reshape(H, (N//2, N))


def ens_mean_cov(ens):
    Ne = ens.shape[0]
    mu = np.mean(ens, axis=0)
    P = (1/(Ne-1)) * sum([np.outer((e - mu), (e - mu)) for e in ens])
    return mu, P


def heatmap(P):
    plt.imshow(P, cmap='hot', interpolation='nearest')
    plt.show()
