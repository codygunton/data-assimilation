import numpy as np
from scipy.optimize import least_squares as lsqs
import time
import matplotlib.pyplot as plt

# specify simultation length. options: T = 10, 20, 50, 100, 500, 1000
T = 50
# specify ensemble size. options: 20, 40, 100, 200, 500
Ne = 20
# all other parameters are derived


# load initial data
true_states = np.load("./true-states/true-states-T{0}.npy".format(T))
obs = np.load("./synthetic-observations/synthetic-obs-T{0}.npy"
              .format(T))

# global parameters
# Ne = len(ens_0)                                     # ensemble size
N = np.shape(true_states)[1]              # number of state variables
F = 8                                                   # coefficient
R = np.identity(N//2)            # covariance matrix for observations
H = np.array([np.eye(1, N, i) for i in range(1, N, 2)])  # obs matrix
H = np.reshape(H, (20, 40))
h = T/len(true_states)                              # Euler time step
n = int(len(true_states)/len(obs))    # number of iters between obsns
obs_gap = n*h                                        # units: seconds



# functions defining lorenz system
def f(A):
    N = len(A)
    out = [(A[(i+1) % N] - A[(i-2) % N]) * A[(i-1) % N] - A[i] + F
           for i in range(N)]
    return np.array(out)


# the function defining one Euler step of size h
def Euler_step(x, h):
    return x + f(x) * h


# nonlinear model for propagating obs_gap seconds (or n steps)
# into the future
def M(x):
    for i in range(n):
        x = Euler_step(x, h)
    return x

# the Jacobian of f (checked once)
def Df(x):
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

def DM(x):  # total derivative of DM
    # dx_(i+1)/dx0 = (I + Df(xi)h)@(dxi/dx0)
    # start with i = 0, compute dx1/dx0 by hand, then iterate
    # the first value of i in the iteration is 1
    dxidx0 = np.identity(N) + Df(x) * h
    xi = M(x)
    dxidx0 = np.identity(N)
    for i in range(1, n):
        dxidx0 = (np.identity(N) + Df(xi)*h) @ dxidx0
        xi = M(x)
    return dxidx0
# # check:
# ep = np.array([np.random.normal(0,.001) for i in range(N)])
# sum(M(x_0+ep) - M(x_0) - DM(x_0) @ ep)


def fun_r(x, mu_0, y, sqrtB_inv, sqrtR_inv):  # r such that f = (1/2)r^Tr
    r1 = sqrtB_inv @ (x - mu_0)
    r2 = sqrtR_inv @ (H @ M(x) - y)
    r = np.concatenate((r1, r2))
    return r


def fun_J(x, sqrtB_inv, sqrtR_inv):  # the jacobian of fun_r
    J1 = sqrtB_inv
    J2 = sqrtR_inv @ H @ DM(x)
    J = np.concatenate((J1, J2))
    return J


def four_d_var(mu_0, B_0, R, y):
    # notation: moving from time 0 to time next
    # y should be obs[n] if at time 0

    # compute matrix square roots
    evals, U = np.linalg.eigh(B_0)
    sqrtD_inv = np.diag(1/np.sqrt(evals))
    sqrtB_0_inv = U @ sqrtD_inv @ U.T
    sqrtR_inv = R  # hack; in general, compute as for B_0

    # run gauss-newton
    x_0_opt = lsqs(lambda x: fun_r(x, mu_0, y, sqrtB_0_inv, sqrtR_inv),
                   mu_0, jac= lambda x: fun_J(x, sqrtB_0_inv, sqrtR_inv)).x

    mu_next = M(mu_0)  # bad name
    mu_next_opt = M(x_0_opt)

    M_lin = DM(x_0_opt)
    J = fun_J(x_0_opt, sqrtB_0_inv, sqrtR_inv)
    B_0_post = np.linalg.inv(2*J.T @ J)

    B_next_prior = M_lin @ B_0 @ M_lin.T
    B_next_post = M_lin @ np.linalg.inv(2*J.T @ J) @ M_lin.T

    return (x_0_opt, mu_next, mu_next_opt,
            B_0_post, B_next_prior, B_next_post)


mu_0 = np.load("initial_mean.npy")
B_0 = np.load("initial_cov.npy")
y = obs[1]

(x_0_opt,
 mu_next,
 mu_next_opt,
 B_0_post,
 B_next_prior,
 B_next_post) = four_d_var(mu_0, B_0, R, y)


# at time 0, plot samples from prior and posterior
RMSE = np.sqrt(sum((x_0_opt - true_states[0])**2)/N)
print('RMSE = {0}'.format(RMSE))
sample_size = 100
prior_samples = [np.random.multivariate_normal(mu_0, B_0)
                 for i in range(sample_size)]
prior_maxes = np.amax(prior_samples, axis=0)
prior_mins = np.amin(prior_samples, axis=0)

posterior_samples = [np.random.multivariate_normal(x_0_opt, B_0_post)
                     for i in range(sample_size)]
post_maxes = np.amax(posterior_samples, axis=0)
post_mins = np.amin(posterior_samples, axis=0)

plt.figure(figsize=(12, 6))
plt.suptitle('Samples from prior and posterior at time 0',
             fontsize=14, fontweight='bold')
plt.xlabel(r"$i$"' (coordinate index)',
                  fontsize=14, fontweight='bold')
coords = list(range(1, N+1))
plt.vlines(coords, prior_mins, prior_maxes, colors = 'C0', lw=10)
plt.vlines(coords, post_mins, post_maxes, colors = 'C6', lw=5)
x_0 = true_states[0]
plt.plot(coords, x_0, c='k', marker='o', linestyle='None')
plt.show()


# at time n, plot samples from prior and posterior
RMSE = np.sqrt(sum((mu_next_opt - true_states[n])**2)/N)
print('RMSE = {0}'.format(RMSE))

sample_size = 100
prior_samples = [np.random.multivariate_normal(mu_next, B_next_prior)
                 for i in range(sample_size)]
prior_maxes = np.amax(prior_samples, axis=0)
prior_mins = np.amin(prior_samples, axis=0)

posterior_samples = [np.random.multivariate_normal(mu_next_opt, B_next_post)
                     for i in range(sample_size)]
post_maxes = np.amax(posterior_samples, axis=0)
post_mins = np.amin(posterior_samples, axis=0)

plt.figure(figsize=(12, 6))
plt.suptitle('Samples from prior and posterior at 0.2s in future',
             fontsize=14, fontweight='bold')
plt.xlabel(r"$i$"' (coordinate index)',
                  fontsize=14, fontweight='bold')
coords = list(range(1, N+1))
plt.vlines(coords, prior_mins, prior_maxes, colors = 'C0', lw=10)
plt.vlines(coords, post_mins, post_maxes, colors = 'C6', lw=5)
x_n = true_states[n]
plt.plot(coords, x_n, c='k', marker='o', linestyle='None')
plt.show()

