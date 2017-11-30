import numpy as np
from scipy.optimize import least_squares as lsqs
import matplotlib.pyplot as plt
from method import method
from globals_etc import N, H, R
if method == 'RK':
    from L96_functions import M_RK as M, DM_RK as DM
elif method == 'Eu':
    from L96_functions import M_Eu as M, DM_Eu as DM
else:
    print('Invalid method specification.')


def fun_r(x, mu_0, y, sqrtB_inv, sqrtR_inv, h, n):  # f = (1/2)r^Tr
    r1 = sqrtB_inv @ (x - mu_0)
    r2 = sqrtR_inv @ (H @ M(x, h, n) - y)
    r = np.concatenate((r1, r2))
    return r


def fun_J(x, sqrtB_inv, sqrtR_inv, h, n):  # the jacobian of fun_r
    J1 = sqrtB_inv
    J2 = sqrtR_inv @ H @ DM(x, h, n)
    J = np.concatenate((J1, J2))
    return J


def fourdvar_fun(mu_0, B_0, R, y, h, n):
    # notation: moving from time 0 to time next
    # y should be obs[n] if at time 0

    # compute matrix square roots
    evals, U = np.linalg.eigh(B_0)
    sqrtD_inv = np.diag(1/np.sqrt(evals))
    sqrtB_0_inv = U @ sqrtD_inv @ U.T
    sqrtR_inv = R  # hack; in general, compute as for B_0

    def r(x):
        return fun_r(x, mu_0, y, sqrtB_0_inv, sqrtR_inv, h, n)

    def J(x):
        return fun_J(x, sqrtB_0_inv, sqrtR_inv, h, n)

    # run gauss-newton
    x_0_opt = lsqs(r, mu_0, jac=J).x

    mu_next = M(mu_0, h, n)  # bad name
    mu_next_opt = M(x_0_opt, h, n)

    M_lin = DM(x_0_opt, h, n)
    J = fun_J(x_0_opt, sqrtB_0_inv, sqrtR_inv, h, n)
    B_0_post = np.linalg.inv(2*J.T @ J)

    B_next_prior = M_lin @ B_0 @ M_lin.T
    B_next_post = M_lin @ np.linalg.inv(2*J.T @ J) @ M_lin.T

    return (x_0_opt, mu_next, mu_next_opt,
            B_0_post, B_next_prior, B_next_post)

###############


# test fourdvar functions

# mu_0 = np.load("initial_mean.npy")
# B_0 = np.load("initial_cov.npy")
# T = 10
# obs = np.load('synthetic-observations/synthetic-obs-T{0}.npy'.format(T))
# y = obs[0]
# true_states = np.load('true-states/true-states-T{0}.npy'.format(T))
# h = T/(len(true_states)-1)             # discretization step size
# n = (len(true_states)-1)//(len(obs))   # gap between obs


def show_metrics():
    # at time 0, plot samples from prior and posterior
    RMSE = np.sqrt(sum((x_0_opt - true_states[0])**2)/N)
    print('RMSE = {0}'.format(RMSE))
    sample_size = 100
    prior_samples = [np.random.multivariate_normal(mu_0, B_0)
                     for i in range(sample_size)]
    prior_maxes = np.amax(prior_samples, axis=0)
    prior_mins = np.amin(prior_samples, axis=0)

    posterior_samples = [np.random.multivariate_normal(x_0_opt,
                                                       B_0_post)
                         for i in range(sample_size)]
    post_maxes = np.amax(posterior_samples, axis=0)
    post_mins = np.amin(posterior_samples, axis=0)

    plt.figure(figsize=(12, 6))
    plt.suptitle('Samples from prior and posterior at time 0',
                 fontsize=14, fontweight='bold')
    plt.xlabel(r"$i$"' (coordinate index)', fontsize=14,
               fontweight='bold')
    coords = list(range(1, N+1))
    plt.vlines(coords, prior_mins, prior_maxes, colors='C0', lw=10)
    plt.vlines(coords, post_mins, post_maxes, colors='C6', lw=5)
    x_0 = true_states[0]
    plt.plot(coords, x_0, c='k', marker='o', linestyle='None')
    plt.show()


    # at time n, plot samples from prior and posterior
    RMSE = np.sqrt(sum((mu_next_opt - true_states[n])**2)/N)
    print('RMSE = {0}'.format(RMSE))

    sample_size = 100
    prior_samples = [np.random.multivariate_normal(mu_next,
                                                   B_next_prior)
                     for i in range(sample_size)]
    prior_maxes = np.amax(prior_samples, axis=0)
    prior_mins = np.amin(prior_samples, axis=0)

    posterior_samples = [np.random.multivariate_normal(mu_next_opt,
                                                       B_next_post)
                         for i in range(sample_size)]
    post_maxes = np.amax(posterior_samples, axis=0)
    post_mins = np.amin(posterior_samples, axis=0)

    plt.figure(figsize=(12, 6))
    plt.suptitle('Samples from prior and posterior at 0.2s in future',
                 fontsize=14, fontweight='bold')
    plt.xlabel(r"$i$"' (coordinate index)',
               fontsize=14, fontweight='bold')
    coords = list(range(1, N+1))
    plt.vlines(coords, prior_mins, prior_maxes, colors='C0', lw=10)
    plt.vlines(coords, post_mins, post_maxes, colors='C6', lw=5)
    x_n = true_states[n]
    plt.plot(coords, x_n, c='k', marker='o', linestyle='None')
    plt.show()


# (x_0_opt,
#  mu_next,
#  mu_next_opt,
#  B_0_post,
#  B_next_prior,
#  B_next_post) = fourdvar_fun(mu_0, B_0, R, y, h, n)

# show_metrics()
