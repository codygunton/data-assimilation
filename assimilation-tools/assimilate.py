# here i can specify T and Ne and test an enkf
import numpy as np
import matplotlib.pyplot as plt
from sqrt_enkf import sqrt_enkf
from globals_etc import ens_mean_cov

# specify simultation length.
# options: T = 10, 20, 50, 100, 200, 300, 500, 1000
T = 100
# specify ensemble size. options: 20, 40, 50, 100, 200, 500
Ne = 40
# all other parameters are derived


# load states and initial ensemble
true_states = np.load("./true-states/true-states-T{0}.npy".format(T))
obs = np.load("./synthetic-observations/synthetic-obs-T{0}.npy"
              .format(T))
ens_0 = np.load("./initial-ensembles/initial-ens-{0}.npy".format(Ne))


def show_plots(RMSEs, spreads):
    # show RMSEs
    plt.figure(figsize=(12, 6))
    times = range(len(RMSEs))
    plt.scatter(times, RMSEs, s=10, c='#ff6699')
    plt.show()

    # show spreads
    plt.figure(figsize=(12, 6))
    times = range(len(spreads))
    plt.scatter(times, spreads, s=10, c='#ff6699')
    plt.show()


## sqrt enkf
## some good values: RK method, T=100, Ne=40, alpha=0.16, r=8
## changing Ne to 20 gives cov. collapse; r=4 better
(ensembles, mu_as,
 P_as, RMSEs, spreads) = sqrt_enkf(0.16, 8, T, ens_0,
                                   true_states, obs, 'log')
show_plots(RMSEs, spreads)


# ## eda
# B_0 = np.load('initial_cov.npy')
# mu, _ = ens_mean_cov(ens_0)

# starting_ens = []
# evals, U = np.linalg.eigh(B_0)
# sqrtD = np.diag(np.sqrt(evals))
# sqrtB = U @ sqrtD @ U.T
# for i in range(Ne):
#     xi = np.random.multivariate_normal(np.zeros(N), np.identity(N))
#     e = mu + sqrtB @ xi
#     starting_ens.append(e)
# starting_ens = np.array(starting_ens)
