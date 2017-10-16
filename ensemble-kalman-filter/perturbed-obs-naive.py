import numpy as np
import cProfile

obs = np.load("synthetic-obs.npy")
ens_0 = np.load("initial-ens.npy")
true_states = np.load("true-states.npy")
Ne = len(ens_0)
N = 40             # number of state variables
F = 8              # coefficient
R = np.identity(N)
H = np.zeros(N)
np.put(H, range(1, N, 2), 1)  # we observe odd-indexed variables
H = np.diag(H)
h = 0.005


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


n = 20
h = 0.005


def Mn(x):  # want to use map below so i assume h is defined
    for i in range(n):
        x = RK_step(x, h)
    return x


# kalman filter

def this():
    ensembles = [ens_0]
    mu_as = []
    P_as = []

    np.random.seed(2017)
    epsilon = np.array([np.random.multivariate_normal(np.zeros(N), R)
                        for y in obs])
    perturbed_obs = obs + epsilon
    tilde_y = perturbed_obs[0]

    for tilde_y in perturbed_obs:
        # get previous ensemble and analysis mean and covariance
        ens = ensembles[-1]

        # forecasting step
        ens_f = list(map(Mn, ens))
        mu_f = (1/Ne) * sum(ens_f)
        P_f = (1/(Ne-1)) * sum([np.outer((e - mu_f),
                                         (e - mu_f).T) for e in ens_f])
        K = np.linalg.solve(R.T + (H @ P_f @ H.T).T, H @ P_f.T).T

        # form analysis ensemble
        ens_a = map(lambda e: e + K @ (tilde_y - H @ e), ens_f)
        ens_a = np.array(list(ens_a))
        mu_a = (1/Ne) * sum(ens_a)
        P_a = (1/(Ne-1)) * sum([np.outer((e - mu_a),
                                         (e - mu_a).T) for e in ens_a])

        ensembles += [ens_a]
        mu_as += [mu_a]
        P_as += [P_a]

    ensembles = ensembles[1:]
    return ensembles, mu_as, P_as


cProfile.run('ensembles, mu_as, P_as = this()')
xts = true_states[range(0, len(true_states), 20)]



# generate and plots RMSEs and traces of analysis covariances
RMSEs = [np.sqrt(0.5*((true_xs[k][0] - kalman_mus[k][0])**2
                      + (true_xs[k][1] - kalman_mus[k][1])**2))
         for k in range(T+1)]

plt.figure(figsize=(12, 6))
plt.scatter(times, RMSEs, s=1, c='#ff6699')
cov_traces = [np.sqrt(0.5 * np.trace(P)) for P in kalman_Ps]
plt.scatter(times, cov_traces, s=1, c='#42f4ce')
plt.suptitle('Root MSE of reconstruction and normalized trace of analysis covariances',
             fontsize=14, fontweight='bold')
plt.xlabel(r"$t$"' (thousands of time steps of size dt)',
                  fontsize=14, fontweight='bold')
plt.legend(['RMSE', 'Normalized trace of analysis covariance'], markerscale=4)
plt.show()


# maybe i need to "iterate M"?
# why are the P_f so big
# why f(ens[0]) == f(ens)[0] is FALSE?
