import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import cProfile

obs = np.load("synthetic-obs.npy")
ens_0 = np.load("initial-ens-200.npy")
true_states = np.load("true-states.npy")
Ne = len(ens_0)
N = 40             # number of state variables
F = 8              # coefficient
R = np.identity(N//2)
H = np.array([np.eye(1, 40, i) for i in range(1, 40, 2)])
H = np.reshape(H, (20, 40))
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
    y = obs[0]
    I = np.identity(Ne)
    y = obs[2]

    for y in obs:
        # get previous ensemble and analysis mean and covariance
        ens = ensembles[-1]

        # forecasting step
        ens_f = list(map(Mn, ens))
        mu_f = np.mean(ens_f, axis=0)
        X_f = (1/np.sqrt(Ne-1)) * np.array([e - mu_f for e in ens_f]).T
        # expect X_f to have shape (Ne, N)
        P_f = X_f @ X_f.T  # shape (N, N)
        K = np.linalg.solve(R.T + (H @ P_f @ H.T).T, H @ P_f.T).T
            ### it breaks on second round at K
        # form analysis ensemble
        mu_a = mu_f + K @ (y - H @ mu_f)
        V = X_f.T @ H.T
        Z = scipy.linalg.sqrtm(I - V @ np.linalg.inv(V.T @ V + R) @ V.T)
        tilde_ens_a = X_f @ Z
        ens_a = np.repeat([mu_a], Ne, axis=0).T + tilde_ens_a
        # might need to adjust this
        # LEFT OFF having just defined ens_a
        # do i need to recompute the mu_a?
        # this is not working
        ensembles += [ens_a.T]
        mu_as += [mu_a]
        P_as += []

    ensembles = ensembles[1:]
    mu_as = np.array(mu_as)
    P_as = np.array(P_as)
    return ensembles, mu_as, P_as

cProfile.run('ensembles, mu_as, P_as = this()')
# runs in 50 seconds with ensemble size 40
# runs in 347 seconds with ensemble size 200
mu_as = np.array(mu_as)
xts = true_states[range(0, len(true_states), 20)]
RMSEs = list(map(lambda A: np.sqrt(sum(A)//N), (mu_as-xts)**2))
plt.figure(figsize=(12, 6))
times = range(0, 100)
plt.scatter(times, RMSEs, s=10, c='#ff6699')
plt.show()




# maybe i need to "iterate M"?
# why are the P_f so big
# why f(ens[0]) == f(ens)[0] is FALSE?
