import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import multiprocessing as mp

# specify simultation length. options: T = 10, 20, 50, 100, 500, 1000
T = 10
# specify ensemble size. options: 20, 40, 100, 200, 500
Ne = 20


# load states and initial ensemble
true_states = np.load("./true-states/true-states-T{0}.npy".format(T))
obs = np.load("./synthetic-observations/synthetic-obs-T{0}.npy"
              .format(T))
ens_0 = np.load("./initial-ensembles/initial-ens-{0}.npy".format(Ne))


# global parameters
Ne = len(ens_0)                                       # ensemble size
N = np.shape(true_states)[1]              # number of state variables
F = 8                                                   # coefficient
R = np.identity(N//2)            # covariance matrix for observations
H = np.array([np.eye(1, N, i) for i in range(1, N, 2)])  # obs matrix
H = np.reshape(H, (20, 40))
h = T/len(true_states)                                 # RK time step
n = int(len(true_states)/len(obs))    # number of iters between obsns


# functions defining lorenz system
def f(A):
    N = len(A)
    out = [(A[(i+1) % N] - A[(i-2) % N]) * A[(i-1) % N] - A[i] + F
           for i in range(N)]
    return np.array(out)


# the function defining one Runge-Kutta step of size h
def M(A, h):
    k1 = f(A)
    k2 = f(A + h*k1/2)
    k3 = f(A + h*k2/2)
    k4 = f(A + h*k3)
    return A + (h/6)*(k1 + 2*k2 + 2*k3 + k4)


# observations happen every obs_gap seconds, which is n RK steps;
# we need to iterate M that many times
def Mn(A):  # want to use map below so i assume h is defined
    for i in range(n):
        A = M(A, h)
    return A


# kalman filter
def main(alpha, r, log_name):
    t0 = time.time()
    ensembles = [ens_0]
    mu_as = []
    P_as = []

    # localization function
    D = sum([np.eye(N, k=d) * np.exp(-(d/r)**2)
             for d in range(-N+1, N)])
    E = sum([np.eye(N, k=d) * np.exp(-(abs(abs(N-1)-abs(d))/r)**2)
             for d in range(-N+1, N)])
    L = (E + D)/(E + D)[0][0]

    # generate perturbed observations
    np.random.seed(2018)
    epsilon = np.array([np.random.multivariate_normal(np.zeros(N//2), R)
                        for y in obs])
    perturbed_obs = obs + epsilon

    y = obs[0]
    ty = perturbed_obs[0]

    for t in range(len(obs)-1):
        y = obs[t+1]
        ty = perturbed_obs[t+1]

        # get previous ensemble and analysis mean and covariance
        ens = ensembles[-1]

        # forecast
        ens_f = list(map(Mn, ens))
        mu_f = (1/Ne) * sum(ens_f)

        # inflate ensemble
        infl_ens_f = np.array([mu_f + np.sqrt(1+alpha)*(e - mu_f)
                               for e in ens_f])
        infl_P_f = (1/(Ne-1)) * sum([np.outer((e - mu_f), (e - mu_f))
                                    for e in infl_ens_f])
        # infl_ens_f = ens_f
        # infl_P_f = (1/(Ne-1)) * sum([np.outer((e - mu_f), (e - mu_f).T)
                                    # for e in infl_ens_f])

        # localize covariance matrix
        P_f_loc = L * infl_P_f
        # P_f_loc = infl_P_f

        K = np.linalg.solve(H @ P_f_loc.T @ H.T + R.T, H @ P_f_loc.T).T

        # analysis
        ens_a = map(lambda e: e + K @ (ty - H @ e), infl_ens_f)
        # tilde should be here; should be on infl_ens_f?
        ens_a = np.array(list(ens_a))
        mu_a = mu_f + K @ (ty - H @ mu_f)  # should use tilde-y?
        # P_a = (np.identity(N)-K @ H) @ P_f_loc
        P_a = (1/(Ne-1)) * sum([np.outer((e - mu_f), (e - mu_f).T)
                                for e in ens_a])

        ensembles += [ens_a]
        mu_as += [mu_a]
        P_as += [P_a]

    mu_as = np.array(mu_as)
    P_as = np.array(P_as)

    x_ts = true_states[range(n, len(true_states), n)]

    RMSEs = list(map(lambda A: np.sqrt(sum(A)/N),
                     (mu_as-x_ts)**2))

    spreads = [np.sqrt(np.trace(P)/N) for P in P_as]

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

    # discard data points from spin-up time
    RMSEs = RMSEs[len(obs)//2:]
    spreads = spreads[len(obs)//2:]

    t1 = time.time()

    log_file = open(log_name, 'a')
    log_file.write("[{0}, {1}, {2}, {3}]\n"
                   .format(alpha, r, np.mean(RMSEs), t1-t0))
    log_file.close()

    print("\n α = {0}, r = {1} "
          " =>   mean RMSE after spin-up = {2} \n runtime: {3}"
          .format(alpha, r, np.mean(RMSEs), t1-t0))

    return ensembles, mu_as, P_as, RMSEs, spreads


def iterate_main(param_set, log_name, q):  # q is instance of mp.Queue()
    RMSE_info = []
    for (alpha, r) in param_set:
                _, _, _, RMSEs, _ = main(alpha, r, log_name)
                RMSE_info += [((alpha, r), RMSEs)]
    q.put(RMSE_info)


# ensembles, mu_as, P_as, RMSEs, spreads = main(0.2, 5, 'whatever')


if __name__ == "__main__":
    alpha_lower = 0.02
    alpha_upper = 0.5
    alpha_step = 0.02
    r_lower = 1
    r_upper = 6
    r_step = 0.5
    params = [(alpha, r)
              for alpha in
              np.arange(alpha_lower, alpha_upper, alpha_step)
              for r in
              np.arange(r_lower, r_upper, r_step)]
    params = params[:3]
    # set up log
    log_name = 'po-T{0}-Ne{1}'.format(T, Ne)
    log_file = open(log_name, 'w')
    log_file.write("T = {4}, Ne = {0}, N = {1}, h = {2}, n = {3}\n"
                   .format(Ne, N, h, n, T))
    log_file.write("alpha range: ({0})\n"
                   .format((alpha_lower, alpha_lower, alpha_step)))
    log_file.write("r range: ({0})\n"
                   .format((r_lower, r_upper, r_step)))
    log_file.close()

    # get RMSEs
    all_RMSEs = []
    for i in range(len(params)//5+1):
        q = mp.Queue()
        param_set = params[5*i:5*(i+1)]
        p = mp.Process(name='iterate_main',
                       target=iterate_main,
                       args=(param_set, log_name, q))
        p.start()
        RMSE_info = q.get()
        all_RMSEs += RMSE_info
        p.join()

    log_file.close()
    np.save('po-T{0}-Ne{1}-RMSEs'.format(T, Ne), np.array(all_RMSEs))
