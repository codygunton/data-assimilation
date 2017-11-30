import numpy as np
import time
import matplotlib.pyplot as plt
import multiprocessing as mp

# specify simultation length.
# options: T = 10, 20, 50, 100, 200, 300, 500, 1000
T = 20
# specify ensemble size. options: 20, 40, 50, 100, 200, 500
Ne = 21
# all other parameters are derived

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
H = np.reshape(H, (N//2, N))
h = T/(len(true_states)-1)                           # time step size
n = (len(true_states)-1)//(len(obs))    # number of iters between obsns


def ens_mean_cov(ens):
    Ne = ens.shape[0]
    mu = np.mean(ens, axis=0)
    P = (1/(Ne-1)) * sum([np.outer((e - mu), (e - mu)) for e in ens])
    return mu, P


def heatmap(P):
    plt.imshow(P, cmap='hot', interpolation='nearest')
    plt.show()


# functions defining lorenz system
def f(x):
    N = len(x)
    out = [(x[(i+1) % N] - x[(i-2) % N]) * x[(i-1) % N] - x[i] + F
           for i in range(N)]
    return np.array(out)


def Euler_step(x, h):
    return x + f(x) * h


def RK_step(x, h):
    k1 = f(x)
    k2 = f(x + h*k1/2)
    k3 = f(x + h*k2/2)
    k4 = f(x + h*k3)
    return x + (h/6)*(k1 + 2*k2 + 2*k3 + k4)


def M(x):
    for i in range(n):
        x = RK_step(x, h)
    return x


# enkf
def main(alpha, r, obs, log_name):
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

    for y in obs:
        # get previous ensemble and analysis mean and covariance
        ens = ensembles[-1]

        # forecast
        ens_f = list(map(M, ens))
        mu_f = np.mean(ens_f, axis=0)

        # inflate ensemble
        infl_ens_f = np.array([mu_f + np.sqrt(1+alpha)*(e - mu_f)
                               for e in ens_f])
        mu_f, infl_P_f = ens_mean_cov(infl_ens_f)

        # localize covariance matrix
        P_f_loc = L * infl_P_f

        K = np.linalg.solve(H @ P_f_loc.T @ H.T + R.T, H @ P_f_loc.T).T
        # K = P_f_loc @ H.T @ np.linalg.inv(H @ P_f_loc @ H.T + R)

        # analysis
        ty_s = []
        np.random.seed(2012)
        for e in infl_ens_f:
             eta = np.random.multivariate_normal(np.zeros(N//2), R)
             ty = y + eta
             ty_s += [ty]
        ty_s = np.array(ty_s)

        ens_a = []
        # ty = ty_s[0]
        for i in range(Ne):
            e = infl_ens_f[i]  # should this be inflated? i had it inflated
            ty = ty_s[i]
            new_ens_member = e + K @ (ty - H @ e)
            ens_a.append(new_ens_member)
        ens_a = np.array(ens_a)

        # mu_a, P_a = ens_mean_cov(ens_a)
        mu_a = mu_f + K @ (y - H @ mu_f)
        P_a = (np.identity(N) - K @ H) @ P_f_loc

        ensembles += [ens_a]
        mu_as += [mu_a]
        P_as += [P_a]

    mu_as = np.array(mu_as)
    P_as = np.array(P_as)

    x_ts = true_states[n::n]

    RMSEs = list(map(lambda d: np.sqrt(np.sum(d)/N),
                     (mu_as - x_ts)**2))

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


(ensembles,
 mu_as,
 P_as,
 RMSEs,
 spreads) = main(0.15, 3, obs, 'whatever')

# np.save('eda_ens_0', ensembles[len(ensembles)//2])

# # show RMSEs
# plt.figure(figsize=(12, 6))
# times = range(len(RMSEs))
# plt.scatter(times, RMSEs, s=10, c='#ff6699')
# plt.show()

min_eigval = min(list(map(lambda P: np.linalg.eigvalsh(P)[0], P_as)))
print('min eigenvalue: {0}'.format(min_eigval))

# for i in range(5):
#     P = P_as[np.random.randint(len(P_as))]
#     heatmap(P)
#     print(np.linalg.eigvalsh(P)[0])


# if __name__ == "__main__":
#     alpha_lower = 0
#     alpha_upper = 0.3
#     alpha_step = 0.02
#     r_lower = 2
#     r_upper = 18
#     r_step = 0.2
#     params = [(alpha, r)
#               for alpha in
#               np.arange(alpha_lower, alpha_upper, alpha_step)
#               for r in
#               np.arange(r_lower, r_upper, r_step)]

#     # set up log
#     log_name = 'po-T{0}-Ne{1}'.format(T, Ne)
#     log_file = open(log_name, 'w')
#     log_file.write("T = {4}, Ne = {0}, N = {1}, h = {2}, n = {3}\n"
#                    .format(Ne, N, h, n, T))
#     log_file.write("alpha range: ({0})\n"
#                    .format((alpha_lower, alpha_lower, alpha_step)))
#     log_file.write("r range: ({0})\n"
#                    .format((r_lower, r_upper, r_step)))
#     log_file.close()

#     # get RMSEs
#     all_RMSEs = []
#     for i in range(len(params)//5+1):
#         q = mp.Queue()
#         param_set = params[5*i:5*(i+1)]
#         p = mp.Process(name='iterate_main',
#                        target=iterate_main,
#                        args=(param_set, log_name, q))
#         p.start()
#         RMSE_info = q.get()
#         all_RMSEs += RMSE_info
#         p.join()

#     log_file.close()
#     np.save('po-T{0}-Ne{1}-RMSEs'.format(T, Ne), np.array(all_RMSEs))
