import numpy as np
import time
import matplotlib.pyplot as plt
import multiprocessing as mp
# from fourdvar import fourdvar_fun
import fourdvar

# specify simultation length.
# options: T = 10, 20, 50, 100, 200, 300, 500, 1000
T = 10
# specify ensemble size. options: 20, 40, 100, 200, 500
Ne = 50


# load states and initial ensemble
true_states = np.load("./true-states/true-states-T{0}.npy".format(T))
true_states = true_states[len(true_states)//2:]
obs = np.load("./synthetic-observations/synthetic-obs-T{0}.npy"
              .format(T))
obs = obs[len(obs)//2:]
# the way i define obs, obs[0] is observation at time 0,
# but the first observation we use is obs[n]
# below i assume we use the first element of obs, so i drop obs[0] here
obs = obs[1:]
ens_0 = np.load("eda_ens_0.npy")


# global parameters
Ne = len(ens_0)                                       # ensemble size
N = np.shape(true_states)[1]              # number of state variables
F = 8                                                   # coefficient
R = np.identity(N//2)            # covariance matrix for observations
H = np.array([np.eye(1, N, i) for i in range(1, N, 2)])  # obs matrix
H = np.reshape(H, (20, 40))
h = T/len(true_states)                               # time step size
n = len(true_states)//(len(obs)+1)    # number of iters between obsns


# functions defining lorenz system
def f(A):
    N = len(A)
    out = [(A[(i+1) % N] - A[(i-2) % N]) * A[(i-1) % N] - A[i] + F
           for i in range(N)]
    return np.array(out)


def Euler_step(x, h):
    return x + f(x) * h


# nonlinear model for propagating obs_gap seconds (or n steps)
# into the future
def M(x):
    for i in range(n):
        x = Euler_step(x, h)
    return x

def ens_mean_cov(ens):
    Ne = ens.shape[0]
    mu = np.mean(ens, axis=0)
    P = (1/(Ne-1)) * sum([np.outer((e - mu), (e - mu)) for e in ens])
    return mu, P


# nonlinear model for propagating obs_gap seconds (or n steps)
# into the future
def M(x):
    for i in range(n):
        x = Euler_step(x, h)
    return x


B_0 = np.load('initial_cov.npy')
mu, _ = ens_mean_cov(ens_0)

starting_ens = []
evals, U = np.linalg.eigh(B_0)
sqrtD = np.diag(np.sqrt(evals))
sqrtB = U @ sqrtD @ U.T
for i in range(Ne):
    xi = np.random.multivariate_normal(np.zeros(N), np.identity(N))
    e = mu + sqrtB @ xi
    starting_ens.append(e)
starting_ens = np.array(starting_ens)

# eda
def main(alpha, r, obs, ens_0, B_0, log_name):
    t0 = time.time()
    mu_a, _ = ens_mean_cov(ens_0)
    mu_as = [mu_a]
    B_s = [B_0]

    # localization function
    D = sum([np.eye(N, k=d) * np.exp(-(d/r)**2)
             for d in range(-N+1, N)])
    E = sum([np.eye(N, k=d) * np.exp(-(abs(abs(N-1)-abs(d))/(r))**2)
             for d in range(-N+1, N)])
    L = (E + D)/(E + D)[0][0]

    # plt.imshow(new_R, cmap='hot', interpolation='nearest')
    # plt.show()

    np.random.seed(2018)

    for i in range(len(obs)):
        y = obs[i]
        mu = mu_as[i]
        B = B_s[i]

        tx_0_opts = []
        for i in range(Ne):
            print(i)
            eta_y = np.random.multivariate_normal(np.zeros(N//2), R)
            ty = y + eta_y

            eta_mu = np.random.multivariate_normal(np.zeros(N), B)
            tmu = mu + eta_mu

            tx_0_opt,_, _, _, _, _, = fourdvar_fun(tmu, B, R, ty)
            tx_0_opts += [tx_0_opt]

        tx_0_opts = np.array(tx_0_opts)
        x_0_opt, _, _, _, _, _, = fourdvar_fun(mu, B, R, y)

        mu_a = M(x_0_opt)

        _, B = ens_mean_cov(tx_0_opts)

        B_infl_loc = (1+alpha) * L * B

        mu_as += [mu_a]
        B_s += [B_infl_loc]

    mu_as = np.array(mu_as[1:])
    B_s = np.array(B_s)

    x_ts = true_states[::n][:len(obs)]

    RMSEs = list(map(lambda A: np.sqrt(sum(A)/N),
                     (mu_as-x_ts)**2))

    spreads = [np.sqrt(np.trace(P)/N) for P in B_s]

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

    return mu_as, B_s, RMSEs, spreads


(mu_as,
 B_s,
 RMSEs,
 spreads) = main(0.15, 2,
                 obs[len(obs)//2:len(obs)//2+5],
                 starting_ens, B_0, 'whatever')

# # show RMSEs
# plt.figure(figsize=(12, 6))
# times = range(len(RMSEs))
# plt.scatter(times, RMSEs, s=10, c='#ff6699')
# plt.show()

# # heatmap
# P = np.load('initial_cov.npy')
# _, P = ens_mean_cov(ens_0)
# def heatmap(P):
#     plt.imshow(P, cmap='hot', interpolation='nearest')
#     plt.show()



# if __name__ == "__main__":
#     alpha_lower = 0
#     alpha_upper = 0.5
#     alpha_step = 0.02
#     r_lower = 1
#     r_upper = 6
#     r_step = 0.5
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
