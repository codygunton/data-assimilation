import numpy as np
import time
from globals_etc import ens_mean_cov, R
from fourdvar import fourdvar_fun as fourdvar


# eda
def eda(alpha, r, T, ens_0, B_0, true_states, obs, log_name):
    t0 = time.time()
    N = len(ens_0[0])                      # number of state variables
    Ne = len(ens_0)                        # ensemble size
    mu_a, _ = ens_mean_cov(ens_0)          # mean of initial ensemble
    h = T/(len(true_states)-1)             # discretization step size
    n = (len(true_states)-1)//(len(obs))   # gap between obs

    mu_as = [mu_a]
    B_s = [B_0]

    # localization function
    D = sum([np.eye(N, k=d) * np.exp(-(d/r)**2)
             for d in range(-N+1, N)])
    E = sum([np.eye(N, k=d) * np.exp(-(abs(abs(N-1)-abs(d))/(r))**2)
             for d in range(-N+1, N)])
    L = (E + D)/(E + D)[0][0]

    np.random.seed(2018)

    for i in range(len(obs)):
        y = obs[i]
        mu = mu_as[i]
        B = B_s[i]

        min_eigval = np.linalg.eigvalsh(B)[0]
        if min_eigval < 1e-10:
            print('covariance collapse!')
            break

        tx_0_opts = []
        for i in range(Ne):
            print(i)
            eta_y = np.random.multivariate_normal(np.zeros(N//2), R)
            ty = y + eta_y

            eta_mu = np.random.multivariate_normal(np.zeros(N), B)
            tmu = mu + eta_mu

            tx_0_opt, _, _, _, _, _, = fourdvar(tmu, B, R, ty, h, n)
            tx_0_opts += [tx_0_opt]

        tx_0_opts = np.array(tx_0_opts)
        x_0_opt, _, _, _, _, _, = fourdvar(mu, B, R, y, h, n)

        mu_a = M(x_0_opt, h, n)

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

    # discard data points from spin-up time
    later_RMSEs = RMSEs[len(obs)//2:]

    t1 = time.time()

    log_file = open(log_name, 'a')
    log_file.write("[{0}, {1}, {2}, {3}]\n"
                   .format(alpha, r, np.mean(later_RMSEs), t1-t0))
    log_file.close()

    print("\n α = {0}, r = {1} "
          " =>   mean RMSE after spin-up = {2} \n runtime: {3}"
          .format(alpha, r, np.mean(RMSEs), t1-t0))

    return mu_as, B_s, RMSEs, spreads


T = 10
ens_0 = np.load('initial-ensembles/initial-ens-20.npy')
B_0 = np.load('initial_cov.npy')
obs = np.load('synthetic-observations/synthetic-obs-T{0}.npy'.format(T))
true_states = np.load('true-states/true-states-T{0}.npy'.format(T))
h = T/(len(true_states)-1)             # discretization step size
n = (len(true_states)-1)//(len(obs))   # gap between obs


(mu_as,
 B_s,
 RMSEs,
 spreads) = eda(0.15, 2, T, ens_0, B_0, true_states,
                obs[len(obs)//2:len(obs)//2+5], 'log')
