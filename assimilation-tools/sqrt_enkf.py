import numpy as np
import time

from globals_etc import R, H, ens_mean_cov
from method import method
if method == 'RK':
    from L96_functions import M_RK as M
elif method == 'Eu':
    from L96_functions import M_Eu as M
else:
    print('Invalid method specification.')


def sqrt_enkf(alpha, r, T, ens_0, true_states, obs, log_name):
    t0 = time.time()
    Ne = len(ens_0)                        # ensemble size
    N = np.shape(true_states)[1]           # number of state variables
    h = T/(len(true_states)-1)             # discretization step size
    n = (len(true_states)-1)//(len(obs))   # gap between obs

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
        ens_f = list(map(lambda x: M(x, h, n), ens))
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
        mu_a = mu_f + K @ (y - H @ mu_f)
        P_a = (np.identity(N) - K @ H) @ P_f_loc

        mu_f_rep = np.array([mu_f for e in infl_ens_f])
        X_f = (1/np.sqrt(Ne-1))*np.array(infl_ens_f - mu_f_rep).T
        V = X_f.T @ H.T
        Z2 = np.identity(Ne) - V @ np.linalg.inv(R + V.T @ V) @ V.T
        evals, U = np.linalg.eigh(Z2)

        D = np.diag(np.sqrt(evals))
        Z = U @ D @ U.T
        X_a = X_f @ Z

        mu_a_rep = np.array([mu_a for e in infl_ens_f])
        ens_a = mu_a_rep + np.sqrt(Ne-1) * X_a.T

        ensembles += [ens_a]
        mu_as += [mu_a]
        P_as += [P_a]

    mu_as = np.array(mu_as)
    P_as = np.array(P_as)

    x_ts = true_states[n::n]

    RMSEs = list(map(lambda d: np.sqrt(np.sum(d)/N),
                     (mu_as - x_ts)**2))

    spreads = [np.sqrt(np.trace(P)/N) for P in P_as]

    # discard data points from spin-up time
    later_RMSEs = RMSEs[len(obs)//2:]
    # when i renamed this to later_RMSEs my mean changed...

    t1 = time.time()

    log_file = open(log_name, 'a')
    log_file.write("[{0}, {1}, {2}, {3}]\n"
                   .format(alpha, r, np.mean(later_RMSEs), t1-t0))
    log_file.close()

    print("\n α = {0}, r = {1} "
          " =>   mean RMSE after spin-up = {2} \n runtime: {3}"
          .format(alpha, r, np.mean(RMSEs), t1-t0))

    return ensembles, mu_as, P_as, RMSEs, spreads
