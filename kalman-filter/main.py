import numpy as np
import matplotlib.pyplot as plt

# we work with the dampened oscillator d^2z/dt^2 + 2w(xi)dz/dt + w^2z = 0
# we converted to a system in x = (z, dz/dt)

# global parameters
# good values to see dampening: w = 2., xi = 0.1
T = 25000                   # number of time steps
dt = 0.001                  # time step
w = 1.                      # omega
xi = 0.001                  # xi
H = np.array([[1., 0]])     # measurement matrix
R = 5                       # variance of measurement error
I = np.identity(2)          # identity matrix
x0 = np.random.multivariate_normal([0, 0], I)  # initial condition

A = np.array([[0, 1], [-w**2, -2 * xi * w]])  # matrix of system
M = I + dt * A                   # forecasting matrix


# generate numerical solution and plot
times = [0 + k * dt for k in range(T+1)]
true_xs = [x0]
for k in range(T):
    true_xs += [M @ true_xs[-1]]
true_zs = [x[0] for x in true_xs]
true_zts = [x[1] for x in true_xs]
plt.scatter(times, true_zs, s=1, c=(0, 0, 1))
plt.scatter(times, true_zts, s=1, c=(1, 0, 0))
plt.show()


# generate synthetic data and plot
synth_ys = [H @ x + np.random.normal(0, np.sqrt(R)) for x in true_xs]
plt.scatter(times, synth_ys, s=1, c=(0, 1, 0))
plt.scatter(times, true_zs, s=1, c=(0, 0, 1))
plt.legend(['Synthetic z', 'True z'])
plt.show()


# kalman filter
kalman_mus = [[0,0]]
kalman_Ps = [I]
kalman_gains = []
for k in range(T):
    mu_last = kalman_mus[-1]
    P_last = kalman_Ps[-1]

    mu_f = M @ mu_last
    P_f = A @ P_last @ A.T
    K = P_f @ H.T @ np.linalg.inv(R + H @ P_f @ H.T)
    kalman_gains += [K]

    new_mu = mu_f - K @ (H @ mu_f - synth_ys[k])
    new_P = P_f - K @ H @ P_f

    kalman_mus += [new_mu]
    kalman_Ps += [new_P]
kalman_zs = [mu[0] for mu in kalman_mus]
kalman_zts = [mu[1] for mu in kalman_mus]


# plot kalman reconstructions
plt.scatter(times, true_zs, s=1, c=(0, 0, 1))
plt.scatter(times, kalman_zs, s=1, c=(0, 1, 0))
plt.legend(['True z', 'Kalman reconstruction'])
plt.show()
plt.scatter(times, true_zts, s=1, c=(1, 0, 0))
plt.scatter(times, kalman_zts, s=1, c=(0, 1, 0))
plt.legend(['True dz/dt', 'Kalman reconstruction'])
plt.show()


# plot kalman gain components
K0s = [K[0] for K in kalman_gains]
K1s = [K[1] for K in kalman_gains]
plt.scatter(times[1:], K0s, s=1, c='#ff6699')
plt.legend(['First component of Kalman gain'])
plt.show()
plt.scatter(times[1:], K1s, s=1, c='#ff6699')
plt.legend(['Second component of Kalman gain'])
plt.show()


# generate and plots MSEs and traces of analysis covariances
MSEs = [0.5*((true_xs[k][0] - kalman_mus[k][0])**2
             + (true_xs[k][1] - kalman_mus[k][1])**2)
        for k in range(T+1)]
plt.scatter(times, MSEs, s=1, c='#ff6699')
plt.legend(['MSE_k'])
plt.show()

cov_traces = [0.5 * np.trace(P) for P in kalman_Ps]
plt.scatter(times, cov_traces, s=1, c='#ff6699')
plt.legend(['Normalized trace of analysis covariance'])
plt.show()
