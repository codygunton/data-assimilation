# to do
# put iterate and multiprocessing functions here

def iterate_sqrt_enkf(param_set, obs, log_name, q):  # q is inst of mp.Queue()
    RMSE_info = []
    for (alpha, r) in param_set:
                _, _, _, RMSEs, _ = main(alpha, r, obs, log_name)
                RMSE_info += [((alpha, r), RMSEs)]
    q.put(RMSE_info)


# # # show RMSEs
# # plt.figure(figsize=(12, 6))
# f# times = range(len(RMSEs))
# # plt.scatter(times, RMSEs, s=10, c='#ff6699')
# # plt.show()

# # min_eigval = min(list(map(lambda P: np.linalg.eigvalsh(P)[0], P_as)))
# # print('min eigenvalue: {0}'.format(min_eigval))

# # for i in range(5):
# #     P = P_as[np.random.randint(len(P_as))]
# #     heatmap(P)
# #     print(np.linalg.eigvalsh(P)[0])


# if __name__ == "__main__":
#     alpha_lower = 0.01
#     alpha_upper = 0.2
#     alpha_step = 0.01
#     r_lower = 1
#     r_upper = 20
#     r_step = 0.5
#     params = [(alpha, r)
#               for alpha in
#               np.arange(alpha_lower, alpha_upper, alpha_step)
#               for r in
#               np.arange(r_lower, r_upper, r_step)]

#     # set up log
#     log_name = 'sqrt-T{0}-Ne{1}'.format(T, Ne)
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
#         p = mp.Process(name='iterate_sqrt_enkf',
#                        target=iterate_sqrt_enkf,
#                        args=(param_set, obs, log_name, q))
#         p.start()
#         print('process started')
#         RMSE_info = q.get()
#         all_RMSEs += RMSE_info
#         p.join()

#     log_file.close()
#     np.save('sqrt-T{0}-Ne{1}-RMSEs'.format(T, Ne), np.array(all_RMSEs))

