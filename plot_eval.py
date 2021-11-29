#%%

import numpy as np
import glob
import matplotlib.pyplot as plt

filenames = sorted(glob.glob("./scores/*.csv"))

rew_eval = []
rew_test = []
rate_eval = []
rate_test = []

steps = []
rew_name = []
rate_name = []

modes = ['background','red','noise']

for mode in modes:
    for i,fname in enumerate(filenames):
        data = np.mean(np.loadtxt(fname, dtype=float))
    
        if (fname.find(mode) != -1):
            if (fname.find('reward') != -1):
                plt.figure(1)
                plt.plot(i,data, 'o')
                rew_name.append(fname)
                plt.legend(rew_name)
            elif (fname.find('rate') != -1):
                pass
                plt.figure(2)
                plt.plot(i,data, 'o')
                rate_name.append(fname)
                plt.legend(rate_name)
    rew_name = []
    rate_name = []

    # plt.legend(name)
    plt.show()
            
        

        





# #%% Plot
# import matplotlib.pyplot as plt
# # if we have several runs we could do the following: https://stackoverflow.com/questions/12957582/plot-yerr-xerr-as-shaded-region-rather-than-error-bars

# # clr = ['orange','steelblue','mediumseagreen','crimson']
# plt.figure()
# for i in range(len(rate_eval)):
#     plt.plot(steps, rate_eval[i])

# plt.legend(name)
# # clr = ['orange','steelblue','mediumseagreen','crimson']
# for i in range(len(rate_test)):
#     plt.plot(steps, rate_test[i], linestyle='--')
# plt.grid(True)
# plt.xlim([0,2.5e6])
# plt.show()


# # clr = ['orange','steelblue','mediumseagreen','crimson']
# plt.figure()
# for i in range(len(rew_eval)):
#     plt.plot(steps, rew_eval[i] )

# plt.legend(name)
# # clr = ['orange','steelblue','mediumseagreen','crimson']
# for i in range(len(rew_test)):
#     plt.plot(steps, rew_test[i], linestyle='--')
# plt.grid(True)
# plt.xlim([0,2.5e6])
# plt.show()


# # %%


# name = ['10', '10000', '1000', '100', '5000','500']

# rew_eval_lvl = []
# rew_test_lvl = []
# rate_eval_lvl = []
# rate_test_lvl = []

# for i in range(len(rate_eval)):
#     rew_eval_lvl.append(np.mean(rew_eval[i][-20:]))
#     rew_test_lvl.append(np.mean(rew_test[i][-20:]))
#     rate_eval_lvl.append(np.mean(rate_eval[i][-20:]))
#     rate_test_lvl.append(np.mean(rate_test[i][-20:]))

# lvl = [10, 10000,1000,100,5000,500]

# plt.figure()
# plt.plot(lvl,rate_eval_lvl, 'x')
# plt.plot(lvl,rew_test_lvl, 'o')
# plt.xscale('log')
# plt.legend(['test','train'])
# plt.show()
# # %%
