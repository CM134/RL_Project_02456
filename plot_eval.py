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
            
        

        



# %%
