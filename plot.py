#%%
import numpy as np
# import sys

from os import listdir
from os.path import isfile, join

import glob

from numpy.lib.function_base import append

filenames = sorted(glob.glob("./scores/*"))


rew_eval = []
rew_test = []
rate_eval = []
rate_test = []
steps = []
name = []


for fname in filenames:
    if (fname.find('eval') != -1):
        if (fname.find('reward') != -1):
            rew_eval.append(np.loadtxt(fname, dtype=float))
            # just do this name filling in one of the cases
            if (fname.find('large') != -1):
                name.append('IMPALA-large')
            elif (fname.find('baseline') != -1):
                name.append('baseline')
            elif (fname.find('policy') != -1):
                name.append('IMPALA + 3-layered Policy')
            
        elif (fname.find('success_rate')):
            rate_eval.append(np.loadtxt(fname, dtype=float))
    elif (fname.find('test')) != -1:
        if (fname.find('reward') != -1):
            rew_test.append(np.loadtxt(fname, dtype=float))
        elif (fname.find('success_rate')):
            rate_test.append(np.loadtxt(fname, dtype=float))
    else:
        steps.append(np.loadtxt(fname, dtype=float))

    
        

#%% Plot
import matplotlib.pyplot as plt
# if we have several runs we could do the following: https://stackoverflow.com/questions/12957582/plot-yerr-xerr-as-shaded-region-rather-than-error-bars

clr = ['orange','steelblue','mediumseagreen']
plt.figure()
for i in range(len(steps)):
    plt.plot(steps[i], rate_eval[i] , clr[i])

plt.legend(name)
# clr = ['darkorange','steelblue','seagreen']
for i in range(len(steps)):
    plt.plot(steps[i], rate_test[i], color=clr[i], linestyle='--')
plt.grid(True)
plt.xlim([0,2.5e6])
plt.show()


