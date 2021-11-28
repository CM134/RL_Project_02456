#%%
import numpy as np
# import sys

from os import listdir
from os.path import isfile, join

import glob

from numpy.lib.function_base import append

filenames = sorted(glob.glob("*.csv"))


rew_eval = []
rew_test = []
rate_eval = []
rate_test = []

steps = []
name = []


for fname in filenames:
    data = np.loadtxt(fname, dtype=float)
    # one data set is trained on 5e6 steps. Meaning fails if different length
    # if len(data) > 306:
    #     data = data[:306]
    
    if (fname.find('eval') != -1):
        if (fname.find('reward') != -1):
            rew_eval.append(data)
            name.append(fname)
        elif (fname.find('success_rate')):
            rate_eval.append(data)
    elif (fname.find('test')) != -1:
        if (fname.find('reward') != -1):
            rew_test.append(data)
        elif (fname.find('success_rate')):
            rate_test.append(data)
    elif (fname.find('step') != -1):
        steps = data # always the same just overwrite
        
# std_rew_eval =  np.std(np.array(rew_eval ), axis = 0)
# std_rew_test =  np.std(np.array(rew_test ), axis = 0)
# std_rate_eval = np.std(np.array(rate_eval), axis = 0)
# std_rate_test = np.std(np.array(rate_test), axis = 0)

# rew_eval =  np.mean(np.array(rew_eval ), axis = 0)
# rew_test =  np.mean(np.array(rew_test ), axis = 0)
# rate_eval = np.mean(np.array(rate_eval), axis = 0)
# rate_test = np.mean(np.array(rate_test), axis = 0)
print(name)
name = ['10', '100', '500', '1000', '5000']

#%% Plot
import matplotlib.pyplot as plt
# if we have several runs we could do the following: https://stackoverflow.com/questions/12957582/plot-yerr-xerr-as-shaded-region-rather-than-error-bars

clr = ['orange','steelblue','mediumseagreen','crimson']
plt.figure()
for i in range(len(rate_eval)):
    plt.plot(steps, rate_eval[i] , clr[i])

plt.legend(name)
# clr = ['orange','steelblue','mediumseagreen','crimson']
for i in range(len(rate_test)):
    plt.plot(steps, rate_test[i], color=clr[i], linestyle='--')
plt.grid(True)
plt.xlim([0,2.5e6])
plt.show()


clr = ['orange','steelblue','mediumseagreen','crimson']
plt.figure()
for i in range(len(rew_eval)):
    plt.plot(steps, rew_eval[i] , clr[i])

plt.legend(name)
# clr = ['orange','steelblue','mediumseagreen','crimson']
for i in range(len(rew_test)):
    plt.plot(steps, rew_test[i], color=clr[i], linestyle='--')
plt.grid(True)
plt.xlim([0,2.5e6])
plt.show()
