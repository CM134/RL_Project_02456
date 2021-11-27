#%%
import numpy as np
# import sys

from os import listdir
from os.path import isfile, join

import glob

from numpy.lib.function_base import append

filenames = sorted(glob.glob("./scores/*"))


PolicyDe_rew_eval = []
IMPlarge_rew_eval = []
IMPshort_rew_eval = []
baseline_rew_eval = []

PolicyDe_rew_test = []
IMPlarge_rew_test = []
IMPshort_rew_test = []
baseline_rew_test = []

PolicyDe_rate_eval = []
IMPlarge_rate_eval = []
IMPshort_rate_eval = []
baseline_rate_eval = []

PolicyDe_rate_test = []
IMPlarge_rate_test = []
IMPshort_rate_test = []
baseline_rate_test = []

steps = []
name = []


for fname in filenames:
    data = np.loadtxt(fname, dtype=float)
    # one data set is trained on 5e6 steps. Meaning fails if different length
    if len(data) > 306:
        data = data[:306]
    if (fname.find('large') != -1):
        if (fname.find('eval') != -1):
            if (fname.find('reward') != -1):
                IMPlarge_rew_eval.append(data)
            elif (fname.find('success_rate')):
                IMPlarge_rate_eval.appen(data)
        elif (fname.find('test')) != -1:
            if (fname.find('reward') != -1):
                IMPlarge_rew_test.append(data)
            elif (fname.find('success_rate')):
                IMPlarge_rate_test.append(data)

    elif (fname.find('baseline') != -1):
        if (fname.find('eval') != -1):
            if (fname.find('reward') != -1):
                baseline_rew_eval.append(data)
            elif (fname.find('success_rate')):
                baseline_rate_eval.append(data)
        elif (fname.find('test')) != -1:
            if (fname.find('reward') != -1):
                baseline_rew_test.append(data)
            elif (fname.find('success_rate')):
                baseline_rate_test.append(data)

    elif (fname.find('policy') != -1):
        if (fname.find('eval') != -1):
            if (fname.find('reward') != -1):
                PolicyDe_rew_eval.append(data)            
            elif (fname.find('success_rate')):
                PolicyDe_rate_eval.append(data)
        elif (fname.find('test')) != -1:
            if (fname.find('reward') != -1):
                PolicyDe_rew_test.append(data)
            elif (fname.find('success_rate')):
                PolicyDe_rate_test.append(data)

    elif (fname.find('short') != -1):
        if (fname.find('eval') != -1):
            if (fname.find('reward') != -1):
                IMPshort_rew_eval.append(data)
            elif (fname.find('success_rate')):
                IMPshort_rate_eval.append(data)
        elif (fname.find('test')) != -1:
            if (fname.find('reward') != -1):
                IMPshort_rew_test.append(data)
            elif (fname.find('success_rate')):
                IMPshort_rate_test.append(data)


PolicyDe_rew_eval =  np.mean(np.array(PolicyDe_rew_eval ), axis = 0)
IMPlarge_rew_eval =  np.mean(np.array(IMPlarge_rew_eval ), axis = 0)
IMPshort_rew_eval =  np.mean(np.array(IMPshort_rew_eval ), axis = 0)
baseline_rew_eval =  np.mean(np.array(baseline_rew_eval ), axis = 0)
PolicyDe_rew_test =  np.mean(np.array(PolicyDe_rew_test ), axis = 0)
IMPlarge_rew_test =  np.mean(np.array(IMPlarge_rew_test ), axis = 0)
IMPshort_rew_test =  np.mean(np.array(IMPshort_rew_test ), axis = 0)
baseline_rew_test =  np.mean(np.array(baseline_rew_test ), axis = 0)
PolicyDe_rate_eval = np.mean(np.array(PolicyDe_rate_eval), axis = 0)
IMPlarge_rate_eval = np.mean(np.array(IMPlarge_rate_eval), axis = 0)
IMPshort_rate_eval = np.mean(np.array(IMPshort_rate_eval), axis = 0)
baseline_rate_eval = np.mean(np.array(baseline_rate_eval), axis = 0)
PolicyDe_rate_test = np.mean(np.array(PolicyDe_rate_test), axis = 0)
IMPlarge_rate_test = np.mean(np.array(IMPlarge_rate_test), axis = 0)
IMPshort_rate_test = np.mean(np.array(IMPshort_rate_test), axis = 0)
baseline_rate_test = np.mean(np.array(baseline_rate_test), axis = 0)

rate_eval = [PolicyDe_rate_eval, IMPlarge_rate_eval, IMPshort_rate_eval, baseline_rate_eval]
rate_test = [PolicyDe_rate_test, IMPlarge_rate_test, IMPshort_rate_test, baseline_rate_test]

#%% Plot
import matplotlib.pyplot as plt
# if we have several runs we could do the following: https://stackoverflow.com/questions/12957582/plot-yerr-xerr-as-shaded-region-rather-than-error-bars

clr = ['orange','steelblue','mediumseagreen','crimson']
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


