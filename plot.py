#%%
import numpy as np
# import sys

from os import listdir
from os.path import isfile, join

import glob

filenames = sorted(glob.glob("./scores/*"))

PolicyDe_rew_eval = []
IMPlarge_rew_eval = []
IMPshort_rew_eval = []
baseline_rew_eval = []

baseline_reg_rew_eval = []
IMP_noLSTM_rew_eval = []

PolicyDe_rew_test = []
IMPlarge_rew_test = []
IMPshort_rew_test = []
baseline_rew_test = []

baseline_reg_rew_test = []
IMP_noLSTM_rew_test = []

PolicyDe_rate_eval = []
IMPlarge_rate_eval = []
IMPshort_rate_eval = []
baseline_rate_eval = []

baseline_reg_rate_eval = []
IMP_noLSTM_rate_eval = []

PolicyDe_rate_test = []
IMPlarge_rate_test = []
IMPshort_rate_test = []
baseline_rate_test = []

baseline_reg_rate_test = []
IMP_noLSTM_rate_test = []

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
                IMPlarge_rate_eval.append(data)
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
                
    elif (fname.find('reg') != -1):
        if (fname.find('eval') != -1):
            if (fname.find('reward') != -1):
                baseline_reg_rew_eval.append(data)            
            elif (fname.find('success_rate')):
                baseline_reg_rate_eval.append(data)
        elif (fname.find('test')) != -1:
            if (fname.find('reward') != -1):
                baseline_reg_rew_test.append(data)
            elif (fname.find('success_rate')):
                baseline_reg_rate_test.append(data)
                
    elif (fname.find('noLSTM') != -1):
        if (fname.find('eval') != -1):
            if (fname.find('reward') != -1):
                IMP_noLSTM_rew_eval.append(data)            
            elif (fname.find('success_rate')):
                IMP_noLSTM_rate_eval.append(data)
        elif (fname.find('test')) != -1:
            if (fname.find('reward') != -1):
                IMP_noLSTM_rew_test.append(data)
            elif (fname.find('success_rate')):
                IMP_noLSTM_rate_test.append(data)

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
        elif (fname.find('step') != -1):
            steps = data # always the same just overwrite

#%%
      
std_PolicyDe_rew_eval =  np.std(np.array(PolicyDe_rew_eval ), axis = 0)
std_IMPlarge_rew_eval =  np.std(np.array(IMPlarge_rew_eval ), axis = 0)
std_IMPshort_rew_eval =  np.std(np.array(IMPshort_rew_eval ), axis = 0)
std_baseline_rew_eval =  np.std(np.array(baseline_rew_eval ), axis = 0)

std_baseline_reg_rew_eval =  np.std(np.array(baseline_reg_rew_eval ), axis = 0)
std_IMP_noLSTM_rew_eval =  np.std(np.array(IMP_noLSTM_rew_eval ), axis = 0)

std_PolicyDe_rew_test =  np.std(np.array(PolicyDe_rew_test ), axis = 0)
std_IMPlarge_rew_test =  np.std(np.array(IMPlarge_rew_test ), axis = 0)
std_IMPshort_rew_test =  np.std(np.array(IMPshort_rew_test ), axis = 0)
std_baseline_rew_test =  np.std(np.array(baseline_rew_test ), axis = 0)

std_baseline_reg_rew_test =  np.std(np.array(baseline_reg_rew_test ), axis = 0)
std_IMP_noLSTM_rew_test =  np.std(np.array(IMP_noLSTM_rew_test ), axis = 0)

std_PolicyDe_rate_eval = np.std(np.array(PolicyDe_rate_eval), axis = 0)
std_IMPlarge_rate_eval = np.std(np.array(IMPlarge_rate_eval), axis = 0)
std_IMPshort_rate_eval = np.std(np.array(IMPshort_rate_eval), axis = 0)
std_baseline_rate_eval = np.std(np.array(baseline_rate_eval), axis = 0)

std_baseline_reg_rate_eval = np.std(np.array(baseline_reg_rate_eval), axis = 0)
std_IMP_noLSTM_rate_eval = np.std(np.array(IMP_noLSTM_rate_eval), axis = 0)

std_PolicyDe_rate_test = np.std(np.array(PolicyDe_rate_test), axis = 0)
std_IMPlarge_rate_test = np.std(np.array(IMPlarge_rate_test), axis = 0)
std_IMPshort_rate_test = np.std(np.array(IMPshort_rate_test), axis = 0)
std_baseline_rate_test = np.std(np.array(baseline_rate_test), axis = 0)

std_baseline_reg_rate_test = np.std(np.array(baseline_reg_rate_test), axis = 0)
std_IMP_noLSTM_rate_test = np.std(np.array(IMP_noLSTM_rate_test), axis = 0)


PolicyDe_rew_eval =  np.mean(np.array(PolicyDe_rew_eval ), axis = 0)
IMPlarge_rew_eval =  np.mean(np.array(IMPlarge_rew_eval ), axis = 0)
IMPshort_rew_eval =  np.mean(np.array(IMPshort_rew_eval ), axis = 0)
baseline_rew_eval =  np.mean(np.array(baseline_rew_eval ), axis = 0)

baseline_reg_rew_eval =  np.mean(np.array(baseline_reg_rew_eval ), axis = 0)
IMP_noLSTM_rew_eval =  np.mean(np.array(IMP_noLSTM_rew_eval ), axis = 0)

PolicyDe_rew_test =  np.mean(np.array(PolicyDe_rew_test ), axis = 0)
IMPlarge_rew_test =  np.mean(np.array(IMPlarge_rew_test ), axis = 0)
IMPshort_rew_test =  np.mean(np.array(IMPshort_rew_test ), axis = 0)
baseline_rew_test =  np.mean(np.array(baseline_rew_test ), axis = 0)

baseline_reg_rew_test =  np.mean(np.array(baseline_reg_rew_test ), axis = 0)
IMP_noLSTM_rew_test =  np.mean(np.array(IMP_noLSTM_rew_test ), axis = 0)

PolicyDe_rate_eval = np.mean(np.array(PolicyDe_rate_eval), axis = 0)
IMPlarge_rate_eval = np.mean(np.array(IMPlarge_rate_eval), axis = 0)
IMPshort_rate_eval = np.mean(np.array(IMPshort_rate_eval), axis = 0)
baseline_rate_eval = np.mean(np.array(baseline_rate_eval), axis = 0)

baseline_reg_rate_eval = np.mean(np.array(baseline_reg_rate_eval), axis = 0)
IMP_noLSTM_rate_eval = np.mean(np.array(IMP_noLSTM_rate_eval), axis = 0)

PolicyDe_rate_test = np.mean(np.array(PolicyDe_rate_test), axis = 0)
IMPlarge_rate_test = np.mean(np.array(IMPlarge_rate_test), axis = 0)
IMPshort_rate_test = np.mean(np.array(IMPshort_rate_test), axis = 0)
baseline_rate_test = np.mean(np.array(baseline_rate_test), axis = 0)

baseline_reg_rate_test = np.mean(np.array(baseline_reg_rate_test), axis = 0)
IMP_noLSTM_rate_test = np.mean(np.array(IMP_noLSTM_rate_test), axis = 0)



rate_eval = [PolicyDe_rate_eval, IMPlarge_rate_eval, IMPshort_rate_eval, baseline_rate_eval, baseline_reg_rate_eval, IMP_noLSTM_rate_eval]
rate_test = [PolicyDe_rate_test, IMPlarge_rate_test, IMPshort_rate_test, baseline_rate_test, baseline_reg_rate_test, IMP_noLSTM_rate_test]

rew_eval = [PolicyDe_rew_eval, IMPlarge_rew_eval, IMPshort_rew_eval, baseline_rew_eval, baseline_reg_rate_eval, IMP_noLSTM_rew_eval]
rew_test = [PolicyDe_rew_test, IMPlarge_rew_test, IMPshort_rew_test, baseline_rew_test, baseline_reg_rew_test, IMP_noLSTM_rew_test]


name = ['IMPALA+3-Layered Policy', 'IMPALA-LARGE', 'IMPALA', 'Baseline', 'Baseline + regulisation', 'IMPALA-LARGE no LSTM']

#%% Compute moving average

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

win_size = 20

for i in range(len(rate_eval)):
    rate_eval[i] = moving_average(rate_eval[i], n= win_size)
    rate_test[i] = moving_average(rate_test[i], n= win_size)
    rew_eval[i] = moving_average(rew_eval[i], n= win_size)
    rew_test[i] = moving_average(rew_test[i], n= win_size)

steps_mov = steps[:-(win_size-1)]



#%% Plot
import matplotlib.pyplot as plt
# if we have several runs we could do the following: https://stackoverflow.com/questions/12957582/plot-yerr-xerr-as-shaded-region-rather-than-error-bars

clr = ['orange','steelblue','mediumseagreen','crimson','darkred','blueviolet']
plt.figure()
for i in range(len(rate_eval)):
    plt.plot(steps_mov, rate_eval[i] , clr[i])

plt.legend(name)
# clr = ['orange','steelblue','mediumseagreen','crimson']
for i in range(len(rate_test)):
    plt.plot(steps_mov, rate_test[i], color=clr[i], linestyle='--')
plt.grid(True)
plt.xlim([0,2.5e6])
plt.show()


# clr = ['orange','steelblue','mediumseagreen','crimson']
plt.figure()
for i in range(len(rew_eval)):
    plt.plot(steps_mov, rew_eval[i] , clr[i])

plt.legend(name)
# clr = ['orange','steelblue','mediumseagreen','crimson']
for i in range(len(rew_test)):
    plt.plot(steps_mov, rew_test[i], color=clr[i], linestyle='--')
plt.grid(True)
plt.xlim([0,2.5e6])
plt.show()


# %%
