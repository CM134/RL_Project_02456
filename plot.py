#%%
import numpy as np
# import sys
# from os import listdir
# from os.path import isfile, join

import matplotlib.pyplot as plt

import glob

filenames = sorted(glob.glob("./scores/*"))

PolicyDe_rew_eval = []
IMPlarge_rew_eval = []
IMPshort_rew_eval = []
baseline_rew_eval = []
baseline_reg_rew_eval = []
IMP_noLSTM_rew_eval = []
NatureDQN_LSTM_rew_eval = []

PolicyDe_rew_test = []
IMPlarge_rew_test = []
IMPshort_rew_test = []
baseline_rew_test = []
baseline_reg_rew_test = []
IMP_noLSTM_rew_test = []
NatureDQN_LSTM_rew_test = []

PolicyDe_rate_eval = []
IMPlarge_rate_eval = []
IMPshort_rate_eval = []
baseline_rate_eval = []
baseline_reg_rate_eval = []
IMP_noLSTM_rate_eval = []
NatureDQN_LSTM_rate_eval = []


PolicyDe_rate_test = []
IMPlarge_rate_test = []
IMPshort_rate_test = []
baseline_rate_test = []
baseline_reg_rate_test = []
IMP_noLSTM_rate_test = []
NatureDQN_LSTM_rate_test = []


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

    elif (fname.find('NatureDQNwLSTM') != -1):
        if (fname.find('eval') != -1):
            if (fname.find('reward') != -1):
                NatureDQN_LSTM_rew_eval.append(data)            
            elif (fname.find('success_rate')):
                NatureDQN_LSTM_rate_eval.append(data)
        elif (fname.find('test')) != -1:
            if (fname.find('reward') != -1):
                NatureDQN_LSTM_rew_test.append(data)
            elif (fname.find('success_rate')):
                NatureDQN_LSTM_rate_test.append(data)
                
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
std_NatureDQNwLSTM_rew_eval =  np.std(np.array(NatureDQN_LSTM_rew_eval ), axis = 0)

std_PolicyDe_rew_test =  np.std(np.array(PolicyDe_rew_test ), axis = 0)
std_IMPlarge_rew_test =  np.std(np.array(IMPlarge_rew_test ), axis = 0)
std_IMPshort_rew_test =  np.std(np.array(IMPshort_rew_test ), axis = 0)
std_baseline_rew_test =  np.std(np.array(baseline_rew_test ), axis = 0)
std_baseline_reg_rew_test =  np.std(np.array(baseline_reg_rew_test ), axis = 0)
std_IMP_noLSTM_rew_test =  np.std(np.array(IMP_noLSTM_rew_test ), axis = 0)
std_NatureDQNwLSTM_rew_test =  np.std(np.array(NatureDQN_LSTM_rew_test), axis = 0)

std_PolicyDe_rate_eval = np.std(np.array(PolicyDe_rate_eval), axis = 0)
std_IMPlarge_rate_eval = np.std(np.array(IMPlarge_rate_eval), axis = 0)
std_IMPshort_rate_eval = np.std(np.array(IMPshort_rate_eval), axis = 0)
std_baseline_rate_eval = np.std(np.array(baseline_rate_eval), axis = 0)
std_baseline_reg_rate_eval = np.std(np.array(baseline_reg_rate_eval), axis = 0)
std_IMP_noLSTM_rate_eval = np.std(np.array(IMP_noLSTM_rate_eval), axis = 0)
std_NatureDQNwLSTM_rate_eval =  np.std(np.array(NatureDQN_LSTM_rate_eval ), axis = 0)

std_PolicyDe_rate_test = np.std(np.array(PolicyDe_rate_test), axis = 0)
std_IMPlarge_rate_test = np.std(np.array(IMPlarge_rate_test), axis = 0)
std_IMPshort_rate_test = np.std(np.array(IMPshort_rate_test), axis = 0)
std_baseline_rate_test = np.std(np.array(baseline_rate_test), axis = 0)
std_baseline_reg_rate_test = np.std(np.array(baseline_reg_rate_test), axis = 0)
std_IMP_noLSTM_rate_test = np.std(np.array(IMP_noLSTM_rate_test), axis = 0)
std_NatureDQNwLSTM_rate_test =  np.std(np.array(NatureDQN_LSTM_rate_test), axis = 0)

#-----

PolicyDe_rew_eval =  np.mean(np.array(PolicyDe_rew_eval ), axis = 0)
IMPlarge_rew_eval =  np.mean(np.array(IMPlarge_rew_eval ), axis = 0)
IMPshort_rew_eval =  np.mean(np.array(IMPshort_rew_eval ), axis = 0)
baseline_rew_eval =  np.mean(np.array(baseline_rew_eval ), axis = 0)
baseline_reg_rew_eval =  np.mean(np.array(baseline_reg_rew_eval ), axis = 0)
IMP_noLSTM_rew_eval =  np.mean(np.array(IMP_noLSTM_rew_eval ), axis = 0)
NatureDQN_LSTM_rew_eval =  np.mean(np.array(NatureDQN_LSTM_rew_eval), axis = 0)

PolicyDe_rew_test =  np.mean(np.array(PolicyDe_rew_test ), axis = 0)
IMPlarge_rew_test =  np.mean(np.array(IMPlarge_rew_test ), axis = 0)
IMPshort_rew_test =  np.mean(np.array(IMPshort_rew_test ), axis = 0)
baseline_rew_test =  np.mean(np.array(baseline_rew_test ), axis = 0)
baseline_reg_rew_test =  np.mean(np.array(baseline_reg_rew_test ), axis = 0)
IMP_noLSTM_rew_test =  np.mean(np.array(IMP_noLSTM_rew_test ), axis = 0)
NatureDQN_LSTM_rew_test =  np.mean(np.array(NatureDQN_LSTM_rew_test), axis = 0)


PolicyDe_rate_eval = np.mean(np.array(PolicyDe_rate_eval), axis = 0)
IMPlarge_rate_eval = np.mean(np.array(IMPlarge_rate_eval), axis = 0)
IMPshort_rate_eval = np.mean(np.array(IMPshort_rate_eval), axis = 0)
baseline_rate_eval = np.mean(np.array(baseline_rate_eval), axis = 0)
baseline_reg_rate_eval = np.mean(np.array(baseline_reg_rate_eval), axis = 0)
IMP_noLSTM_rate_eval = np.mean(np.array(IMP_noLSTM_rate_eval), axis = 0)
NatureDQN_LSTM_rate_eval =  np.mean(np.array(NatureDQN_LSTM_rate_eval), axis = 0)

PolicyDe_rate_test = np.mean(np.array(PolicyDe_rate_test), axis = 0)
IMPlarge_rate_test = np.mean(np.array(IMPlarge_rate_test), axis = 0)
IMPshort_rate_test = np.mean(np.array(IMPshort_rate_test), axis = 0)
baseline_rate_test = np.mean(np.array(baseline_rate_test), axis = 0)
baseline_reg_rate_test = np.mean(np.array(baseline_reg_rate_test), axis = 0)
IMP_noLSTM_rate_test = np.mean(np.array(IMP_noLSTM_rate_test), axis = 0)
NatureDQN_LSTM_rate_test =  np.mean(np.array(NatureDQN_LSTM_rate_test), axis = 0)


# All the data
rate_eval = [PolicyDe_rate_eval, IMPlarge_rate_eval, IMPshort_rate_eval, baseline_rate_eval, baseline_reg_rate_eval, IMP_noLSTM_rate_eval, NatureDQN_LSTM_rate_eval]
rate_test = [PolicyDe_rate_test, IMPlarge_rate_test, IMPshort_rate_test, baseline_rate_test, baseline_reg_rate_test, IMP_noLSTM_rate_test, NatureDQN_LSTM_rate_test]

rew_eval = [PolicyDe_rew_eval, IMPlarge_rew_eval, IMPshort_rew_eval, baseline_rew_eval, baseline_reg_rew_eval, IMP_noLSTM_rew_eval, NatureDQN_LSTM_rew_eval]
rew_test = [PolicyDe_rew_test, IMPlarge_rew_test, IMPshort_rew_test, baseline_rew_test, baseline_reg_rew_test, IMP_noLSTM_rew_test, NatureDQN_LSTM_rew_test]

name = ['IMPALA-small+3-Layered Policy', 'IMPALA-LARGE w. LSTM', 'IMPALA-small', 'NatureDQN', 'NatureDQN + BatchNorm', 'IMPALA-LARGE no LSTM', 'NatureDQN w. LSTM']



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

# if we have several runs we could do the following: https://stackoverflow.com/questions/12957582/plot-yerr-xerr-as-shaded-region-rather-than-error-bars

clr = ['orange','steelblue','mediumseagreen','crimson','darkred','blueviolet', 'tomato']
plt.figure()
for i in range(len(rate_eval)):
    plt.plot(steps_mov, rate_eval[i] , clr[i])

plt.legend(name)
# clr = ['orange','steelblue','mediumseagreen','crimson']
for i in range(len(rate_test)):
    plt.plot(steps_mov, rate_test[i], color=clr[i], linestyle='--')
plt.grid(True)
plt.xlim([0,2.5e6])
plt.title('Success rate')
plt.xlabel('Steps')
plt.ylabel('Success rate [%]')
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
plt.title('Reward')
plt.xlabel('Steps')
plt.ylabel('Reward')
plt.show()


# %% Cleaned plots for NatureDQN:

rate_eval = [baseline_rate_eval, baseline_reg_rate_eval, NatureDQN_LSTM_rate_eval]
rate_test = [baseline_rate_test, baseline_reg_rate_test, NatureDQN_LSTM_rate_test]

rew_eval = [baseline_rew_eval, baseline_reg_rew_eval, NatureDQN_LSTM_rew_eval]
rew_test = [baseline_rew_test, baseline_reg_rew_test, NatureDQN_LSTM_rew_test]

name = ['NatureDQN', 'NatureDQN + BatchNorm', 'NatureDQN w. LSTM']

# Apply moving average
for i in range(len(rate_eval)):
    rate_eval[i] = moving_average(rate_eval[i], n= win_size)
    rate_test[i] = moving_average(rate_test[i], n= win_size)
    rew_eval[i] = moving_average(rew_eval[i], n= win_size)
    rew_test[i] = moving_average(rew_test[i], n= win_size)
    
    
    
#Plot
clr = ['red','orange','darkred']
plt.figure()
for i in range(len(rate_eval)):
    plt.plot(steps_mov, rate_eval[i] , clr[i])

plt.legend(name)
for i in range(len(rate_test)):
    plt.plot(steps_mov, rate_test[i], color=clr[i], linestyle='--')
plt.grid(True)
plt.xlim([0,2.5e6])
plt.title('NatureDQN: Success rate')
plt.xlabel('Steps')
plt.ylabel('Success rate [%]')
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
plt.title('NatureDQN: Reward')
plt.xlabel('Steps')
plt.ylabel('Reward')
plt.show()

#%% IMPALA all


# All the data
rate_eval = [PolicyDe_rate_eval, IMPlarge_rate_eval, IMPshort_rate_eval, IMP_noLSTM_rate_eval]
rate_test = [PolicyDe_rate_test, IMPlarge_rate_test, IMPshort_rate_test, IMP_noLSTM_rate_test]

rew_eval = [PolicyDe_rew_eval, IMPlarge_rew_eval, IMPshort_rew_eval, IMP_noLSTM_rew_eval]
rew_test = [PolicyDe_rew_test, IMPlarge_rew_test, IMPshort_rew_test, IMP_noLSTM_rew_test]

name = ['IMPALA-SMALL w. LSTM + 3-Layered Policy', 'IMPALA-LARGE w. LSTM', 'IMPALA-SMALL w. LSTM', 'IMPALA-LARGE no LSTM']



# Apply moving average
for i in range(len(rate_eval)):
    rate_eval[i] = moving_average(rate_eval[i], n= win_size)
    rate_test[i] = moving_average(rate_test[i], n= win_size)
    rew_eval[i] = moving_average(rew_eval[i], n= win_size)
    rew_test[i] = moving_average(rew_test[i], n= win_size)
    
    
    
#Plot
clr = ['green','darkblue','lime','blue']
plt.figure()
for i in range(len(rate_eval)):
    plt.plot(steps_mov, rate_eval[i] , clr[i])

plt.legend(name)
for i in range(len(rate_test)):
    plt.plot(steps_mov, rate_test[i], color=clr[i], linestyle='--')
plt.grid(True)
plt.xlim([0,2.5e6])
plt.title('IMPALA: Success rate')
plt.xlabel('Steps')
plt.ylabel('Success rate [%]')
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
plt.title('IMPALA: Reward')
plt.xlabel('Steps')
plt.ylabel('Reward')
plt.show()

#%% IMPALA large


# All the data
rate_eval = [IMPlarge_rate_eval, IMP_noLSTM_rate_eval]
rate_test = [IMPlarge_rate_test, IMP_noLSTM_rate_test]

rew_eval = [IMPlarge_rew_eval, IMP_noLSTM_rew_eval]
rew_test = [IMPlarge_rew_test, IMP_noLSTM_rew_test]

name = ['IMPALA w. LSTM', 'IMPALA no LSTM']



# Apply moving average
for i in range(len(rate_eval)):
    rate_eval[i] = moving_average(rate_eval[i], n= win_size)
    rate_test[i] = moving_average(rate_test[i], n= win_size)
    rew_eval[i] = moving_average(rew_eval[i], n= win_size)
    rew_test[i] = moving_average(rew_test[i], n= win_size)
    
    
    
#Plot
clr = ['limegreen','blue']
plt.figure()
for i in range(len(rate_eval)):
    plt.plot(steps_mov, rate_eval[i] , clr[i])

plt.legend(name)
for i in range(len(rate_test)):
    plt.plot(steps_mov, rate_test[i], color=clr[i], linestyle='--')
plt.grid(True)
plt.xlim([0,2.5e6])
plt.title('IMPALA: Success rate')
plt.xlabel('Steps')
plt.ylabel('Success rate [%]')
plt.savefig('Plots/IMPALA_LARGE_rate.png')
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
plt.title('IMPALA: Reward')
plt.xlabel('Steps')
plt.ylabel('Reward')
plt.savefig('Plots/IMPALA_LARGE_reward.png')
plt.show()

#%% IMPALA vs. NatureDQN



# All the data
rate_eval = [IMPlarge_rate_eval, baseline_rate_eval, IMP_noLSTM_rate_eval, NatureDQN_LSTM_rate_eval]
rate_test = [IMPlarge_rate_test, baseline_rate_test, IMP_noLSTM_rate_test, NatureDQN_LSTM_rate_test]

rew_eval = [IMPlarge_rew_eval, baseline_rew_eval,  IMP_noLSTM_rew_eval, NatureDQN_LSTM_rew_eval]
rew_test = [IMPlarge_rew_test, baseline_rew_test,  IMP_noLSTM_rew_test, NatureDQN_LSTM_rew_test]

name = ['IMPALA-LARGE w. LSTM', 'NatureDQN', 'IMPALA-LARGE no LSTM', 'NatureDQN w. LSTM']

# Apply moving average
for i in range(len(rate_eval)):
    rate_eval[i] = moving_average(rate_eval[i], n= win_size)
    rate_test[i] = moving_average(rate_test[i], n= win_size)
    rew_eval[i] = moving_average(rew_eval[i], n= win_size)
    rew_test[i] = moving_average(rew_test[i], n= win_size)
    
    
    
#Plot
clr = ['limegreen','red','blue','darkred']
plt.figure()
for i in range(len(rate_eval)):
    plt.plot(steps_mov, rate_eval[i] , clr[i])

plt.legend(name)
for i in range(len(rate_test)):
    plt.plot(steps_mov, rate_test[i], color=clr[i], linestyle='--')
plt.grid(True)
plt.xlim([0,2.5e6])
plt.title('IMPALA vs. NatureDQN: Success rate')
plt.xlabel('Steps')
plt.ylabel('Success rate [%]')
plt.savefig('Plots/IMPALAvsDQN_rate.png')
plt.show()

plt.figure()
for i in range(len(rew_eval)):
    plt.plot(steps_mov, rew_eval[i] , clr[i])

plt.legend(name)
# clr = ['orange','steelblue','mediumseagreen','crimson']
for i in range(len(rew_test)):
    plt.plot(steps_mov, rew_test[i], color=clr[i], linestyle='--')
plt.grid(True)
plt.xlim([0,2.5e6])
plt.title('IMPALA vs. NatureDQN: Reward')
plt.xlabel('Steps')
plt.ylabel('Reward')
plt.savefig('Plots/IMPALAvsDQN_reward.png')
plt.show()
# %%

# %%

# %%
