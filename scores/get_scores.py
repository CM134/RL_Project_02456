import glob
import numpy as np



scores = glob.glob('./scores/*')

arch = ['baseline','NatureDQN','base_reg', 'IMP_large','noLSTM']
modes = ['background_False', 'background_True','noise','red','permute', 'drop_rate_0.25', 'drop_rate_0.5', 'drop_rate_1']

print(arch)
print(modes)


data_rew = np.ndarray((len(arch),len(modes)))
data_rate = np.ndarray((len(arch),len(modes)))



for s in scores:
    if (s.find('rate.csv') != -1):
        for a_cnt,a_str in enumerate(arch):
            for m_cnt, m_str in enumerate(modes):
                if s.find(a_str) !=-1 and s.find(m_str) !=-1:
                    data_rate[a_cnt,m_cnt] = np.mean(np.loadtxt(s, dtype=float))
                
    
        
    elif (s.find('rewar') != -1):
        for a_cnt,a_str in enumerate(arch):
            for m_cnt, m_str in enumerate(modes):
                if s.find(a_str) !=-1 and s.find(m_str) !=-1:
                    data_rew[a_cnt,m_cnt] = np.mean(np.loadtxt(s, dtype=float))

                    

                

print('\nReward:\n',data_rew.transpose())
print('\nSuccess Rate:\n',data_rate.transpose())

        