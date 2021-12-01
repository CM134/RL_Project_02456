#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import make_env, Storage, orthogonal_init
import glob
import imageio
import numpy as np

#--------------------------------------
feature_dim = 256
envname = 'coinrun'

num_envs = 1
num_levels = 1000

make_video = True
vid_name = 'noLSTM'
runs = 512  # runs per eval mode
netname= 'noLSTM'
from train_IMPALAlarge_no_LSTM_config import Encoder,Policy
dictnames = glob.glob('./state_dicts/*'+netname + '*')

print(dictnames)

#%%============================================================================================
# Start Eval
#==============================================================================================
print('\n============================')
print('Mode: Background')
print('============================\n')

# Make evaluation environment
#eval_env = make_env(num_envs, num_levels=num_levels, env_name = envname, seed=2)
eval_env = make_env(n_envs=num_envs, start_level= 10000, num_levels=1000,
                    env_name = envname, seed=2, use_backgrounds=False)

encoder_in = eval_env.observation_space.shape[0]
num_actions = eval_env.action_space.n

# load the policy
encoder_load = Encoder(encoder_in,feature_dim)
policy = Policy(encoder_load,feature_dim,num_actions).cuda()

level_success = 0
levels_played = 0

mean_reward = []
success_rate = []

obs = eval_env.reset()
levels_played = len(obs)
frames = []
total_reward = []
for net in dictnames:
    # Load policy
    policy.load_state_dict(torch.load(net))
    print('loaded', net)
    # Evaluate policy
    policy.eval()
    for _ in range(runs):
        # Use policy
        action, log_prob, value = policy.act(obs)
        # Take step in environment
        obs, reward, done, info = eval_env.step(action)
        total_reward.append(torch.Tensor(reward))
        # determine successful levels
        for i in range(len(reward)):
            if done[i] == 1:
                levels_played = levels_played + 1 
            if reward[i] >0:
                level_success = level_success+1
                
        if make_video == True:
            # Render environment and store
            frame = (torch.Tensor(eval_env.render(mode='rgb_array'))*255.).byte()
            frames.append(frame)
    
    mean_reward.append(torch.stack(total_reward).sum(0).mean(0).numpy())
    success_rate.append(level_success/levels_played * 100) 

    print('Eval of net done:')
    print('Average return:', mean_reward[-1])
    print('Successful levels:', success_rate[-1],'%')

    level_success = 0
    levels_played = len(obs)
    total_reward = []

    if make_video == True:
        # Save frames as video
        frames = torch.stack(frames)
        imageio.mimsave(('background' + vid_name + '.mp4'), frames, fps=25)
        frames = []
        break





print('Eval modus finished:')
# Calculate average return
print('Average return:', mean_reward)
# print('Levels played:', levels_played)
# print('Successful level:', level_success)
print('Successful levels:', success_rate,'%')

import numpy as np

np.savetxt('./scores/' + netname + '_background_reward.csv', mean_reward, delimiter=', ', fmt = '% s')
np.savetxt('./scores/' + netname + '_background_rate.csv', success_rate, delimiter=', ', fmt = '% s')

#--------------------------------------------------------------------------------------------------
# Change colors

print('\n============================')
print('Mode: Red color')
print('============================\n')

# Make evaluation environment
#eval_env = make_env(num_envs, num_levels=num_levels, env_name = envname, seed=2)
eval_env = make_env(n_envs=num_envs, start_level= 10000, num_levels=1000,
                    env_name = envname, seed=2, use_backgrounds=False)

encoder_in = eval_env.observation_space.shape[0]
num_actions = eval_env.action_space.n

# load the policy
encoder_load = Encoder(encoder_in,feature_dim)
policy = Policy(encoder_load,feature_dim,num_actions).cuda()

level_success = 0
levels_played = 0

mean_reward = []
success_rate = []

obs = eval_env.reset()
levels_played = len(obs)
frames = []
total_reward = []
for net in dictnames:
    # Load policy
    policy.load_state_dict(torch.load(net))
    print('loaded', net)
    # Evaluate policy
    policy.eval()
    for _ in range(runs):
        # apply color changes to obs
        obs[:,0,:,:] = obs[:,0,:,:] + 0.75
        obs = obs/obs.max()
        # Use policy
        action, log_prob, value = policy.act(obs)
        # Take step in environment
        obs, reward, done, info = eval_env.step(action)
        total_reward.append(torch.Tensor(reward))
        # determine successful levels
        for i in range(len(reward)):
            if done[i] == 1:
                levels_played = levels_played + 1 
            if reward[i] >0:
                level_success = level_success+1
                
        if make_video == True:
            # Render environment and store
            frame = (torch.Tensor(eval_env.render(mode='rgb_array'))*255.).byte()
            frames.append(frame)


    mean_reward.append(torch.stack(total_reward).sum(0).mean(0).numpy())
    success_rate.append(level_success/levels_played * 100) 

    print('Eval of net done:')
    print('Average return:', mean_reward[-1])
    print('Successful levels:', success_rate[-1],'%')

    level_success = 0
    levels_played = len(obs)
    total_reward = []
    
    if make_video == True:
        # Save frames as video
        frames = torch.stack(frames)
        imageio.mimsave(('red' + vid_name + '.mp4'), frames, fps=25)
        break


print('Eval modus finished:')
# Calculate average return
print('Average return:', mean_reward)
# print('Levels played:', levels_played)
# print('Successful level:', level_success)
print('Successful levels:', success_rate,'%')

import numpy as np

np.savetxt('./scores/' + netname + '_red_reward.csv', mean_reward, delimiter=', ', fmt = '% s')
np.savetxt('./scores/' + netname + '_red_rate.csv', success_rate, delimiter=', ', fmt = '% s')

#--------------------------------------------------------------------------------------------------
#%% Random noise


print('\n============================')
print('Mode: Random noise')
print('============================\n')
# Make evaluation environment
#eval_env = make_env(num_envs, num_levels=num_levels, env_name = envname, seed=2)
eval_env = make_env(n_envs=num_envs, start_level= 10000, num_levels=1000,
                    env_name = envname, seed=2, use_backgrounds=False)

encoder_in = eval_env.observation_space.shape[0]
num_actions = eval_env.action_space.n

# load the policy
encoder_load = Encoder(encoder_in,feature_dim)
policy = Policy(encoder_load,feature_dim,num_actions).cuda()

level_success = 0
levels_played = 0

mean_reward = []
success_rate = []

obs = eval_env.reset()

levels_played = len(obs)

frames = []
total_reward = []
for net in dictnames:
    # Load policy
    policy.load_state_dict(torch.load(net))
    print('loaded', net)
    # Evaluate policy
    policy.eval()
    for _ in range(runs):
        # apply color changes to obs
        d0,d1,d2,d3 = list(obs.size())
        torch_rand = torch.from_numpy(np.random.rand(d0,d1,d2,d3)/2)
        obs = obs + torch_rand
        obs = obs/obs.max()
        obs = obs.type(torch.FloatTensor) # needs to be casted to fit weights
        # Use policy
        action, log_prob, value = policy.act(obs)
        # Take step in environment
        obs, reward, done, info = eval_env.step(action)
        total_reward.append(torch.Tensor(reward))
        # determine successful levels
        for i in range(len(reward)):
            if done[i] == 1:
                levels_played = levels_played + 1
                #print(levels_played)
            if reward[i] >0:
                level_success = level_success+1
                # print(level_success, reward[i])
                
        if make_video == True:
            # Render environment and store
            frame = (torch.Tensor(eval_env.render(mode='rgb_array'))*255.).byte()
            frames.append(frame)

    mean_reward.append(torch.stack(total_reward).sum(0).mean(0).numpy())
    success_rate.append(level_success/levels_played * 100) 

    print('Eval of net done:')
    print('Average return:', mean_reward[-1])
    print('Successful levels:', success_rate[-1],'%')

    level_success = 0
    levels_played = len(obs)
    total_reward = []
    
    if make_video == True:
        # Save frames as video
        frames = torch.stack(frames)
        imageio.mimsave(('noise' + vid_name + '.mp4'), frames, fps=25)
        break


print('Eval modus finished:')
# Calculate average return
print('Average return:', mean_reward)
# print('Levels played:', levels_played)
# print('Successful level:', level_success)
print('Successful levels:', success_rate,'%')

import numpy as np

np.savetxt('./scores/' + netname + '_noise_reward.csv', mean_reward, delimiter=', ', fmt = '% s')
np.savetxt('./scores/' + netname + '_noise_rate.csv', success_rate, delimiter=', ', fmt = '% s')
