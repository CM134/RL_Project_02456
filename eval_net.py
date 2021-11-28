#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import make_env, Storage, orthogonal_init
import glob
import imageio

from train_baseline_config import Encoder,Policy

#--------------------------------------
feature_dim = 256
envname = 'coinrun'

num_envs = 32
num_levels = 1000

make_video = False
vid_name = 'test'
runs = 512  # runs per eval mode
netname= 'baseline'
from train_baseline_config import Encoder,Policy
dictnames = glob.glob('./state_dicts/*'+netname + '*')

print(dictnames)
#%%============================================================================================
# Test module and create video
#============================================================================================


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
level_success_eval = []

obs = eval_env.reset()
frames = []
total_reward = []
for net in dictnames:
    # Load policy
    policy.load_state_dict(torch.load(net))
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
    
    if make_video == True:
        # Save frames as video
        frames = torch.stack(frames)
        imageio.mimsave((vid_name + '.mp4'), frames, fps=25)

# Calculate average return
mean_reward = torch.stack(total_reward).sum(0).mean(0)
print('Average return:', mean_reward)
# print('Levels played:', levels_played)
# print('Successful level:', level_success)
success_rate = level_success/levels_played * 100
print('Successful levels:', success_rate,'%')

#============================================================================================
# Safe each in csv file. 
#============================================================================================

import numpy as np

np.savetxt('./scores/' + netname + '_background_reward.csv', mean_reward, delimiter=', ', fmt = '% s')
np.savetxt('./scores/' + netname + '_background_rate.csv', mean_reward, delimiter=', ', fmt = '% s')


# %%
