

state_dict_name = 'baseline_std.pt'

num_envs = 32
num_levels = 10

feature_dim = 256
envname = 'coinrun'

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import make_env, Storage, orthogonal_init

#===========================================================================================
from train_baseline_config import Encoder,Policy

#============================================================================================
# Test module and create video
#============================================================================================

"""Below cell can be used for policy evaluation and saves an episode to mp4 for you to view."""

import imageio

# Make evaluation environment
#eval_env = make_env(num_envs, num_levels=num_levels, env_name = envname, seed=2)
eval_env = make_env(1, num_levels=1000, env_name = envname, seed=2)

encoder_in = eval_env.observation_space.shape[0]
num_actions = eval_env.action_space.n

# load the policy
encoder_load = Encoder(encoder_in,feature_dim)
policy = Policy(encoder_load,feature_dim,num_actions).cuda()
policy.load_state_dict(torch.load('./state_dicts/' + state_dict_name))
print(policy.eval())

level_success = 0
levels_played = 0

level_success_eval = []

obs = eval_env.reset()

frames = []
total_reward = []

# Evaluate policy
policy.eval()
for _ in range(4096):

  # Use policy
  action, log_prob, value = policy.act(obs)

  # Take step in environment
  obs, reward, done, info = eval_env.step(action)
  total_reward.append(torch.Tensor(reward))

  for i in range(len(reward)):
      if done[i] == 1:
        levels_played = levels_played + 1 
        if reward[i] >0:
          level_success = level_success+1

  # Render environment and store
  frame = (torch.Tensor(eval_env.render(mode='rgb_array'))*255.).byte()
  frames.append(frame)

# Calculate average return
total_reward = torch.stack(total_reward).sum(0).mean(0)
print('Average return:', total_reward)

# Save frames as video
frames = torch.stack(frames)
imageio.mimsave('vid_new.mp4', frames, fps=25)

print('Levels played:', levels_played)
print('Successful level:', level_success)
print('Successful levels:', level_success/levels_played * 100 ,'%')

