
num_envs = 32
num_levels = 10

feature_dim = 256
envname = 'coinrun'

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import make_env, Storage, orthogonal_init

#============================================================================================
# Define Network Architecture
#============================================================================================

class Encoder(nn.Module):
  def __init__(self, in_channels, feature_dim):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=8, stride=4), nn.ReLU(),
        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2), nn.ReLU(),
        Flatten(),
        nn.Linear(in_features=1152, out_features=256), nn.ReLU()
    )
    self.LSTM = nn.Sequential(
        nn.LSTM(input_size=256, hidden_size=256, num_layers=1)
    )
    self.apply(orthogonal_init)

  def forward(self, x):
    out=self.layers(x)
    out = out.view(1,out.shape[0],out.shape[1])
    out,(h_n, c_n) = self.LSTM(out)
    #print(out.shape)
    
    return out.squeeze(0)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Policy(nn.Module):
  def __init__(self, encoder, feature_dim, num_actions):
    super().__init__()
    self.encoder = encoder
    self.policy = nn.Sequential(
        nn.Linear(feature_dim, 256), nn.ReLU(),
        nn.Linear(256, 126), nn.ReLU(),
        nn.Linear(126, num_actions)
    )
    self.value = nn.Sequential(
        nn.Linear(feature_dim, 256), nn.ReLU(),
        nn.Linear(feature_dim, 1)
    )

  def act(self, x):
    with torch.no_grad():
      x = x.cuda().contiguous()
      x = x.contiguous()
      dist, value = self.forward(x)
      action = dist.sample()
      log_prob = dist.log_prob(action)
    
    return action.cpu(), log_prob.cpu(), value.cpu()

  def forward(self, x):
    x = self.encoder(x)
    logits = self.policy(x)
    value = self.value(x).squeeze(1)
    dist = torch.distributions.Categorical(logits=logits)

    return dist, value


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
policy.load_state_dict(torch.load('./state_dicts/sd1.pt'))
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

