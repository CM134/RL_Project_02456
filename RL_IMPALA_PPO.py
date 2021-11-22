# -*- coding: utf-8 -*-

"""Hyperparameters. These values should be a good starting point. You can modify them later once you have a working implementation."""

# Hyperparameters
total_steps = 21e3
print('running for', total_steps, 'steps')
num_envs = 32
num_levels = 10
num_steps = 256
num_epochs = 3
batch_size = 512
eps = .2
grad_eps = .5
value_coef = .5
entropy_coef = .01

feature_dim = 256
envname = 'coinrun'
#============================================================================================
# Define network
#============================================================================================

"""Network definitions. We have defined a policy network for you in advance. It uses the popular `NatureDQN` encoder architecture (see below), while policy and value functions are linear projections from the encodings. There is plenty of opportunity to experiment with architectures, so feel free to do that! Perhaps implement the `Impala` encoder from [this paper](https://arxiv.org/pdf/1802.01561.pdf) (perhaps minus the LSTM)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import make_env, Storage, orthogonal_init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

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
# Define environment
#============================================================================================
# check the utils.py file for info on arguments
env = make_env(num_envs, num_levels=num_levels, env_name = envname)
print('Observation space:', env.observation_space)
print('Action space:', env.action_space.n)


encoder_in = env.observation_space.shape[0]
num_actions = env.action_space.n


# Define network
encoder = Encoder(encoder_in,feature_dim)
policy = Policy(encoder,feature_dim,num_actions)
policy.cuda()

# Define optimizer
# these are reasonable values but probably not optimal
optimizer = torch.optim.Adam(policy.parameters(), lr=5e-4, eps=1e-5)

# Define temporary storage
# we use this to collect transitions during each iteration
storage = Storage(
    env.observation_space.shape,
    num_steps,
    num_envs
)
#============================================================================================
# Run training
#============================================================================================

cnt_inter = 0 

obs = env.reset()
step = 0
while step < total_steps:

  # Use policy to collect data for num_steps steps
  policy.eval()
  for _ in range(num_steps):
    

    # Use policy
    action, log_prob, value = policy.act(obs)
    
    # Take step in environment
    next_obs, reward, done, info = env.step(action)

    # Store data
    storage.store(obs, action, reward, done, info, log_prob, value)
    
    # Update current observation
    obs = next_obs

  # Add the last observation to collected data
  _, _, value = policy.act(obs)
  storage.store_last(obs, value)

  # Compute return and advantage
  storage.compute_return_advantage()

  # Optimize policy
  policy.train()
  for epoch in range(num_epochs):

    # Iterate over batches of transitions
    generator = storage.get_generator(batch_size)
    for batch in generator:
      b_obs, b_action, b_log_prob, b_value, b_returns, b_advantage = batch

      # Get current policy outputs
      new_dist, new_value = policy(b_obs)
      new_log_prob = new_dist.log_prob(b_action)

      # Clipped policy objective
      ratio = torch.exp((new_log_prob - b_log_prob))
      surr1 = ratio * b_advantage
      surr2 = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * b_advantage
      pi_loss = - torch.min(surr1, surr2).mean()

      # Clipped value function objective
      value_loss = (b_returns - new_value).pow(2).mean()

      # Entropy loss
      entropy_loss = new_dist.entropy()

      # Backpropagate losses
      loss = torch.mul(value_coef, value_loss) + pi_loss - torch.mul(entropy_coef, entropy_loss)
      loss.mean().backward()

      # Clip gradients
      torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_eps)

      # Update policy
      optimizer.step()
      optimizer.zero_grad()

  # Update stats
  step += num_envs * num_steps
  print(f'Step: {step}\tMean reward: {storage.get_reward()}')

  cnt_inter = cnt_inter + 1
  if cnt_inter == 3:
    torch.save(policy.state_dict(), 'checkpoint_intermediate.pt')
    cnt_inter = 0

print('Completed training!')
torch.save(policy.state_dict(), 'checkpoint.pt')

#============================================================================================
# Test module and create video
#============================================================================================

"""Below cell can be used for policy evaluation and saves an episode to mp4 for you to view."""

import imageio

# Make evaluation environment
eval_env = make_env(num_envs, start_level=num_levels, num_levels=num_levels, env_name = envname)
obs = eval_env.reset()

frames = []
total_reward = []

# Evaluate policy
policy.eval()
for _ in range(512):

  # Use policy
  action, log_prob, value = policy.act(obs)

  # Take step in environment
  obs, reward, done, info = eval_env.step(action)
  total_reward.append(torch.Tensor(reward))

  # Render environment and store
  frame = (torch.Tensor(eval_env.render(mode='rgb_array'))*255.).byte()
  frames.append(frame)

# Calculate average return
total_reward = torch.stack(total_reward).sum(0).mean(0)
print('Average return:', total_reward)

# Save frames as video
frames = torch.stack(frames)
imageio.mimsave('vid_new.mp4', frames, fps=25)
