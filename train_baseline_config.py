# -*- coding: utf-8 -*-

"""Hyperparameters. These values should be a good starting point. You can modify them later once you have a working implementation."""

# Hyperparameters

import sys
import json

try:
  num = str(sys.argv[2]) # number to set behind names to indicate which run it is we run it 
except:
  num = None
try:
    inFile = sys.argv[1]
    
    f = open(inFile)
    data = json.load(f)
    print(data)
    total_steps = data["total_steps"]
    learning_rate = data["learning_rate"]
    epsilon = data["epsilon"]
    state_dict_name = data["state_dict_name"]+'.pt'
    outfile_name = data["state_dict_name"]
    

except:
    print('loading form config failed: using default')
    total_steps = 21e3
    learning_rate = 5e-4
    epsilon = 1e-5
    state_dict_name = 'check.pt'
    outfile_name = 'bl'
    
if num is not None:
  state_dict_name = num + '_' + state_dict_name
  outfile_name = num + '_' + outfile_name 
  
    
num_envs = 32
num_levels = 100
num_steps = 256
num_epochs = 3
batch_size = 512
ppo_eps = .2
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
        nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4), nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2), nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1), nn.ReLU(),
        Flatten(),
        nn.Linear(in_features=1024, out_features=feature_dim), nn.ReLU()
    )
    self.apply(orthogonal_init)

  def forward(self, x):
    return self.layers(x)


class Policy(nn.Module):
  def __init__(self, encoder, feature_dim, num_actions):
    super().__init__()
    self.encoder = encoder
    self.policy = orthogonal_init(nn.Linear(feature_dim, num_actions), gain=.01)
    self.value = orthogonal_init(nn.Linear(feature_dim, 1), gain=1.)

  def act(self, x):
    with torch.no_grad():
      x = x.cuda().contiguous()
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


if __name__ == "__main__":
  #-------------------------------------------------------------------------------------------
  # Define environment
  #-------------------------------------------------------------------------------------------
  # check the utils.py file for info on arguments
  env      = make_env(n_envs=num_envs, num_levels=num_levels,   env_name = envname, seed = 0, start_level=0)
  eval_env = make_env(n_envs=num_envs, num_levels=num_levels, env_name = envname, seed = 0, start_level=num_levels)

  encoder_in = env.observation_space.shape[0]
  num_actions = env.action_space.n

  # Define network
  encoder = Encoder(in_channels=encoder_in, feature_dim=feature_dim)
  policy = Policy(encoder=encoder, feature_dim=feature_dim, num_actions=env.action_space.n)
  policy.cuda()

  # Define optimizer
  # these are reasonable values but probably not optimal
  optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate, eps=epsilon)

  # Define temporary storage
  # we use this to collect transitions during each iteration
  storage = Storage(
      env.observation_space.shape,
      num_steps,
      num_envs
  )
  #-------------------------------------------------------------------------------------------
  # Run training
  #-------------------------------------------------------------------------------------------
  print('Start Training...')

  cnt_inter = 0 
  levels_played_test = 0
  level_success_test = 0
  levels_played_eval = 0
  level_success_eval = 0
  eval_reward = []
  success_rate_test_list = []
  success_rate_eval_list = []
  eval_reward_list = []
  test_reward_list = []
  step_list = []

  obs = env.reset()
  eval_obs = eval_env.reset()
  step = 0

  while step < total_steps:
    # ----------------------------------------------------------------
    # Use policy to collect data for num_steps steps
    policy.eval()
    for _ in range(num_steps):
      
      # Use policy
      action, log_prob, value = policy.act(obs)
      
      # Take step in environment
      next_obs, reward, done, info = env.step(action)

      for i in range(len(reward)):
        if done[i] == 1:
          levels_played_test = levels_played_test + 1 
          if reward[i] > 0:
            level_success_test = level_success_test+1

      # Store data
      storage.store(obs, action, reward, done, info, log_prob, value)
      
      # Update current observation
      obs = next_obs

    # Add the last observation to collected data
    _, _, value = policy.act(obs)
    storage.store_last(obs, value)

    # Compute return and advantage
    storage.compute_return_advantage()
    # ----------------------------------------------------------------
    # Eval policy
    for _ in range(num_steps):
      
      # Use policy
      action, log_prob, value = policy.act(eval_obs)
      
      # Take step in environment
      eval_obs, reward, done, info = eval_env.step(action)
      eval_reward.append(torch.Tensor(reward))

      for i in range(len(reward)):
        if done[i] == 1:
          levels_played_eval = levels_played_eval + 1 
          if reward[i] >0:
            level_success_eval = level_success_eval + 1
    # ----------------------------------------------------------------
    # safe training and eval  
    mean_eval_reward = torch.stack(eval_reward).sum(0).mean(0)
    print(f'Step: {step+num_envs*num_steps}')
    print(f'Mean test reward: {storage.get_reward()}')
    print(f'Mean eval reward: {mean_eval_reward}')
    print('Successful test levels:', level_success_test/levels_played_test * 100 ,'%')
    print('Successful eval levels:', level_success_eval/levels_played_eval * 100 ,'%')

    test_reward_list.append(storage.get_reward())
    eval_reward_list.append(mean_eval_reward)
    success_rate_test_list.append(level_success_test/levels_played_test * 100 )
    success_rate_eval_list.append(level_success_eval/levels_played_eval * 100 )

    levels_played_test = 0
    level_success_test = 0
    levels_played_eval = 0
    level_success_eval = 0 
    eval_reward = []

    # ----------------------------------------------------------------
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
        surr2 = torch.clamp(ratio, 1.0 - ppo_eps, 1.0 + ppo_eps) * b_advantage
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
    step_list.append(step)
    
    cnt_inter = cnt_inter + 1
    if cnt_inter == 1:
      torch.save(policy.state_dict(), './state_dicts/inter_' + state_dict_name)


  print('Completed training!')
  torch.save(policy.state_dict(), './state_dicts/' + state_dict_name)


  #============================================================================================
  # Safe each in csv file. 
  #============================================================================================

  import numpy as np

  np.savetxt('./scores/' + outfile_name+'_test_reward.csv', test_reward_list, delimiter=', ', fmt = '% s')
  np.savetxt('./scores/' + outfile_name+'_eval_reward.csv', eval_reward_list, delimiter=', ', fmt = '% s')
  np.savetxt('./scores/' + outfile_name+'_success_rate_test.csv', success_rate_test_list, delimiter=', ', fmt = '% s')
  np.savetxt('./scores/' + outfile_name+'_success_rate_eval.csv', success_rate_eval_list, delimiter=', ', fmt = '% s')
  np.savetxt('./scores/' + outfile_name+'_steps.csv', step_list, delimiter=', ', fmt = '% s')
