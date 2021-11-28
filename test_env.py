
num_envs = 32
num_levels = 10

envname = 'coinrun'

from utils import make_env, Storage, orthogonal_init

"""Below cell can be used for policy evaluation and saves an episode to mp4 for you to view."""

import matplotlib.pyplot as plt
import torch
import numpy as np


# Make evaluation environment
#eval_env = make_env(num_envs, num_levels=num_levels, env_name = envname, seed=2)

for _ in range(5):
  eval_env = make_env(1, num_levels=1000, env_name = envname, seed=np.random.randint(0,100), use_backgrounds=True)
  eval_env.reset()

img[:,:,0] = 255
img = eval_env.render(mode='rgb_array')
plt.imshow(img)
plt.show()



# plt.imshow(eval_env.observation_space[0])


  # Render environment and store
  # frame = (torch.Tensor(eval_env.render(mode='rgb_array'))*255.).byte()
