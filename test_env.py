#%%
num_envs = 32
num_levels = 1000

envname = 'coinrun'

from utils import make_env, Storage, orthogonal_init

"""Below cell can be used for policy evaluation and saves an episode to mp4 for you to view."""

import matplotlib.pyplot as plt
import torch
import numpy as np


# Make evaluation environment
eval_env = make_env(num_envs, num_levels=num_levels, env_name = envname, seed=2, use_backgrounds=True)
obs = eval_env.reset()
plt.imshow(obs[0].permute(1,2,0)) 
plt.show()
# -------
# red in torch
obs = eval_env.reset()
plt.imshow(obs[0].permute(1,2,0)) 
plt.show()
img = obs
# np_rand = np.random.rand(d0,d1,d2,d3)/2
img[:,0,:,:] = obs[:,0,:,:] + 0.75
img_norm = img/img.max()
img_plt = img[0].permute(1,2,0)
print(img_plt.shape)
plt.imshow(img_plt) 
plt.show()

d0,d1,d2,d3 = list(obs.size())
#%% Random in torch
obs = eval_env.reset()
img = obs
d0,d1,d2,d3 = list(obs.size())
torch_rand = torch.from_numpy(np.random.rand(d0,d1,d2,d3)/2)
img = obs + torch_rand
img_norm = img/img.max()
img_plt = img[0].permute(1,2,0)
print(img_plt.shape)
plt.imshow(img_plt) 
plt.show()



print(d0,d1,d2,d3)
# print(np.random.rand(d0,d1,d2,d3))
# print(obs.numpy() + np.random.rand(d0,d1,d2,d3))
np_rand = np.random.rand(d0,d1,d2,d3)/2
img = obs.numpy() + np_rand
img_norm = img/img.max()
img_plt = img[0].transpose(1,2,0)
print(img_plt.shape)
plt.imshow(img_plt) 
plt.show()


#%%red
obs = eval_env.reset()
plt.imshow(obs[0].permute(1,2,0)) 
plt.show()
# np_rand = np.random.rand(d0,d1,d2,d3)/2
img[:,0,:,:] = obs[:,0,:,:] + 0.75
img_norm = img/img.max()
img_plt = img[0].permute(1,2,0)
print(img_plt.shape)
plt.imshow(img_plt) 
plt.show()

#%%green
obs = eval_env.reset()
plt.imshow(obs[0].permute(1,2,0)) 
plt.show()
# np_rand = np.random.rand(d0,d1,d2,d3)/2
img[:,1,:,:] = obs[:,1,:,:].numpy() + 0.25
img[:,2,:,:] = obs[:,2,:,:].numpy() - 0.25
img[:,0,:,:] = obs[:,0,:,:].numpy() - 0.25
img_norm = img/img.max()
img_plt = img[0].transpose(1,2,0)
print(img_plt.shape)
plt.imshow(img_plt) 
plt.show()

#%%blue
obs = eval_env.reset()
plt.imshow(obs[0].permute(1,2,0)) 
plt.show()
# np_rand = np.random.rand(d0,d1,d2,d3)/2
img[:,2,:,:] = obs[:,2,:,:].numpy() + 0.5
img_norm = img/img.max()
img_plt = img[0].transpose(1,2,0)
print(img_plt.shape)
plt.imshow(img_plt) 
plt.show()

# #%%
# for _ in range(5):
#   eval_env = make_env(1, num_levels=1000, env_name = envname, seed=np.random.randint(0,100), use_backgrounds=True)
#   obs = eval_env.reset()
#   print(obs.size)
#   # print(obs)
#   obs[0,:,:,:] = torch.from_numpy(np.uint8(obs[0,:,:,:]) + )
#   print(obs.shape)
#   # print(obs) 
#   plt.show()

# # img[:,:,0] = 255
# # img = eval_env.render(mode='rgb_array')
# # plt.imshow(img)
# plt.show()



# plt.imshow(eval_env.observation_space[0])


  # Render environment and store
  # frame = (torch.Tensor(eval_env.render(mode='rgb_array'))*255.).byte()

# %%
