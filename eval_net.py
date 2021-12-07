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

num_envs = 16
num_levels = 1000

make_video = False
vid_name = 'VIDEO'
runs = 512  # runs per eval mode

netname= 'base_reg'
from train_baseline_regulated_config import Encoder,Policy

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
eval_env = make_env(n_envs=num_envs, start_level= 10000, num_levels=num_levels,
                    env_name = envname, seed=2, use_backgrounds=True)

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
    levels_played = 0
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

if make_video == False:
    np.savetxt('./scores/' + netname + '_background_reward.csv', mean_reward, delimiter=', ', fmt = '% s')
    np.savetxt('./scores/' + netname + '_background_rate.csv', success_rate, delimiter=', ', fmt = '% s')

#--------------------------------------------------------------------------------------------------
# Change colors

print('\n============================')
print('Mode: Red color')
print('============================\n')

# Make evaluation environment
#eval_env = make_env(num_envs, num_levels=num_levels, env_name = envname, seed=2)
eval_env = make_env(n_envs=num_envs, start_level= 10000, num_levels=num_levels,
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
            frame[:,:,0] = frame[:,:,0] + 255*0.75
            frame = frame/frame.max()
            frames.append(frame)


    mean_reward.append(torch.stack(total_reward).sum(0).mean(0).numpy())
    success_rate.append(level_success/levels_played * 100) 

    print('Eval of net done:')
    print('Average return:', mean_reward[-1])
    print('Successful levels:', success_rate[-1],'%')

    level_success = 0
    levels_played = 0
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


if make_video == False:
    np.savetxt('./scores/' + netname + '_red_reward.csv', mean_reward, delimiter=', ', fmt = '% s')
    np.savetxt('./scores/' + netname + '_red_rate.csv', success_rate, delimiter=', ', fmt = '% s')

#--------------------------------------------------------------------------------------------------
#%% Random noise


print('\n============================')
print('Mode: Random noise')
print('============================\n')
# Make evaluation environment
#eval_env = make_env(num_envs, num_levels=num_levels, env_name = envname, seed=2)
eval_env = make_env(n_envs=num_envs, start_level= 10000, num_levels=num_levels,
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
            if reward[i] >0:
                level_success = level_success+1
                
        if make_video == True:
            # Render environment and store
            frame = (torch.Tensor(eval_env.render(mode='rgb_array'))*255.).byte()
            # apply color changes to obs
            d0,d1,d2 = list(frame.size())
            np_rand = (np.random.randint(low=0, high=255, size=(d0,d1,d2))/2)
            frame = frame.numpy() + np_rand
            frame = (frame/frame.max())*255
            frame = frame.astype(np.uint8)
            frames.append(frame)

    mean_reward.append(torch.stack(total_reward).sum(0).mean(0).numpy())
    success_rate.append(level_success/levels_played * 100) 

    print('Eval of net done:')
    print('Average return:', mean_reward[-1])
    print('Successful levels:', success_rate[-1],'%')

    level_success = 0
    levels_played = 0
    total_reward = []
    
    if make_video == True:
        # Save frames as video
        frames = np.stack(frames)
        imageio.mimsave(('noise' + vid_name + '.mp4'), frames, fps=25)
        break


print('Eval modus finished:')
# Calculate average return
print('Average return:', mean_reward)
# print('Levels played:', levels_played)
# print('Successful level:', level_success)
print('Successful levels:', success_rate,'%')

if make_video == False:
    np.savetxt('./scores/' + netname + '_noise_reward.csv', mean_reward, delimiter=', ', fmt = '% s')
    np.savetxt('./scores/' + netname + '_noise_rate.csv', success_rate, delimiter=', ', fmt = '% s')


#--------------------------------------------------------------------------------------------------
# permute observation space

print('\n============================')
print('Mode: Permute')
print('============================\n')

# Make evaluation environment
#eval_env = make_env(num_envs, num_levels=num_levels, env_name = envname, seed=2)
eval_env = make_env(n_envs=num_envs, start_level= 10000, num_levels=num_levels,
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
levels_played = 0
frames = []
total_reward = []
for net in dictnames:
    # Load policy
    policy.load_state_dict(torch.load(net))
    print('loaded', net)
    # Evaluate policy
    policy.eval()
    for _ in range(runs):
        obs = obs.permute(0,1,3,2)
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
            frame = frame.permute(1,0,2)
            frames.append(frame)
    
    mean_reward.append(torch.stack(total_reward).sum(0).mean(0).numpy())
    success_rate.append(level_success/levels_played * 100) 

    print('Eval of net done:')
    print('Average return:', mean_reward[-1])
    print('Successful levels:', success_rate[-1],'%')

    level_success = 0
    levels_played = 0
    total_reward = []

    if make_video == True:
        # Save frames as video
        frames = torch.stack(frames)
        imageio.mimsave(('permute' + vid_name + '.mp4'), frames, fps=25)
        frames = []
        break





print('Eval modus finished:')
# Calculate average return
print('Average return:', mean_reward)
# print('Levels played:', levels_played)
# print('Successful level:', level_success)
print('Successful levels:', success_rate,'%')

if make_video == False:
    np.savetxt('./scores/' + netname + '_permute_reware.csv', mean_reward, delimiter=', ', fmt = '% s')
    np.savetxt('./scores/' + netname + '_permute_rate.csv', success_rate, delimiter=', ', fmt = '% s')



#--------------------------------------------------------------------------------------------------
# Pickel Drop

print('\n============================')
print('Mode: Pixel Drop')
print('============================\n')

def drop_frame(p=0):
    n = int(p*100)
    X = np.random.randint(0, 100,(64,64))
    X[X<n] = 0
    X[X>=n] = 1
    X_ = np.repeat(np.repeat(X,8, axis=0),8, axis=1)
    X_ = np.expand_dims(X_,-1)
    return X, X_   

drop = [0.25, 0.5, 1]



# Make evaluation environment
#eval_env = make_env(num_envs, num_levels=num_levels, env_name = envname, seed=2)
eval_env = make_env(n_envs=num_envs, start_level= 10000, num_levels=num_levels,
                    env_name = envname, seed=2, use_backgrounds=False)

encoder_in = eval_env.observation_space.shape[0]
num_actions = eval_env.action_space.n

# load the policy
encoder_load = Encoder(encoder_in,feature_dim)
policy = Policy(encoder_load,feature_dim,num_actions).cuda()

for drop_rate in drop:

    print('\nDrop rate:', drop_rate)
    print('')
    level_success = 0
    levels_played = 0

    mean_reward = []
    success_rate = []

    obs = eval_env.reset()

    frames = []
    total_reward = []

    for net in dictnames:
        # Load policy
        policy.load_state_dict(torch.load(net))
        print('loaded', net)
        # Evaluate policy
        policy.eval()
        for _ in range(runs):
            # Drop pixels in obs
            X_64, X_512 = drop_frame(drop_rate)
            obs = obs*torch.tensor(X_64)

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
                frame = (torch.Tensor(X_512*eval_env.render(mode='rgb_array'))*255.).byte()
                frames.append(frame)
        
        mean_reward.append(torch.stack(total_reward).sum(0).mean(0).numpy())
        success_rate.append(level_success/levels_played * 100) 

        print('Eval of net done:')
        print('Average return:', mean_reward[-1])
        print('Successful levels:', success_rate[-1],'%')

        level_success = 0
        levels_played = 0
        total_reward = []

        if make_video == True:
            # Save frames as video
            frames = torch.stack(frames)
            imageio.mimsave(('DropPix_rate_' +str(drop_rate) + '_' + vid_name + '.mp4'), frames, fps=25)
            frames = []
            break

    print('Eval modus finished:')
    # Calculate average return
    print('Average return:', mean_reward)
    # print('Levels played:', levels_played)
    # print('Successful level:', level_success)
    print('Successful levels:', success_rate,'%')

    if make_video == False:
        np.savetxt('./scores/' + netname + '_drop_rate_' + str(drop_rate) + '_reward.csv', mean_reward, delimiter=', ', fmt = '% s')
        np.savetxt('./scores/' + netname + '_drop_rate_' + str(drop_rate) + '_rate.csv', success_rate, delimiter=', ', fmt = '% s')


