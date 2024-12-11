# -*- coding: utf-8 -*-
import os
from torch import nn
import torch
import gym
import math
from collections import deque
import itertools
import numpy as np
import random
from typing import Tuple

#from IPython import display as ipythondisplay
#import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

import msgpack
from msgpack_numpy import patch as msgpack_numpy_patch
msgpack_numpy_patch()

from tqdm import tqdm
from gym import Env, spaces

GAMMA = 0.99 # Gamma is our discount rate for computing our temporal difference target
BATCH_SIZE = 32 #how many transitions we're going to sample from the replay buffer
BUFFER_SIZE = 50000 # the maximum number of transitions we're going to store before overwriting old transitions
MIN_REPLAY_SIZE = 1000 # how many transitons we want in the buffer before we start computing gradients and doing training
EPSILON_START = 1.0  # the start value, not a constant, going to be decayed
EPSILON_END = 0.02   # the end value
EPSILON_DECAY = 10000 # decay over this many steps (start to end)
TARGET_UPDATE_FREQ = 1000 # the number of steps where we set the tartget parameters equal to the online parameters
LEARNING_RATE = 5e-4
LOG_DIR = './logs/x' 
LOG_INTERVAL = 1000
SAVE_PATH = './x_model'
SAVE_INTERVAL = 10000
REWARD_BUFFER_SIZE = 100
EPISODES = 30000
TYPE_DIC ={'=':0, '>':1, '<':2}

class Platform:
    def __init__(self, size):
        self.size = size
        self.a = np.random.randint(1,size+1)
        self.b = np.random.randint(1,size+1)
    
    def action(self, choice):
        '''
        Gives us 9 total movement options. (0,1,2,3,4,5,6,7,8)
        ''' 
        if choice == 0:
            self.move(a=-1, b=-1)
        elif choice == 1:
            self.move(a=-1, b=0)
        elif choice == 2:
            self.move(a=-1, b=1)
        elif choice == 3:
            self.move(a=0, b=-1)
        elif choice == 4:
            self.move(a=0, b=0)
        elif choice == 5:
            self.move(a=0, b=1)
        elif choice == 6:
            self.move(a=1, b=-1)
        elif choice == 7:
            self.move(a=1, b=0)
        elif choice == 8:
            self.move(a=1, b=1)
    
    def move(self, a, b):
        self.a += a
        self.b += b
        # If we are out of bounds, fix!
        if self.a < 1:
            self.a = 1
        elif self.a > self.size:
            self.a = self.size
        if self.b < 1:
            self.b = 1
        elif self.b > self.size:
            self.b = self.size


class Company:
    def __init__(self, size, n):
        self.size = size
        self.n = n
        self.cs = np.random.choice(a=np.arange(1,size+1), size = n)
    
    def action(self, values):
        for i in range(self.n):
            if values[i]>0 and self.cs[i]<self.size:
                self.cs[i]+=1
            if values[i]<0 and self.cs[i]>1:
                self.cs[i]-=1


class LSEnv(Env):
    def __init__(self, user_number, type):
        super(LSEnv, self).__init__()
        self.psize = 9
        self.csize = 9
        self.cnum = 5
        self.observation_shape = 10
        self.user_number = user_number
        self.type = type
        #self.observation_space = spaces.Tuple((Discrete(self.psize, start=1),Discrete(self.psize, start=1),Discrete(self.csize, start=1),Discrete(self.csize, start=1),Discrete(self.csize, start=1),Discrete(self.csize, start=1),Discrete(self.csize, start=1)))
        self.action_space = spaces.Discrete(self.psize)
        self.done_flags = np.zeros(2,dtype=int)
    
    def reset(self):
        self.platform = Platform(self.psize)
        self.company = Company(self.csize, self.cnum)
        self.pd = np.random.dirichlet(np.ones(3),size=1)[0]
        self.cvalues = self.computeCvalue()
        return self.getState()

    def step(self, action):
        self.platform.action(action)
        self.company.action(self.cvalues)
        
        done = self.q_evolution()

        pvalue = self.computePvalue()

        if pvalue > 0:
            reward = 1.0
        elif pvalue < 0:
            reward = -1.0

        info = {}
        
        new_state = self.getState()

        return new_state, reward, done, info
    

    def getState(self):
        state = [self.platform.a, self.platform.b]
        for item in self.company.cs:
            state.append(item)
        for item in self.pd:
            state.append(int(item*self.user_number))
        return state

    
    def computePvalue(self):
        b = sum(self.company.cs) + sum([math.log(v,k+1) for k,v in zip([1,2,3],self.pd*self.user_number)])
        c = (self.platform.a+self.platform.b) * math.log(self.user_number, 10)
        return b-c
    
    def computeCvalue(self):
        boef = sum([(3-k)*math.log(v,k+1) for k,v in zip([1,2,3],self.pd*self.user_number)]) * (math.log(self.platform.a,10)+1)
        c = self.company.cs
        b = np.array([boef*r/sum(c) for r in c])
        return b-c
    
    def q_evolution(self):
        done = False
        a = self.platform.a
        b = self.platform.b
        if a==b:
            if self.type == '=':
                if self.pd[2] < 0.5:
                    temps = np.random.random(2)/10
                    self.pd[0] -= temps[0]
                    self.pd[1] -= temps[1]
                    self.pd[2] += sum(temps)
                elif self.pd[2] >0.5:
                    temps = np.random.random(2)/10
                    self.pd[0] += temps[0]
                    self.pd[1] += temps[1]
                    self.pd[2] -= sum(temps)
                    
                if (self.pd <= 0).any():
                    self.pd[self.pd<=0] = 0.01
                self.pd = np.array([ r/sum(self.pd) for r in self.pd]) 
                
                if self.pd[2]>=0.47 and self.pd[2]<=0.52:
                    done = True
                    self.done_flags[0]+=1

            elif self.type == '>': 
                temps = np.random.random(2)/10
                if self.pd[0] <= 0.5:
                    self.pd[0] += temps[0]
                if self.pd[1] >= 0.1:
                    self.pd[1] -= temps[1]
                self.pd[2] = 1- self.pd[0] - self.pd[1]
                
                if (self.pd <= 0).any():
                    self.pd[self.pd<=0] = 0.01
                self.pd = np.array([ r/sum(self.pd) for r in self.pd]) 

            elif self.type == '<':
                temps = np.random.random(2)/10
                if self.pd[2] <= 0.5:
                    self.pd[2] += temps[0]
                if self.pd[0] >= 0.1:
                    self.pd[0] -= temps[1]
                self.pd[1] = 1- self.pd[0] - self.pd[2]

                if (self.pd <= 0).any():
                    self.pd[self.pd<=0] = 0.01
                self.pd = np.array([ r/sum(self.pd) for r in self.pd]) 
            
        elif a<b:
            self.pd[0]*=np.random.random(1)
            self.pd[1]*=np.random.random(1)
            self.pd[2] = 1-self.pd[0]-self.pd[1]

        elif a>b:
            self.pd[1]*=np.random.random(1)
            self.pd[2]*=np.random.random(1)
            self.pd[0] = 1-self.pd[1]-self.pd[2]

        if (self.pd <= 0.001).any():
            done = True
            self.done_flags[1]+=1
        
        if (self.pd >= 0.997).any():
            done = True
            self.done_flags[1]+=1
        
        return done

   



class Network(nn.Module):
    def __init__(self, env, device):
        super().__init__()

        # compute the number of inputs to this network (how many neurons are in the input layer)
        in_features = env.observation_shape
        self.num_actions = env.action_space.n
        self.device = device

        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_actions))

    def forward(self, x):
        return self.net(x)
    
    def act(self, obs):
        obs_t = torch.as_tensor(obs, device=self.device, dtype=torch.float32) # it's a torch tensor
        q_values = self(obs_t.unsqueeze(0))
        max_q_index = torch.argmax(q_values, dim=1)[0]
        action = max_q_index.detach().item()  # turn this torch tensor into an integer

        return action
    
    def compute_loss(self, transitions, target_net):
        # Grap Tuple data from transitions
        obses = np.asarray([t[0] for t in transitions])
        actions = np.asarray([t[1] for t in transitions])
        rewards = np.asarray([t[2] for t in transitions])
        dones = np.asarray([t[3] for t in transitions])
        new_obses = np.asarray([t[4] for t in transitions])

        obses_t = torch.as_tensor(obses, device=self.device, dtype=torch.float32)
        actions_t = torch.as_tensor(actions, device=self.device, dtype=torch.int64).unsqueeze(-1)
        rewards_t = torch.as_tensor(rewards, device=self.device, dtype=torch.float32).unsqueeze(-1)
        dones_t = torch.as_tensor(dones, device=self.device, dtype=torch.float32).unsqueeze(-1)
        new_obses_t = torch.as_tensor(new_obses, device=self.device, dtype=torch.float32)

        # Compute Targets
        target_q_values = target_net(new_obses_t)
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
        targets = rewards_t + GAMMA*(1-dones_t)*max_target_q_values


        # Compute Loss
        q_values = self(obses_t)
        action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)
        loss = nn.functional.smooth_l1_loss(action_q_values, targets)

        return loss
    
    def save(self, save_path):
        params = {k: t.detach().cpu().numpy() for k, t in self.state_dict().items()}
        params_data = msgpack.dumps(params)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(params_data)

    def load(self, load_path):
        if not os.path.exists(load_path):
            raise FileNotFoundError(load_path)

        with open(load_path, 'rb') as f:
            params_numpy = msgpack.loads(f.read())

        params = {k: torch.as_tensor(v, device=self.device) for k,v in params_numpy.items()}

        self.load_state_dict(params)



if __name__ == '__main__':
    # get the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # define the env
    #env = gym.make('CartPole-v0') # define the experimental environment
    env = LSEnv(user_number=60000, type='>')

    replay_buffer = deque(maxlen=BUFFER_SIZE) # define the replay buffer
    
    reward_buffer = deque([0.0], maxlen=REWARD_BUFFER_SIZE)  # a reward buffer store the rewards earned by our agent in a single episode

    episode_reward = 0.0
    episode_count = 0

    summary_writer = SummaryWriter(LOG_DIR+str(TYPE_DIC[env.type]))

    online_net = Network(env, device=device)
    target_net = Network(env, device=device)

    online_net = online_net.to(device)
    target_net = target_net.to(device)

    target_net.load_state_dict(online_net.state_dict())
    
    optimizer = torch.optim.Adam(online_net.parameters(), lr=LEARNING_RATE)

    # Initialize Replay Buffer
    obs = env.reset()
    for _ in range(MIN_REPLAY_SIZE):
        action = env.action_space.sample()
        new_obs, reward, done, _ = env.step(action)
        transition = (obs, action, reward, done, new_obs)
        replay_buffer.append(transition)
        obs = new_obs
        if done:
            obs = env.reset()


    # Increments counter
    pbar = tqdm(total=EPISODES) # Init pbar

    
    # Main Training Loop
    obs = env.reset()

    for step in itertools.count():
        epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

        rnd_sample = random.random()
        if rnd_sample <= epsilon:
            action = env.action_space.sample()
        else:
            action = online_net.act(obs)
        
        new_obs, reward, done, _ = env.step(action)
        transition = (obs, action, reward, done, new_obs)
        replay_buffer.append(transition)
        obs = new_obs

        episode_reward += reward

        if done:
            obs = env.reset()
            reward_buffer.append(episode_reward)
            episode_reward = 0.0
            episode_count += 1
            if episode_count % 1000 == 0 and episode_count!=0:
                pbar.update(1000)
            
        # Strat Gradient Step
        transitions = random.sample(replay_buffer, BATCH_SIZE)
        loss = online_net.compute_loss(transitions, target_net)

        # Gradient Descent
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update Target Network
        if step % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(online_net.state_dict())

        # Logging 
        if step % LOG_INTERVAL == 0:
            reward_mean = np.mean(reward_buffer)
            flag0, flag1 = (flag/sum(env.done_flags) for flag in env.done_flags)
            #print()
            #print('Step', step)
            #print('Avg Reward', reward_mean)
            #print('Episodes', episode_count)

            summary_writer.add_scalar('Avg Reward', reward_mean, global_step=step)
            summary_writer.add_scalar('Episodes', episode_count, global_step=step)
            summary_writer.add_scalars('Done Flags', {'flag0': flag0, 'flag1': flag1}, global_step=step)
            

        # Save
        if step % SAVE_INTERVAL == 0 and step !=0:
            print('Saving...')
            online_net.save(SAVE_PATH+str(TYPE_DIC[env.type])+'.pack')


        if episode_count == EPISODES:
            break
    
    print('Complete')
    env.close()
    pbar.close()
