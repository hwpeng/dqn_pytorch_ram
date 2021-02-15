import copy
from collections import namedtuple
from itertools import count
import math
import random
import numpy as np 
import time

import gym
from ale_py import ALEInterface

from wrappers import *
from memory import ReplayMemory
from models import *


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

Transition = namedtuple('Transion', 
                        ('state', 'action', 'next_state', 'reward'))

def rm_redundant(ram):
    rd = [1,2,3,4,6,7,8,9,10,11,12,13,23,25,27,29,31,36,37,38,39,40,41,42,43,44,45,48,50,52,54,58,62,76,80,85,86,87,89,100,102,104,106,108,110,112,114,115,116,117,118,120,121,123,124,125,126,127]
    ram = np.delete(ram, rd)
    return ram

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END)* \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            #  return policy_net(state.to('cuda')).max(1)[1].view(1,1)
            return policy_net(state).max(1)[1].view(1,1)
    else:
        return torch.tensor([[random.randrange(4)]], device=device, dtype=torch.long)

    
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    """
    zip(*transitions) unzips the transitions into
    Transition(*) creates new named tuple
    batch.state - tuple of all the states (each state is a tensor)
    batch.next_state - tuple of all the next states (each state is a tensor)
    batch.reward - tuple of all the rewards (each reward is a float)
    batch.action - tuple of all the actions (each action is an int)    
    """
    batch = Transition(*zip(*transitions))
    
    actions = tuple((map(lambda a: torch.tensor([[a]], device='cpu'), batch.action))) 
    rewards = tuple((map(lambda r: torch.tensor([r], device='cpu'), batch.reward))) 

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device, dtype=torch.bool)
    
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None]).to('cpu')
    

    state_batch = torch.cat(batch.state).to('cpu')
    action_batch = torch.cat(actions)
    reward_batch = torch.cat(rewards)
    
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def get_state(obs, RAM_Train):
    if RAM_Train == False:
        state = np.array(obs)
        state = state.transpose((2, 0, 1))
        state = torch.from_numpy(state)
    else:
        state = np.array(obs)
        state = torch.from_numpy(state)
        state = state/255
        state = state.type(torch.float)
    return state.unsqueeze(0)

def train(env, n_episodes, RAM_Train=False, render=False):
    for episode in range(n_episodes):
        if RAM_Train == False:
            obs = env.reset()
        else:
            env.reset_game()
            obs = env.getRAM()
            obs = rm_redundant(obs)
        
        state = get_state(obs, RAM_Train)

        total_reward = 0.0
        for t in count():
            action = select_action(state)

            if render:
                env.render()

            if RAM_Train == False:
                obs, reward, done, info = env.step(action)
            else:
                reward = env.act(action)
                obs = env.getRAM()
                obs = rm_redundant(obs)
                done = env.game_over()

            total_reward += reward

            if not done:
                # next_state = get_state(obs)
                next_state = get_state(obs, RAM_Train)
            else:
                next_state = None

            reward = torch.tensor([reward], device=device)

            memory.push(state, action.to('cpu'), next_state, reward.to('cpu'))
            state = next_state

            if steps_done > INITIAL_MEMORY:
                optimize_model()

                if steps_done % TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            if done:
                break
        if episode % 2 == 0:
                print('Total steps: {} \t Episode: {}/{} \t Total reward: {}'.format(steps_done, episode, n_episodes, total_reward))
                memory.save()

    if RAM_Train == False:
        env.close()
    return

def test(env, n_episodes, policy, render=True):
    env = gym.wrappers.Monitor(env, './videos/' + 'dqn_pong_video')
    for episode in range(n_episodes):
        obs = env.reset()
        state = get_state(obs)
        total_reward = 0.0
        for t in count():
            action = policy(state.to('cpu')).max(1)[1].view(1,1)

            if render:
                env.render()
                time.sleep(0.02)

            obs, reward, done, info = env.step(action)

            total_reward += reward

            if not done:
                next_state = get_state(obs)
            else:
                next_state = None

            state = next_state

            if done:
                print("Finished Episode {} with reward {}".format(episode, total_reward))
                break

    env.close()
    return

if __name__ == '__main__':
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # hyperparameters
    BATCH_SIZE = 32
    GAMMA = 0.99
    EPS_START = 1
    EPS_END = 0.02
    EPS_DECAY = 1000000
    TARGET_UPDATE = 1000
    RENDER = False
    lr = 1e-4
    INITIAL_MEMORY = 10000
    MEMORY_SIZE = 10 * INITIAL_MEMORY

    RAM_Train = True

    # create networks
    if RAM_Train == False:
        policy_net = DQN(n_actions=4).to(device)
        target_net = DQN(n_actions=4).to(device)
    else:
        policy_net = DQN_RAM(in_features=70).to(device)
        target_net = DQN_RAM(in_features=70).to(device)

    target_net.load_state_dict(policy_net.state_dict())

    # setup optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    steps_done = 0

    # create environment
    if RAM_Train == False:
        env = gym.make("PongNoFrameskip-v4")
        env = make_env(env)
    else:
        env = ALEInterface()
        env.loadROM('boxing.bin')

    # initialize replay memory
    memory = ReplayMemory(MEMORY_SIZE)
    
    # train model
    train(env, 400, RAM_Train)

    torch.save(policy_net, "dqn_pong_model")
    policy_net = torch.load("dqn_pong_model")
    if RAM_Train == False:
        test(env, 1, policy_net, render=False)

