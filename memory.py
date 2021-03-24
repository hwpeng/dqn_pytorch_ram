from collections import namedtuple
import random
import numpy as np
import torch

Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward', 'nonterminal'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        
    def append(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        sampled_transitions = random.sample(self.memory, batch_size)
        states = torch.stack(list(t.state for t in sampled_transitions), 0)
        actions = list(t.action for t in sampled_transitions)
        rewards = torch.stack(list(torch.tensor(t.reward, dtype=torch.float32) for t in sampled_transitions), 0)
        next_states = torch.stack(list(t.next_state for t in sampled_transitions), 0)
        nonterminals = torch.stack(list(torch.tensor(t.nonterminal, dtype=torch.float32) for t in sampled_transitions), 0)

        return states, actions, rewards, next_states, nonterminals 

    def save(self):
        with open('replay_memory.npy', 'wb') as f:
            np.save(f, np.asarray(self.memory))
    
    def __len__(self):
        return len(self.memory)


class PrioritizedReplay(object):
    def __init__(self, capacity):
        pass
