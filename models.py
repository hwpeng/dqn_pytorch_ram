import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

class DQN(nn.Module):
    def __init__(self, args, action_space):
        super(DQN, self).__init__()
        self.action_space = action_space

        self.fc1 = nn.Linear(args.history_length*128, 512)
        #  self.fc2 = nn.Linear(256, 128)
        #  self.fc3 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, self.action_space)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        #  x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)
