"""
Adapted from Kaixhin's Ranbow implementation
https://github.com/Kaixhin/Rainbow/blob/master/env.py
"""
from collections import deque
import random
import numpy as np
import copy
import cv2
import atari_py
import ale_py
import torch

class Env():
  def __init__(self, args):
    self.device = args.device
    #  self.ale = atari_py.ALEInterface()
    self.ale = ale_py.ALEInterface()
    self.ale.setInt('random_seed', args.seed)
    self.ale.setInt('max_num_frames_per_episode', args.max_episode_length)
    self.ale.setFloat('repeat_action_probability', 0)  # Disable sticky actions
    self.ale.setInt('frame_skip', 0)
    self.ale.setBool('color_averaging', False)
    self.ale.loadROM(atari_py.get_game_path(args.game))  # ROM loading must be done after setting options
    actions = self.ale.getMinimalActionSet()
    self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
    self.lives = 0  # Life counter (used in DeepMind training)
    self.life_termination = False  # Used to check if resetting only from loss of life
    self.window = args.history_length  # Number of frames to concatenate
    self.state_buffer = deque([], maxlen=args.history_length)
    self.training = True  # Consistent with model training mode

    if (args.state_data == 'ram'):
      self.state_size = 128
      self.state_data = 'ram'
    elif (args.state_data == 'ram_tia'):
      self.state_size = 128+0x2A
      self.state_data = 'ram_tia'

  def _get_state(self):
    if (self.state_data == 'ram'):
      state = self.ale.getRAM()
    else:
      state = np.concatenate((self.ale.getRAM(), self.ale.getTIA()), axis=0)
    return torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)

  def _reset_buffer(self):
    for _ in range(self.window):
      self.state_buffer.append(torch.zeros(self.state_size, device=self.device))

  def reset(self):
    if self.life_termination:
      self.life_termination = False  # Reset flag
      self.ale.act(0)  # Use a no-op after loss of life
    else:
      # Reset internals
      self._reset_buffer()
      self.ale.reset_game()
      # Perform up to 30 random no-ops before starting
      for _ in range(random.randrange(30)):
        self.ale.act(0)  # Assumes raw action 0 is always no-op
        if self.ale.game_over():
          self.ale.reset_game()
    # Process and return "initial" state
    observation = self._get_state()
    self.state_buffer.append(observation)
    self.lives = self.ale.lives()
    #  return torch.stack(list(self.state_buffer), 0)
    return torch.cat(list(self.state_buffer), 0)

  def step(self, action):
    # Repeat action 4 times, max pool over last 2 frames
    frame_buffer = torch.zeros(2, self.state_size, device=self.device)
    reward, done = 0, False
    for t in range(4):
      reward += self.ale.act(self.actions.get(action))
      if t == 2:
        frame_buffer[0] = self._get_state()
      elif t == 3:
        frame_buffer[1] = self._get_state()
      done = self.ale.game_over()
      if done:
        break
    observation = frame_buffer.max(0)[0]
    self.state_buffer.append(observation)
    # Detect loss of life as terminal in training mode
    if self.training:
      lives = self.ale.lives()
      if lives < self.lives and lives > 0:  # Lives > 0 for Q*bert
        self.life_termination = not done  # Only set flag when not truly done
        done = True
      self.lives = lives
    # Return state, reward, done
    #  return torch.stack(list(self.state_buffer), 0), reward, done
    return torch.cat(list(self.state_buffer), 0), reward, done

  # Uses loss of life as terminal signal
  def train(self):
    self.training = True

  # Uses standard terminal signal
  def eval(self):
    self.training = False

  def action_space(self):
    return len(self.actions)

  def render(self):
    cv2.imshow('screen', self.ale.getScreenRGB()[:, :, ::-1])
    cv2.waitKey(1)

  def close(self):
    cv2.destroyAllWindows()

  def rm_redundant(ram):
    rd = [1,2,3,4,6,7,8,9,10,11,12,13,23,25,27,29,31,36,37,38,39,40,41,42,43,44,45,48,50,52,54,58,62,76,80,85,86,87,89,100,102,104,106,108,110,112,114,115,116,117,118,120,121,123,124,125,126,127]
    ram = np.delete(ram, rd)
    return ram







