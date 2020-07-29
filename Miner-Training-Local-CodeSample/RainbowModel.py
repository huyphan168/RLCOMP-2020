from Src_rainbow import Network
import math
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.nn.utils import clip_grad_norm_
from typing import Deque, Dict, List, Tuple
from random import random, randrange

class RainbowAgent:
    """DQN Agent interacting with environment.
    
    Attribute:
        env (gym.Env): openAI Gym environment
        memory (PrioritizedReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        target_update (int): period for target model's hard update
        gamma (float): discount factor
        dqn (Network): model to train and select actions
        dqn_target (Network): target model to update
        optimizer (torch.optim): optimizer for training dqn
        transition (list): transition information including 
                           state, action, reward, next_state, done
        v_min (float): min value of support
        v_max (float): max value of support
        atom_size (int): the unit number of support
        support (torch.Tensor): support for categorical dqn
        use_n_step (bool): whether to use n_step memory
        n_step (int): step number to calculate n-step td error
        memory_n (ReplayBuffer): n-step replay buffer
    """

    def __init__(
        self, 
        #DQN parameters
        INPUTNUM: int,
        OUTPUTNUM: int,
        device: str,
        gamma: float = 0.99,
        batch_size : int = 32,
        epsilon = 1,
        epsilon_min = 0.1,
        epsilon_decay = 0.9996,
        # Categorical DQN parameters
        v_min: float = 0.0,
        v_max: float = 200.0,
        atom_size: int = 51,
        #N-step learning
        n_step: int = 3
    ):
        """Initialization.
        
        Args:
            env (gym.Env): openAI Gym environment
            memory_size (int): length of memory
            batch_size (int): batch size for sampling
            target_update (int): period for target model's hard update
            lr (float): learning rate
            gamma (float): discount factor
            alpha (float): determines how much prioritization is used
            beta (float): determines how much importance sampling is used
            prior_eps (float): guarantees every transition can be sampled
            v_min (float): min value of support
            v_max (float): max value of support
            atom_size (int): the unit number of support
            n_step (int): step number to calculate n-step td error
        """
        self.action_space = OUTPUTNUM
        obs_dim = INPUTNUM
        action_dim = OUTPUTNUM
        #Epsilon parameters
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.gamma = gamma
        #Maker
        self.maker = None
        #N-step learning
        self.n_step = n_step
        
        # device: cpu / gpu / TPU
        self.device = device
            
        # Categorical DQN parameters
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = torch.linspace(
            self.v_min, self.v_max, self.atom_size
        ).to(self.device)

        # networks: dqn, dqn_target
        self.dqn = Network(
            obs_dim, action_dim, self.atom_size, self.support
        ).to(self.device)
        self.dqn_target = Network(
            obs_dim, action_dim, self.atom_size, self.support
        ).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()
        
        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters(), lr = 0.01)

    def act(self, state, gold, x, y, mine_explore, mine_bound):
        """Select an action from the input state."""
        prediction = self.dqn(torch.FloatTensor(state).to(self.device)).argmax()
        a_max = int(prediction.detach().cpu().numpy())         
        guided_mine = False     
        if (random() < self.epsilon):
          self.maker = "Random"
          for cell in gold:
              if (cell["posx"], cell["posy"]) == (x, y):
                a_chosen = 5
                guided_mine = True
          if guided_mine:
              return a_chosen, guided_mine
          else:
              a_chosen = randrange(self.action_space)
              return a_chosen, guided_mine
        else:
          self.maker = "Agent"
          a_chosen = a_max      
        return a_chosen, guided_mine

    def replay(self, samples_per, samples_n_step):
        """Update the model by gradient descent."""
        # PER needs beta to calculate weights
        weights = torch.FloatTensor(
            samples_per["weights"].reshape(-1, 1)
        ).to(self.device)
        # 1-step Learning loss
        elementwise_loss = self._compute_dqn_loss(samples_per, self.gamma)
        # PER: importance sampling before average
        loss = torch.mean(elementwise_loss * weights)
        # N-step Learning loss
        # we are gonna combine 1-step loss and n-step loss so as to
        # prevent high-variance. The original rainbow employs n-step loss only.
        gamma = self.gamma ** self.n_step
        elementwise_loss_n_loss = self._compute_dqn_loss(samples_n_step, self.gamma)
        elementwise_loss += elementwise_loss_n_loss
        # PER: importance sampling before average
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        return loss.item(), loss_for_prior

        grad_output = grad_output.data
    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], gamma: float) -> torch.Tensor:
        """Return categorical dqn loss."""
        device = self.device  
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
        
        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            # Double DQN
            next_action = self.dqn(next_state).argmax(1)
            next_dist = self.dqn_target.dist(next_state)
            next_dist = next_dist[range(self.batch_size), next_action]

            t_z = reward + (1 - done) * gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.atom_size, self.batch_size
                ).long()
                .unsqueeze(1)
                .expand(self.batch_size, self.atom_size)
                .to(self.device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.dqn.dist(state)
        log_p = torch.log(dist[range(self.batch_size), action])
        elementwise_loss = -(proj_dist * log_p).sum(1)

        return elementwise_loss
    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())  
    def update_epsilon(self):
        self.epsilon =  self.epsilon*self.epsilon_decay
        self.epsilon =  max(self.epsilon_min, self.epsilon)
    def save_model(self, path, name):
      path = path + name + ".pth"
      torch.save(self.dqn.state_dict(), path)




