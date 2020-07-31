from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from random import random, randrange
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Deep Q Network off-policy
class DQN: 
   
    def __init__(
            self,
            input_dim, #The number of inputs for the DQN network
            action_space, #The number of actions for the DQN network
            device,
            gamma = 0.99, #The discount factor
            epsilon = 1, #Epsilon - the exploration factor
            epsilon_min = 0.01, #The minimum epsilon 
            epsilon_decay = 0.9996,#The decay epislon for each update_epsilon time
            learning_rate = 0.00025, #The learning rate for the DQN network
            tau = 0.125, #The factor for updating the DQN target network from the DQN network
            model = None, #The DQN model
            target_model = None, #The DQN target model 
            sess=None
            
    ):
      self.input_dim = input_dim
      self.action_space = action_space
      self.device = device
      self.gamma = gamma
      self.epsilon = epsilon
      self.epsilon_min = epsilon_min
      self.epsilon_decay = epsilon_decay
      self.learning_rate = learning_rate
      self.tau = tau
      self.loss = 0
      self.maker = None
            
      #Creating networks
      self.model        = Network(self.input_dim, self.action_space).to(device) #Creating the DQN model
      self.target_model = Network(self.input_dim, self.action_space).to(device) #Creating the DQN target model 
      self.optimizer = optim.SGD(self.model.parameters(), lr = self.learning_rate)
      self.target_model.load_state_dict(self.model.state_dict())
      self.target_model.eval()


    def act(self,state, gold, x, y, mine_explore, mine_bound):
      #Get the index of the maximum Q values
      guided_mine = False
      a_max = int(self.model(torch.FloatTensor(state).to(self.device)).argmax().detach().cpu().numpy())      
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
    
    
    def replay(self,samples,batch_size):
      
      if True:
        state = torch.FloatTensor(samples[0]).to(self.device)
        action = torch.LongTensor(samples[1].reshape(-1,1)).to(self.device)
        reward = torch.FloatTensor(samples[2].reshape(-1,1)).to(self.device)
        new_state = torch.FloatTensor(samples[3]).to(self.device)
        done= torch.FloatTensor(samples[4].reshape(-1,1)).to(self.device)
        
        curr_q_value = self.model(state).gather(1, action)
        with torch.no_grad():
          next_q_value = self.target_model(new_state).gather( 
            1, self.model(new_state).argmax(dim=1, keepdim=True)
          ).detach()
          mask = 1 - done
          target = (reward + self.gamma * next_q_value * mask).to(self.device)

        self.optimizer.zero_grad()
        loss = F.mse_loss(curr_q_value, target)
        self.loss = loss.item()
        loss.backward()
        self.optimizer.step()
          
    
    def target_train(self): 
      self.target_model.load_state_dict(self.model.state_dict())
    
    
    def update_epsilon(self):
      self.epsilon =  self.epsilon*self.epsilon_decay
      self.epsilon =  max(self.epsilon_min, self.epsilon)
    
    def save_model(self, path, name):
      path = path + name + ".pth"
      torch.save(self.model.state_dict(), path)    


class Network(nn.Module):
  def __init__(self, input_dim, action_space):
    super(Network, self).__init__() 
    self.input_dim = input_dim
    self.action_space = action_space
    self.input_dense = nn.Linear(self.input_dim, 300)
    self.hidden_1 = nn.Linear(300, 300)
    self.out_dense = nn.Linear(300, self.action_space)
  def forward(self, input):
    X = F.relu(self.input_dense(input))
    X = F.relu(self.hidden_1(X))
    q_value = self.out_dense(X)
    return q_value




