# -*- coding: utf-8 -*-
from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import keras
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Activation
from keras import optimizers
from keras import backend as K
import tensorflow as tf
from random import random, randrange


# Deep Q Network off-policy
class D3PAgent: 
   
    def __init__(
            self,
            input_dim, #The number of inputs for the DQN network
            action_space, #The number of actions for the DQN network
            batch_size = 32,
            gamma = 0.99, #The discount factor
            epsilon = 1, #Epsilon - the exploration factor
            epsilon_min = 0.01, #The minimum epsilon 
            epsilon_decay = 0.9997,#The decay epislon for each update_epsilon time
            learning_rate = 0.00025, #The learning rate for the DQN network
            tau = 0.125, #The factor for updating the DQN target network from the DQN network
            model = None, #The DQN model
            target_model = None, #The DQN target model 
            sess=None
            
    ):
      self.input_dim = input_dim
      self.action_space = action_space
      self.gamma = gamma
      self.epsilon = epsilon
      self.epsilon_min = epsilon_min
      self.epsilon_decay = epsilon_decay
      self.learning_rate = learning_rate
      self.tau = tau
      self.maker = None
      self.loss = 0
      self.batch_size = batch_size
            
      #Creating networks
      self.model        = Model(self.input_dim, 300, self.action_space, True) #Creating the DQN model
      self.target_model = Model(self.input_dim, 300, self.action_space, True) #Creating the DQN target model
      self.model.compile(optimizer=keras.optimizers.Adam(), loss="mean_squared_error")
      self.target_model.set_weights(self.model.get_weights())

    def act(self,state, gold, x, y, mine_explore, mine_bound):
      #Get the index of the maximum Q values
      guided_mine = False
      a_max = np.argmax(self.model.predict(state.reshape(1,len(state))))      
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
      return int(a_chosen), guided_mine
    
    
    def replay(self,samples_per):

      state = np.zeros((self.batch_size, self.input_dim))
      next_state = np.zeros((self.batch_size, self.input_dim))
      action, reward, done = [], [], []
      for i in range(self.batch_size):
          state[i] = samples_per["obs"][i]
          action.append(samples_per["acts"][i])
          reward.append(samples_per["rews"][i])
          next_state[i] = samples_per["next_obs"][i]
          done.append(samples_per["done"][i])
      action = [int(ac) for ac in action]
      target = self.model.predict(state)
      target_old = np.array(target)
      target_next = self.model.predict(next_state)
      target_val = self.target_model.predict(next_state)
      for i in range(len(samples_per)):
          if done[i]:
            target[i][action[i]] = reward[i]
          else:
            a = np.argmax(target_next[i])
            target[i][action[i]] = reward[i] + self.gamma * (target_val[i][a])
      indices = np.arange(self.batch_size, dtype=np.int32)
      loss = target_old[indices, np.array(action)]-target[indices, np.array(action)]
      error = np.abs(target_old[indices, np.array(action)]-target[indices, np.array(action)])
      with tf.device("/GPU:0"):  
          loss = self.model.train_on_batch(state, target)
          self.loss = loss
      return error
              
    def target_train(self): 
      weights = self.model.get_weights()
      target_weights = self.target_model.get_weights()
      for i in range(0, len(target_weights)):
        target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
      
      self.target_model.set_weights(target_weights) 
    
    
    def update_epsilon(self):
      self.epsilon =  self.epsilon*self.epsilon_decay
      self.epsilon =  max(self.epsilon_min, self.epsilon)
    
    
    def save_model(self,path, model_name):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(path + model_name + ".json", "w") as json_file:
            json_file.write(model_json)
            # serialize weights to HDF5
            self.model.save_weights(path + model_name + ".h5")
            print("Saved model to disk")
    def huber_loss(self, loss):
        return 0.5 * loss ** 2 if abs(loss) < 1.0 else abs(loss) - 0.5

init = keras.initializers.he_normal()
def Model(obs_dim, hidden_size, num_actions, dueling):
    X_input = keras.layers.Input(shape = (obs_dim,))
    feature = Dense(hidden_size, input_dim = obs_dim, activation = "relu", kernel_initializer = init)(X_input)
    feature = Dense(hidden_size, activation = "relu", kernel_initializer = init)(feature)
    adv = Dense(hidden_size, activation = "relu", kernel_initializer = init)(feature)
    adv = Dense(num_actions, activation = "relu", kernel_initializer = init)(adv)
    if dueling:
        value = Dense(hidden_size, input_dim = obs_dim, activation = "relu", kernel_initializer = init)(feature)
        value = Dense(1, activation = "relu", kernel_initializer = init)(value)
        adv_norm = keras.layers.Lambda(lambda x: x - tf.reduce_mean(x))(adv)
        combine = keras.layers.Add()([value, adv])
    if dueling:
      model = keras.Model(inputs = X_input, outputs = combine)
    else:
      model = keras.Model(inputs = X_input, outputs = adv)
    return model








