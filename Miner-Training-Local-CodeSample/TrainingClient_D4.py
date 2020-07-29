import sys
from  Duelling_DDQN_PER_Model import D3PAgent
from MinerEnv import MinerEnv 
from Memory_PER import PrioritizedReplayBuffer

import pandas as pd
import datetime 
import numpy as np
import random
import torch


HOST = "localhost"
PORT = 1111
if len(sys.argv) == 3:
    HOST = str(sys.argv[1])
    PORT = int(sys.argv[2])

now = datetime.datetime.now() 
header = ["Ep", "Step", "Reward","Score", "Total_reward", "Action", "Energy"] 
filename = "Data/data_" + now.strftime("%Y%m%d-%H%M") + ".csv" 
with open(filename, 'w') as f:
    pd.DataFrame(columns=header).to_csv(f, encoding='utf-8', index=False, header=True)
#Traing Parameters
N_EPISODE = 7000 
MAX_STEP = 100  
BATCH_SIZE = 32    
MEMORY_SIZE = 100000
INITIAL_REPLAY_SIZE = 1000 
INPUTNUM = 198 
ACTIONNUM = 6  
MAP_MAX_X = 21 
MAP_MAX_Y = 9  
#Updating target_network
update_cnt = 0 
update_target = 10000
#Prioritized Memory Parameters
alpha = 0.2
beta = 0.6
prior_eps = 1e-6
#Initialize Agent and memory
Agent = D3PAgent(INPUTNUM, ACTIONNUM)
memory = PrioritizedReplayBuffer(INPUTNUM, MEMORY_SIZE, BATCH_SIZE, alpha=alpha)

# Initialize environment
minerEnv = MinerEnv(HOST, PORT) 
minerEnv.start()  

train = False 
for episode_i in range(0, N_EPISODE):
    try:
        #Begin episode
        loss_lst = [0]
        #Initializing map and position of agent
        mapID = random.choice([1,2]) 
        posID_x = np.random.randint(MAP_MAX_X) 
        posID_y = np.random.randint(MAP_MAX_Y) 
        #Sending request to SOCKET
        request = ("map" + str(mapID) + "," + str(posID_x) + "," + str(posID_y) + ",50,100") 
        minerEnv.send_map_info(request)
        #Randomly initialize the maximum mining exploration
        mine_bound = random.choice([i for i in range(4)])
        mine_explore = 1
        #Initialize the environment
        minerEnv.reset()
        #Get first state 
        s = minerEnv.get_state()
        total_reward = 0 
        terminate = False 
        maxStep = minerEnv.state.mapInfo.maxStep 
        for step in range(0, maxStep):
            #Initialize transition cache for memory push
            transition = list()
            #Guided Exploration
            gold = minerEnv.state.mapInfo.golds
            posx = minerEnv.state.x
            posy = minerEnv.state.y
            action, mine_incre = Agent.act(s, gold, posx, posy, mine_explore, mine_bound)
            #Checking who is making decision for storing data about agent'decisions
            maker = Agent.maker
            #Storing temperory state and action
            transition = [s, action]
            #Increase the number of mining times that exploration mining bound allow
            if mine_incre:
                mine_explore += 1
            #Sending action to environment
            minerEnv.step(str(action))
            #Getting next state and reward 
            s_next = minerEnv.get_state()  
            reward = minerEnv.get_reward() 
            terminate = minerEnv.check_terminate()
            transition += [reward, s_next, terminate]  
            memory.store(*transition)
            #Updating PER parameters
            fraction = min(episode_i / N_EPISODE, 1.0)
            beta = beta + fraction*(1-beta)
            #Step data
            score = minerEnv.state.score
            energy = minerEnv.state.energy
            status = minerEnv.state.status
            if (len(memory) > INITIAL_REPLAY_SIZE):
                #Signal that begin training process
                train = True
                #Sampling memory batch
                samples_per = memory.sample_batch(beta)
                indices_per = samples_per["indices"]
                #Replaying memory for training Agent
                loss_for_prior = Agent.replay(samples_per)
                #Storing loss
                loss_lst.append(Agent.loss)
                #Updating priorities for PER
                new_priorities = loss_for_prior + prior_eps
                memory.update_priorities(indices_per, new_priorities)
                #Updating target_network
                update_cnt += 1
                if (update_cnt % update_target) == 0:
                  Agent.target_train()
            #Accumalated reward     
            total_reward = total_reward + reward 
            #Assigning new state 
            s = s_next 
            #Storing data for agent analysis
            if maker == "Agent":
              save_data = np.hstack(
                [episode_i + 1, step + 1, reward, score, total_reward, action, energy ]).reshape(1, 7)
              with open(filename, 'a') as f:
                  pd.DataFrame(save_data).to_csv(f, encoding='utf-8', index=False, header=False)
            #If episode is ended: break the steps loop
            if terminate == True:
                break
        #Print the training information after the episode
        print('Episode {} ends. Number of steps is: {}. Accumulated Reward = {}. Epsilon = {} .Score: {} .Energy: {} .Status: {}. Loss {}'.format(
            episode_i + 1, step + 1, total_reward, round(Agent.epsilon,2), score, energy, status, round(sum(loss_lst)/(step+1), 3)))       
        #Decreasing the epsilon if the replay starts
        if train == True:
            Agent.update_epsilon()
    except Exception as e:
      import traceback

      traceback.print_exc()
      print("Finished.")
      break
print("Finished")
Agent.save_model("/content/Trained_Agent/", "Rainbow_ver1")
