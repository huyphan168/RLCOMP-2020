import os

import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


from MinerEnv import MinerEnv

ACTIONS = {2: 'left', 3: 'right', 0: 'up', 1: 'down', 4: 'stand', 5: 'mining',6:'DIE'}
class MinerGymEnv(gym.Env):
    def __init__(self, HOST, PORT, debug=False):
        self.minerEnv = MinerEnv(HOST,
                                 PORT)
        self.minerEnv.start()
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Discrete(198)
        self.action = None
        self.reward = None
        self.ob = None
        self.state = self.minerEnv.state
        self.maxstep = self.minerEnv.state.mapInfo.maxStep


    def step(self, action):
        self.minerEnv.step(str(action))
        reward = self.get_reward()
        ob = self.get_state()
        episode_over = self.check_terminate()
        self.ob = ob
        self.action = action
        self.reward = reward
        return ob, reward, episode_over, {'score':self.minerEnv.state.score ,'action':action}

    def render(self, mode='human', close=False):
        pass

    def reset(self, current_eps):

        mapID = 1 #np.random.randint(1, 6)
        posID_x = np.random.randint(21)
        posID_y = np.random.randint(9)
        if current_eps > 0.1:
          request = ("map" + str(mapID) + "," + str(posID_x) + "," + str(posID_y) + ",50,150")
        else:
          request = ("map" + str(mapID) + "," + str(posID_x) + "," + str(posID_y) + ",50,100")
        self.minerEnv.send_map_info(request)
        self.minerEnv.reset()
        state= self.get_state()
        return state
    def check_terminate(self):
      return self.minerEnv.check_terminate()


    def get_reward(self):
      return self.minerEnv.get_reward()
    
    def get_state(self):
      return self.minerEnv.get_state()

    def close(self):
        self.minerEnv.end()

    def start(self):

        self.minerEnv.start()
