import os

import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from MinerEnv import MinerEnv, TreeID, TrapID, SwampID
import cv2
from prettytable import PrettyTable
from PIL import ImageFont, ImageDraw, Image
from google.colab.patches import cv2_imshow


from MinerEnv import MinerEnv

ACTIONS = {2: 'left', 3: 'right', 0: 'up', 1: 'down', 4: 'resting', 5: 'mining',6:'DIE'}
COLORS_ID = {1: (0,255,0), 2: (255, 0, 0), 3: (0, 0, 255), 4: (125,125,0)}
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
        self.view = None
        self.state = self.minerEnv.state
        self.maxstep = self.minerEnv.state.mapInfo.maxStep
        self.img_array = []


    def step(self, action):
        self.minerEnv.step(str(action))
        reward = self.get_reward()
        ob = self.get_state()
        episode_over = self.check_terminate()
        self.ob = ob
        self.action = action
        self.reward = reward
        return ob, reward, episode_over, {'score':self.minerEnv.state.score ,'action':action}

    def render(self, mode='human'):
        img = cv2.imread("/content/map1.png")
        for player in self.minerEnv.state.players:
          if player['playerId'] == 1:
            id = player['playerId']
            score = player['score']
            engergy = player['energy']
            free_count = player['freeCount']
            last_action = ACTIONS[player['lastAction']]
            # last_action = ACTIONS[self.action]
            status = player['status']

            x = player['posx']
            y = player['posy']

            if x >= 21 or y>=9:
                continue
            pos_img = (36 + x*71, 36 + y*71)
            cv2.circle(img, pos_img, 16, COLORS_ID[id], -1)
        self.img_array.append(img)
                


    def reset(self, current_eps):
        mapID = 2 #np.random.randint(1, 6)
        posID_x = 19 #np.random.randint(21)
        posID_y = 8 #np.random.randint(9)
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
        view = np.zeros([self.state.mapInfo.max_x + 1, self.state.mapInfo.max_y + 1], dtype=int)
        for i in range(self.state.mapInfo.max_x + 1):
            for j in range(self.state.mapInfo.max_y + 1):
                if self.state.mapInfo.get_obstacle(i, j) == TreeID:  # Tree
                    view[i, j] = -TreeID
                if self.state.mapInfo.get_obstacle(i, j) == TrapID:  # Trap
                    view[i, j] = -TrapID
                if self.state.mapInfo.get_obstacle(i, j) == SwampID:  # Swamp
                    view[i, j] = -SwampID
                if self.state.mapInfo.gold_amount(i, j) > 0:
                    view[i, j] = self.state.mapInfo.gold_amount(i, j)

        self.view = view
        return self.minerEnv.get_state()

    def close(self):
        self.minerEnv.end()

    def start(self):

        self.minerEnv.start()
