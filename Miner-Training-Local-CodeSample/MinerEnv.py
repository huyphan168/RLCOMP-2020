import sys
import numpy as np
from GAME_SOCKET_DUMMY import GameSocket #in testing version, please use GameSocket instead of GAME_SOCKET_DUMMY
from MINER_STATE import State


TreeID = 1
TrapID = 2
SwampID = 3
class MinerEnv:
    def __init__(self, host, port):
        self.socket = GameSocket(host, port)
        self.state = State()
        self.energy_pre = self.state.energy
        self.score_pre = self.state.score#Storing the last score for designing the reward function

    def start(self): #connect to server
        self.socket.connect()

    def end(self): #disconnect server
        self.socket.close()

    def send_map_info(self, request):#tell server which map to run
        self.socket.send(request)

    def reset(self): #start new game
        try:
            message = self.socket.receive() #receive game info from server
            self.state.init_state(message) #init state
        except Exception as e:
            import traceback
            traceback.print_exc()

    def step(self, action): #step process
        self.socket.send(action) #send action to server
        try:
            message = self.socket.receive() #receive new state from server
            self.state.update_state(message) #update to local state
        except Exception as e:
            import traceback
            traceback.print_exc()

    # Functions are customized by client
    def get_state(self):
        # Building the map
        channel_1 = np.zeros([self.state.mapInfo.max_x + 1, self.state.mapInfo.max_y + 1], dtype=int)
        for i in range(self.state.mapInfo.max_x + 1):
            for j in range(self.state.mapInfo.max_y + 1):
                obs_id, val = self.state.mapInfo.get_obstacle(i, j)
                if obs_id == TreeID:  # Tree
                    channel_1[i, j] = 0.3
                if obs_id == TrapID:  # Trap
                    channel_1[i, j] = 0.6
        
        channel_2 = np.zeros([self.state.mapInfo.max_x + 1, self.state.mapInfo.max_y + 1], dtype=int)
        for i in range(self.state.mapInfo.max_x + 1):
            for j in range(self.state.mapInfo.max_y + 1):
                obs_id, val = self.state.mapInfo.get_obstacle(i, j)
                if obs_id == SwampID:  # Tree
                    if abs(val) == -5:
                      channel_2[i, j] = 0.1
                    if abs(val) == -20:
                      channel_2[i, j] = 0.4
                    if abs(val) > 20:
                      channel_2[i, j] = 0.8
        channel_3 = np.zeros([self.state.mapInfo.max_x + 1, self.state.mapInfo.max_y + 1], dtype=int)
        for i in range(self.state.mapInfo.max_x + 1):
            for j in range(self.state.mapInfo.max_y + 1): 
                if self.state.mapInfo.gold_amount(i, j) > 0:
                  channel_3[i, j] = float(self.state.mapInfo.gold_amount(i, j)/1600)
        
        channel_4 = np.zeros([self.state.mapInfo.max_x + 1, self.state.mapInfo.max_y + 1], dtype=int)
        for i in range(self.state.mapInfo.max_x + 1):
            for j in range(self.state.mapInfo.max_y + 1):
                if self.state.x in range(21) and self.state.y in range(9):
                  channel_4[self.state.x, self.state.y] = 1
        X = []
        Y = []
        for player in self.state.players:
            if player["playerId"] != self.state.id:
                X.append(player["posx"])
                Y.append(player["posy"])
        
        channel_5 = np.zeros([self.state.mapInfo.max_x + 1, self.state.mapInfo.max_y + 1], dtype=int)
        for i in range(self.state.mapInfo.max_x + 1):
            for j in range(self.state.mapInfo.max_y + 1):
                if X[0] in range(21) and Y[0] in range(9):
                  channel_5[X[0], Y[0]] = 1 

        channel_6 = np.zeros([self.state.mapInfo.max_x + 1, self.state.mapInfo.max_y + 1], dtype=int)
        for i in range(self.state.mapInfo.max_x + 1):
            for j in range(self.state.mapInfo.max_y + 1):
                if X[1] in range(21) and Y[1] in range(9):
                  channel_6[X[1], Y[1]] = 1
      
        channel_7 = np.zeros([self.state.mapInfo.max_x + 1, self.state.mapInfo.max_y + 1], dtype=int)
        for i in range(self.state.mapInfo.max_x + 1):
            for j in range(self.state.mapInfo.max_y + 1):
                if X[2] in range(21) and Y[2] in range(9):
                  channel_7[X[2], Y[2]] = 1
        
        channel_8 = np.zeros([self.state.mapInfo.max_x + 1, self.state.mapInfo.max_y + 1], dtype=int)
        for i in range(self.state.mapInfo.max_x + 1):
            for j in range(self.state.mapInfo.max_y + 1):
                channel_8[i, j] = float(self.state.energy/50)
        DQNState = np.dstack([channel_1, channel_2, channel_3, channel_4, channel_5,
                                          channel_6, channel_7, channel_8])
        DQNState = np.rollaxis(DQNState, 2, 0)
        return DQNState

    def get_reward(self):
        # Calculate reward
        reward = 0
        score_action = self.state.score - self.score_pre
        self.score_pre = int(self.state.score)
        if score_action > 0 and self.state.lastAction == 5:
          reward += 5
        if score_action <= 0 and self.state.lastAction == 5:
          reward -= 2
        obs_id, value = self.state.mapInfo.get_obstacle(self.state.x, self.state.y)
        if obs_id not in [1,2,3] and self.state.lastAction != 4:
            reward += 0.5
        if obs_id == TreeID:  # Tree
            reward -= 2
        if obs_id == TrapID:  # Trap
            reward -= 2
        if obs_id == SwampID:  # Swamp
          if abs(value) <= -5:
            reward -= 0.5
          if 15 <= abs(value) <= 40:
            reward -= 4
          if abs(value) > 40:
            reward -= 6
        
        # if self.state.mapInfo.is_row_has_gold(self.state.x):
        #   if self.state.lastAction in [2,3]:
        #     reward += 1
        #   else:
        #     reward += 0.5
        # if self.state.mapInfo.is_column_has_gold(self.state.x):
        #   if self.state.lastAction in [0,1]:
        #     reward += 1
        #   else:
        #     reward += 0.5
        if self.state.lastAction == 4 and self.state.energy > 40:
          reward -= 3
        if self.state.lastAction == 4:
          reward += 1.25
        if self.state.status == State.STATUS_ELIMINATED_WENT_OUT_MAP:
            reward += -10
        if self.state.status == State.STATUS_ELIMINATED_OUT_OF_ENERGY:
            reward += -5    
        return np.sign(reward)*np.log(1 + abs(reward))

    def check_terminate(self):
        #Checking the status of the game
        #it indicates the game ends or is playing
        return self.state.status != State.STATUS_PLAYING

