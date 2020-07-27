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
        view = np.zeros([self.state.mapInfo.max_x + 1, self.state.mapInfo.max_y + 1], dtype=int)
        for i in range(self.state.mapInfo.max_x + 1):
            for j in range(self.state.mapInfo.max_y + 1):
                if self.state.mapInfo.get_obstacle(i, j) == TreeID:  # Tree
                    view[i, j] = -5
                if self.state.mapInfo.get_obstacle(i, j) == TrapID:  # Trap
                    view[i, j] = -10
                if self.state.mapInfo.get_obstacle(i, j) == SwampID: # Swamp
                    view[i, j] = -10
                if self.state.mapInfo.gold_amount(i, j) > 0:
                    view[i, j] = 40

        DQNState = view.flatten().tolist() #Flattening the map matrix to a vector
        
        # Add position and energy of agent to the DQNState
        DQNState.append(self.state.x)
        DQNState.append(self.state.y)
        DQNState.append(self.state.energy)
        #Add position of bots 
        for player in self.state.players:
            if player["playerId"] != self.state.id:
                DQNState.append(player["posx"])
                DQNState.append(player["posy"])
        #Convert the DQNState from list to array for training
        DQNState = np.array(DQNState)

        return DQNState

    def get_reward(self):
        # Calculate reward
        reward = 0
        score_action = self.state.score - self.score_pre
        self.score_pre = int(self.state.score)
        if score_action > 0 and self.state.lastAction == 5:
            #If the DQN agent crafts golds, then it should obtain a positive reward (equal score_action)
          reward += score_action
        if score_action <= 0 and self.state.lastAction == 5:
          reward -= 5
        if self.state.mapInfo.is_row_has_gold(self.state.y) and self.state.lastAction in [2, 3] and self.state.mapInfo.get_obstacle(self.state.x, self.state.y) not in [TreeID,TrapID,SwampID] :
          reward += 5
        if self.state.mapInfo.is_column_has_gold(self.state.x) and self.state.lastAction in [0, 1] and self.state.mapInfo.get_obstacle(self.state.x, self.state.y) not in [TreeID,TrapID,SwampID]:
          reward += 5
        if self.state.lastAction == 4 and self.state.energy > 35:
          reward -= 30
        if self.state.lastAction == 4:
          reward += 10
        A_dis = []
        for cell in self.state.mapInfo.golds:
          dis = np.sqrt((cell["posx"]-self.state.x)**2 + (cell["posy"]-self.state.y)**2)
          A_dis.append(dis)
        min_dis = min(A_dis)
        if min_dis < 4 and self.state.lastAction in [0,1,2,3]:
          reward += 2
        if self.state.mapInfo.get_obstacle(self.state.x, self.state.y) not in [TreeID,TrapID,SwampID]:
            reward += 3
        #If the DQN agent crashs into obstacels (Tree, Trap, Swamp), then it should be punished by a negative reward
        if self.state.mapInfo.get_obstacle(self.state.x, self.state.y) == TreeID:  # Tree
            reward -= 10
        if self.state.mapInfo.get_obstacle(self.state.x, self.state.y) == TrapID:  # Trap
            reward -= 10
        if self.state.mapInfo.get_obstacle(self.state.x, self.state.y) == SwampID:  # Swamp
            reward -= 20
        
        # If out of the map, then the DQN agent should be punished by a larger nagative reward.
        if self.state.status == State.STATUS_ELIMINATED_WENT_OUT_MAP:
            reward += -40
        if self.state.status == State.STATUS_ELIMINATED_OUT_OF_ENERGY:
            reward += -30
            
        # print ("reward",reward)
        return round(reward/10,2)

    def check_terminate(self):
        #Checking the status of the game
        #it indicates the game ends or is playing
        return self.state.status != State.STATUS_PLAYING
