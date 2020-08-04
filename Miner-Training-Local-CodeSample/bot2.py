from MINER_STATE import State
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F


class PlayerInfo:
    def __init__(self, id):
        self.playerId = id
        self.score = 0
        self.energy = 0
        self.posx = 0
        self.posy = 0
        self.lastAction = -1
        self.status = 0
        self.freeCount = 0


class Network(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """Initialization."""
        super(Network, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, 300), 
            nn.ReLU(),
            nn.Linear(300, 300), 
            nn.ReLU(), 
            nn.Linear(300, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        return self.layers(x)

class Bot_ver2:
    ACTION_GO_LEFT = 0
    ACTION_GO_RIGHT = 1
    ACTION_GO_UP = 2
    ACTION_GO_DOWN = 3
    ACTION_FREE = 4
    ACTION_CRAFT = 5

    def __init__(self, id):
        self.state = State()
        self.info = PlayerInfo(id)
        self.dqn = Network(198, 6)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dqn.load_state_dict(torch.load("/content/AGENT1_40000.pth"))
        print("Model loaded for bot 2")
        self.dqn.to(self.device)
    def next_action(self):
        STATE_NET = torch.FloatTensor(self.get_state()).to(self.device)
        action = self.dqn(STATE_NET).argmax()
        action = action.detach().cpu().numpy()  
        action = int(action)
        return action

    def new_game(self, data):
        try:
            self.state.init_state(data)
        except Exception as e:
            import traceback
            traceback.print_exc()

    def get_state(self):
        # Building the map
        view = np.zeros([self.state.mapInfo.max_x + 1, self.state.mapInfo.max_y + 1], dtype=int)
        for i in range(self.state.mapInfo.max_x + 1):
            for j in range(self.state.mapInfo.max_y + 1):
                if self.state.mapInfo.get_obstacle(i, j) == 1:  # Tree
                    view[i, j] = -1
                if self.state.mapInfo.get_obstacle(i, j) == 2:  # Trap
                    view[i, j] = -2
                if self.state.mapInfo.get_obstacle(i, j) == 3: # Swamp
                    view[i, j] = -3
                if self.state.mapInfo.gold_amount(i, j) > 0:
                    view[i, j] = 5

        BOTState = view.flatten().tolist() #Flattening the map matrix to a vector
        BOTState.append(self.state.x)
        BOTState.append(self.state.y)
        # Add position and energy of agent to the DQNState
        BOTState.append(self.state.energy)
        for player in self.state.players:
            if player["playerId"] != self.state.id:
                BOTState.append(player["posx"])
                BOTState.append(player["posy"])
        #Convert the DQNState from list to array for training
        BOTState = np.array(BOTState)

        return BOTState

    def new_state(self, data):
        # action = self.next_action();
        # self.socket.send(action)
        try:
            self.state.update_state(data)
        except Exception as e:
            import traceback
            traceback.print_exc()
