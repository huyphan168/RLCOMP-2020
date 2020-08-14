from MINER_STATE import State
import numpy as np

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
class Node:
    def __init__(self,value,point):
        self.value = value
        self.point = point
        self.parent = None
        self.H = 0
        self.G = 0
    def move_cost(self,other):
        return 0 if self.value == '.' else 1
def children(point,grid):
  x,y = point.point
  links = [grid[d[0]][d[1]] for d in [(x-1, y),(x,y - 1),(x,y + 1),(x+1,y)] if d[0] < 21 and d[1] < 9 ]
  return [link for link in links if link.value != '%']

def manhattan(point,point2):
    return abs(point.point[0] - point2.point[0]) + abs(point.point[1]-point2.point[1])
def aStar(start, goal, grid):
    #The open and closed sets
    openset = set()
    closedset = set()
    #Current point is the starting point
    current = start
    #Add the starting point to the open set
    openset.add(current)
    #While the open set is not empty
    print(openset)
    while openset:
        #Find the item in the open set with the lowest G + H score
        current = min(openset, key=lambda o:o.G + o.H)
        #If it is the item we want, retrace the path and return it
        if current == goal:
            path = []
            while current.parent:
                path.append(current)
                current = current.parent
            path.append(current)
            return path[::-1]
        #Remove the item from the open set
        openset.remove(current)
        #Add it to the closed set
        closedset.add(current)
        #Loop through the node's children/siblings
        for node in children(current,grid):
            #If it is already in the closed set, skip it
            if node in closedset:
                continue
            #Otherwise if it is already in the open set
            if node in openset:
                #Check if we beat the G score 
                new_g = current.G + current.move_cost(node)
                if node.G > new_g:
                    #If so, update the node to have a new parent
                    node.G = new_g
                    node.parent = current
            else:
                #If it isn't in the open set, calculate the G and H score for the node
                node.G = current.G + current.move_cost(node)
                node.H = manhattan(node, goal)
                #Set the parent to our current item
                node.parent = current
                #Add it to the set
                openset.add(node)
    #Throw an exception if there is no path
    raise ValueError('No Path Found')
def next_move(grid):
    #Convert all the points to instances of Node
    for x in range(len(grid)):
        for y in range(len(grid[x])):
            grid[x][y] = Node(grid[x][y],(x,y))
    return grid

   	
class Bot1:
    ACTION_GO_LEFT = 0
    ACTION_GO_RIGHT = 1
    ACTION_GO_UP = 2
    ACTION_GO_DOWN = 3
    ACTION_FREE = 4
    ACTION_CRAFT = 5

    def __init__(self, id):
        self.state = State()
        self.info = PlayerInfo(id)
        self.path = None
        self.grid = next_move([[0 for i in range(9)] for j in range(21)])
        self.idx = 0
    def path_generator(self, grid):
      self.idx = 0
      pos = (self.state.x, self.state.y)
      start = Node(1000, pos)
      self.grid = self.update(self.grid)
      goal = self.gold_sort(self.grid)
      self.path = aStar(start, goal, self.grid)
      for node in self.path:
        print(node.point)

    def next_action(self):
      if self.path is not None and self.idx < len(self.path):
        x = self.path[self.idx]  
        if self.idx != len(self.path)-1:  
          nextx = self.path[self.idx+1]
        if (self.info.energy <=10): 
          return self.ACTION_FREE
        else: 
          if self.state.mapInfo.gold_amount(self.info.posx, self.info.posy) > 0:
            return self.ACTION_CRAFT
          else:
            if ((nextx.point[0]-x.point[0] == 0) and (nextx.point[1]-x.point[1] == 1)):
              return self.ACTION_GO_DOWN
            else:
              if ((nextx.point[0]-x.point[0] == 0) and (nextx.point[1]-x.point[1] == -1)): 
                return self.ACTION_GO_UP
              else:
                if ((nextx.point[0]-x.point[0] == 1) and (nextx.point[1]-x.point[1] == 0)): 
                  return self.ACTION_GO_RIGHT
                else: 
                  return self.ACTION_GO_LEFT
        self.idx += 1    
      else:
        self.path_generator(self.grid)
        return 4
    def update(self, grid):
      for i in range(0, 21):
        for j in range(0, 9):
          for m in self.state.mapInfo.obstacles:
            if (i == m["posx"] and j == m["posy"]): 
              grid[i][j].value = abs(m["value"])
          for m in self.state.mapInfo.golds:
            if (i == m["posx"] and j == m["posy"]): 
              grid[i][j].value = 0
      return grid


    def gold_sort(self, grid):
      gold = []
      for i in range (0,21):
        for j in range (0,9):
	        if (grid[i][j].value == 0): 
	          gold.append(grid[i][j])
      pos = (self.state.x, self.state.y)
      val = 1000
      temp = Node(val, pos)
      for x in gold:
	      x.value = manhattan(x, temp)
      gold.sort(key = lambda x: x.value)
      return gold[0]

         

    def new_game(self, data):
        try:
            self.state.init_state(data)
        except Exception as e:
            import traceback
            traceback.print_exc()

    def new_state(self, data):
        # action = self.next_action();
        # self.socket.send(action)
        try:
            self.state.update_state(data)
        except Exception as e:
            import traceback
            traceback.print_exc()