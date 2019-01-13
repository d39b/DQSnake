import numpy as np
from util import LinkedList

#simple implemntation of the game Snake
class Snake:

    num_actions = 4
    action_names = ["down","left","up","right"]
    EMPTY = 0
    HEAD = 1
    BODY = 2
    GOAL = 3
    obj_colors = [
        [0, 0, 0], #rgb value of EMPTY
        [1.0,0,0], #rgb value of HEAD
        [0,0,1.0], #rgb value of BODY
        [0,1.0,0]  #rgb value of GOAL
    ]
    num_channels = 3
    #reward on game over
    min_reward = -0.2
    #reward for finding a goal
    max_reward = 1.0
    #this value could be used to set a negative reward for all moves that would usually yield a 0 reward
    #thereby forcing the agent to minimize the number of such moves
    empty_reward = 0.0

    def __init__(self,width,height,image_scale_factor,num_goals):
        self.width = width
        self.height = height
        self.image_scale_factor = image_scale_factor
        self.image_width = self.width * image_scale_factor
        self.image_height = self.height * image_scale_factor
        self.num_goals = num_goals
        self.length = 0

        #init game
        self.reset()

    def execute_action(self,action):
        reward = 0
        is_terminal = 0

        #0: left 1: down 2: right 3: up
        x_old = self.head_x
        y_old = self.head_y 
        x_new = x_old
        y_new = y_old
        if action == 0:
            x_new = (x_old - 1) 
        elif action == 1:
            y_new = (y_old - 1)
        elif action == 2:
            x_new = (x_old + 1)
        else:
            y_new = (y_old + 1)

        #check if snake tries to move outside box
        if x_new < 0 or y_new < 0 or x_new >= self.width or y_new >= self.height:
            reward = self.min_reward
            is_terminal = True
            return reward, is_terminal

        self.head_x = x_new
        self.head_y = y_new

        if self.state[x_new,y_new] == self.EMPTY:
            self.update_state(x_new,y_new,self.HEAD)
            if self.length == 1:
                self.update_state(x_old,y_old,self.EMPTY)
            else:
                self.body.add_head([x_old,y_old])
                self.update_state(x_old,y_old,self.BODY)
                tail_pos = self.body.pop_tail()
                self.update_state(tail_pos[0],tail_pos[1],self.EMPTY)
            reward = self.empty_reward
        #snake finds a goal
        elif self.state[x_new,y_new] == self.GOAL: 
            self.update_state(x_new,y_new,self.HEAD)
            self.update_state(x_old,y_old,self.BODY)
            self.body.add_head([x_old,y_old])
            reward = self.max_reward
            self.length += 1
            if self.length == self.width * self.height:
                is_terminal = True
                reward += self.max_reward
            else:
                if self.length < (self.width*self.height - self.num_goals + 1):
                    g_x, g_y = self.get_random_pos()
                    self.update_state(g_x,g_y,self.GOAL)
        #snake hits its own body
        elif self.state[x_new,y_new] == self.BODY:
            head_pos = self.body.get_head()
            tail_pos = self.body.get_tail()
            #snake can move to the position of the current tail
            if x_new == tail_pos[0] and y_new == tail_pos[1]:
                self.update_state(x_new,y_new,self.HEAD)
                self.body.add_head([x_old,y_old])
                self.update_state(x_old,y_old,self.BODY)
                self.body.pop_tail()
                reward = self.empty_reward
            else:
                self.update_state(x_new,y_new,self.HEAD)
                is_terminal = True
                reward = self.min_reward

        return reward, is_terminal

    def reset(self):
        self.state = np.zeros([self.width,self.height],dtype=np.uint8)
        self.state_img = np.zeros([1,self.image_width,self.image_height,self.num_channels],dtype=np.float32)
        self.length = 1
        
        self.head_x, self.head_y = self.get_random_pos()
        self.update_state(self.head_x,self.head_y,self.HEAD)
        for i in range(self.num_goals):
            g_x, g_y = self.get_random_pos() 
            self.update_state(g_x,g_y,self.GOAL)

        self.body = LinkedList()        

    def update_state(self,x_pos,y_pos,obj_type):
        self.state[x_pos,y_pos] = obj_type
        self.set_pixel(x_pos,y_pos,obj_type)

    def set_pixel(self,x_pos,y_pos,obj_type):
        x_start = x_pos * self.image_scale_factor
        x_end = (x_pos+1) * self.image_scale_factor
        y_start = y_pos * self.image_scale_factor
        y_end = (y_pos+1) * self.image_scale_factor
        color = self.obj_colors[obj_type]
        for i in range(self.num_channels):
            self.state_img[0,x_start:x_end,y_start:y_end,i] = color[i]

    def get_random_pos(self):
        found = False
        r_x = 0
        r_y = 0
        while not found:
            r_x = np.random.randint(self.width)
            r_y = np.random.randint(self.height)

            if self.state[r_x,r_y] == self.EMPTY:
                found = True
                
        return r_x,r_y
            
    def get_state(self):
        return self.state_img


#a game in which the player has to push a box onto a goal position
class Box:

    num_actions = 4
    action_names = ["down","left","up","right"]
    EMPTY = 0
    PLAYER = 1
    BOX = 2
    GOAL = 3
    obj_colors = [
        [0, 0, 0], #rgb value of EMPTY
        [1.0,0,0], #rgb value of PLAYER
        [0,0,1.0], #rgb value of BOX
        [0,1.0,0]  #rgb value of GOAL         
    ]
    num_channels = 3
    min_reward = -0.2
    max_reward = 1.0
    empty_reward = 0.0

    def __init__(self,width,height,image_scale_factor,num_boxes):
        self.width = width
        self.height = height
        self.image_scale_factor = image_scale_factor
        self.image_width = self.width * image_scale_factor
        self.image_height = self.height * image_scale_factor
        self.num_boxes = num_boxes

        #init game
        self.reset()

    def execute_action(self,action):
        reward = 0
        is_terminal = 0

        #0: left 1: down 2: right 3: up
        x_old = self.player_x
        y_old = self.player_y 
        x_new = x_old
        y_new = y_old
        if action == 0:
            x_new = (x_old - 1) % self.width
        elif action == 1:
            y_new = (y_old - 1) % self.height
        elif action == 2:
            x_new = (x_old + 1) % self.width
        else:
            y_new = (y_old + 1) % self.height

        self.player_x = x_new
        self.player_y = y_new
        
        #update the old player position only if the player wasn't on the goal position
        if self.state[x_old,y_old] == self.PLAYER: 
            self.update_state(x_old,y_old,self.EMPTY)
        
        #player moves onto an empty position
        if self.state[x_new,y_new] == self.EMPTY:            
            self.update_state(x_new,y_new,self.PLAYER)
        #player pushes the box
        elif self.state[x_new,y_new] == self.BOX:   
            box_new_x = x_new
            box_new_y = y_new
            if action == 0:
                box_new_x = (box_new_x - 1) % self.width
            elif action == 1:
                box_new_y = (box_new_y - 1) % self.height
            elif action == 2:
                box_new_x = (box_new_x + 1) % self.width
            else:
                box_new_y = (box_new_y + 1) % self.height
            
            #if the box lands on the goal position the game is over
            if self.state[box_new_x,box_new_y] == self.GOAL:
                reward = self.max_reward
                is_terminal = True
            
            self.update_state(x_new,y_new,self.PLAYER)
            self.update_state(box_new_x,box_new_y,self.BOX)                
            
        return reward, is_terminal

    def reset(self):
        self.state = np.zeros([self.width,self.height],dtype=np.uint8)
        self.state_img = np.zeros([1,self.image_width,self.image_height,self.num_channels],dtype=np.float32)

        self.player_x, self.player_y = self.get_random_pos()
        self.update_state(self.player_x,self.player_y,self.PLAYER)
        g_x, g_y = self.get_random_pos() 
        self.update_state(g_x,g_y,self.GOAL)
        
        b_x, b_y = self.get_random_pos() 
        self.update_state(b_x,b_y,self.BOX)        

    def update_state(self,x_pos,y_pos,obj_type):
        self.state[x_pos,y_pos] = obj_type
        self.set_pixel(x_pos,y_pos,obj_type)

    def set_pixel(self,x_pos,y_pos,obj_type):
        x_start = x_pos * self.image_scale_factor
        x_end = (x_pos+1) * self.image_scale_factor
        y_start = y_pos * self.image_scale_factor
        y_end = (y_pos+1) * self.image_scale_factor
        color = self.obj_colors[obj_type]
        for i in range(self.num_channels):
            self.state_img[0,x_start:x_end,y_start:y_end,i] = color[i]

    def get_random_pos(self):
        found = False
        r_x = 0
        r_y = 0
        while not found:
            r_x = np.random.randint(self.width)
            r_y = np.random.randint(self.height)

            if self.state[r_x,r_y] == self.EMPTY:
                found = True
                
        return r_x,r_y        

    def get_state(self):
        return self.state_img