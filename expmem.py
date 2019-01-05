import numpy as np
from util import SumTree
from util import MaxHeap

#class that stores experiences of an agent playing a game
#experiences are usually given in the form of transitions
#a transition is a tuple (s,a,s2,r,t) where:
#   s is a RGB or grayscale image representing the current game state
#   a is the action (index) chosen by the agent in the game state s
#   s2 is the RGB or grayscale image of the game state obtained by executing action a in state s
#   r is the reward the agent gets for executing action a in state s, r is usually integer or real valued
#   t is a boolean indicating whether s2 is a terminal state (i.e. the game is over)
class ExperienceMemory:

    def __init__(self,max_size,width,height,num_channels,alpha):
        self.max_size = max_size
        #frames/images are stored in sequence, i.e. frames[i+1] is the state obtained by executing action actions[i+1] in state frames[i]
        self.frames = np.zeros([max_size,width,height,num_channels],dtype=np.float32)
        self.actions = np.zeros(max_size,dtype=int)
        self.rewards = np.zeros([max_size,1],dtype=np.float32)
        self.terminal = np.zeros([max_size,1],dtype=int)
        self.size = 0 
        self.next_index = 0
        #use the next power of 2 as the size for the sum tree/max heap, this simplifies their implementation
        power_2_size = self.get_next_power(max_size)
        self.sum_tree = SumTree(power_2_size,alpha)
        self.max_heap = MaxHeap(power_2_size)
        #additive constant to keep td-values above 0
        self.td_epsilon = 1e-9

    def add(self,s,a,r,t):
        i = self.next_index
        self.frames[i,:,:] = s
        self.actions[i] = a
        self.rewards[i] = r
        self.terminal[i] = t
        
        #a new element gets maximum priority value
        #this ensures that each transition is replayed at least once (with high probability)
        p_value = 1
        if self.size > 0:
            p_value = self.max_heap.get_max()
        self.sum_tree.update(i,p_value)
        self.max_heap.update(i,p_value)
                
        if self.size < self.max_size:
            self.size += 1
        
        self.next_index = (self.next_index + 1) % self.max_size			

    def sample(self,n):
        #if the replay memory is not full we should not sample the frame 0 since it has no predecessor frame
        min_index = 1
        max_index = self.size
        if self.size == self.max_size:
            min_index = 0
        
        i, p_values = self.sum_tree.sample(n)
        i_minus = (i-1) % self.size

        s = self.frames[i_minus,:,:,:] 
        a = self.actions[i] 
        r = self.rewards[i] 
        s2 = self.frames[i,:,:,:] 
        t = self.terminal[i]

        return s,a,r,s2,t,i,p_values
        
    #update priority value of replayed transitions with their new td-error
    def update_p(self,indices,td):
        for i in range(len(indices)):
            self.sum_tree.update(indices[i],td[i]+self.td_epsilon)
            self.max_heap.update(indices[i],td[i]+self.td_epsilon)
    
    #returns y >= x such that y is a power of 2
    def get_next_power(self,x):
        result = 1
        if x > 0:
            x_int = int(x-1)
            while x_int > 0:
                x_int = x_int >> 1
                result = result << 1
            
        return result