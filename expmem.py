import numpy as np
from util import SumTree
from util import MaxHeap

class ExperienceMemory:

    #TODO more complex sampling method to first sample from games and then from frames in that game, that way we can avoid sampling terminal states

    def __init__(self,max_size,width,height,num_channels,alpha):
        #TODO round to next biggest power of 2 here
        self.max_size = max_size
        self.frames = np.zeros([max_size,width,height,num_channels],dtype=np.float32)
        self.actions = np.zeros(max_size,dtype=int)
        self.rewards = np.zeros([max_size,1],dtype=float)
        self.terminal = np.zeros([max_size,1],dtype=int)
        self.size = 0 
        self.next_index = 0
        self.sum_tree = SumTree(max_size,alpha)
        self.max_heap = MaxHeap(max_size)
        self.td_epsilon = 1e-9

    def add(self,s,a,r,t):
        i = self.next_index
        self.frames[i,:,:] = s
        self.actions[i] = a
        self.rewards[i] = r
        self.terminal[i] = t
        
        p_value = 1
        if self.size > 0:
            p_value = self.max_heap.get_max()
        self.sum_tree.update(i,p_value)
        self.max_heap.update(i,p_value)
                
        if self.size < self.max_size:
            self.size += 1
        
        self.next_index = (self.next_index + 1) % self.max_size			

    def sample(self,n):
        #TODO this might be wrong, but because i is the successor state we shouldn't sample 0
        #if buffer is not full we should not sample last element because it has no successor state
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

        #for j in range(len(i)):
        #    if i[j] >= self.next_index:  
        #        print("error: index is too big")
        #        print(self.next_index)
        #        print(i[j])
        #        print()

        return s,a,r,s2,t,i,p_values
        
    def update_p(self,indices,td):
        for i in range(len(indices)):
            self.sum_tree.update(indices[i],td[i]+self.td_epsilon)
            self.max_heap.update(indices[i],td[i]+self.td_epsilon)