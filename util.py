import numpy as np

#simple implementation of a doubly linked list
#a doubly linked list has a head and a tail element
#elements can be appended and removed from either end of the list
class LinkedList:
    
    def __init__(self):
        self.size = 0
        self.head = None
        self.tail = None

    def add_head(self,item): 
        if self.size == 0:
            self.head = Node(item) 
            self.tail = self.head
        else:
            new_head = Node(item,left=self.head)
            self.head.right = new_head
            self.head = new_head

        self.size += 1

    def add_tail(self,item):
        if self.size == 0:
            self.tail = Node(item)
            self.head = self.tail
        else:
            new_tail = Node(item,right=self.tail)
            self.tail.left = new_tail
            self.tail = new_tail

        self.size += 1

    def pop_head(self):
        item = self.head.item

        if self.size == 1:
            self.head = None
            self.tail = None
        else:
            self.head = self.head.left
            self.head.right = None

        self.size -= 1
        return item

    def pop_tail(self):
        item = self.tail.item

        if self.size == 1:
            self.head = None
            self.tail = None
        else:
            self.tail = self.tail.right
            self.tail.left = None

        self.size -= 1
        return item

    def get_head(self):
        return self.head.item

    def get_tail(self):
        return self.tail.item

    def reverse(self):
        if self.size > 1:
            curr = self.tail
            while not curr.right is None:
                tmp = curr.right
                curr.right = curr.left
                curr.left = tmp
                curr = curr.left
            curr.right = curr.left
            self.head = self.tail
            self.tail = curr

 
#a node is an element of a linked list and stores a single item/value
class Node:

    def __init__(self,item,left=None,right=None):
        self.item = item
        self.left = left
        self.right = right
        

#a sum tree is a binary tree in which the value of an internal node is the sum of the values of the child nodes
#this data structure can be used to sample from a probability distribution over {1,2,...,n} given by a set of priority values p_i
#i.e. Pr[i] = p_i / T with T = p_1 + p_2 + ... + p_n
#for this purpose the values p_i are stored at the leafs of the sum tree (the root then contains the value T)
#to sample an index from {1,...,n} according to the distribution defined above, we select a value x in [0,T] uniformly at random and find
#the index i such that p_1 + ... + p_{i-1} < x <= p_1 + ... + p_i
#inserting and updating values as well as sampling indices can be done in O(log n) time
class SumTree:

    def __init__(self,num_leafs,alpha): 
        #to simplify the implementation, the number of leafs is assumed to be a power of 2
        self.num_leafs = num_leafs
        #instead of storing a given value p we store p^alpha
        self.alpha = alpha
        #number of internal nodes is num_leafs - 1
        self.size = 2*self.num_leafs - 1
        #index of the first leaf
        self.base_index = self.num_leafs - 1
        self.arr = np.zeros(self.size,dtype=np.float32)
        
    def update(self,index,value):
        real_index = self.base_index + index
        old_value = self.arr[real_index]
        self.arr[real_index] = value**self.alpha
        
        parent_index = (real_index-1)//2
        #in the end parent_index should be (0-1)//2 = -1 < 0
        while parent_index >= 0:
            left_child = parent_index*2+1
            right_child = parent_index*2+2
            self.arr[parent_index] = self.arr[left_child] + self.arr[right_child]                         
            parent_index = (parent_index-1)//2
    
    def sample(self,n):
        #to sample n indices from the distribution defined by this sum tree we divide the range [0,T] into n parts of equal length,
        #uniformly sample a value from each part and find the index of the corresponding leaf
        p_total = self.arr[0]
        p_step = p_total/n
        indices = np.zeros(n,dtype=int)
        p_values = np.zeros([n,1],dtype=np.float32)
        
        for i in range(n):
            v_min = i*p_step
            v_max = (i+1)*p_step            
            value = np.random.uniform(low=v_min,high=v_max)                
            
            curr_index = 0
            low = 0
            high = p_total
            #if index is < base_index we have not yet reached a leaf of the tree
            while curr_index < self.base_index:
                left_child = curr_index*2 + 1
                right_child = curr_index*2 + 2


                border = low + self.arr[left_child]
                if value <= border:
                    high = border
                    curr_index = left_child
                else:
                    low = border
                    curr_index = right_child
            
            indices[i] = curr_index - self.base_index
            p_values[i] = self.arr[curr_index]/p_total
        
        #return the indices and the corresponding priority values
        return indices, p_values
        
        
#implementation of a max heap using an array
#a max heap is a binary tree in which the value of an internal node is the maximum of the values of the child nodes
#can be used to maintain the maximum of a set of values
#updating/inserting values is O(log n) where n is the number of stored values
#finding the maximum is O(1)
class MaxHeap:

    def __init__(self,num_leafs): 
        self.num_leafs = num_leafs
        #number of internal nodes is num_leafs - 1
        self.size = 2*self.num_leafs - 1
        #index of first leaf
        self.base_index = self.num_leafs - 1
        self.arr = np.zeros(self.size,dtype=np.float32)
        
    def update(self,index,value):
        real_index = self.base_index + index   
        self.arr[real_index] = value        
    
        parent_index = (real_index-1)//2
        while parent_index >= 0:
            parent_value = self.arr[parent_index]
            left_child = self.arr[parent_index*2 + 1]
            right_child = self.arr[parent_index*2 + 2]
            new_parent_value = max(left_child,right_child)
            if new_parent_value == parent_value:
                break
            else:
                self.arr[parent_index] = new_parent_value
            parent_index = (parent_index-1)//2
            
    def get_max(self):
        return self.arr[0]


