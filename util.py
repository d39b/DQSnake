#simple implementation of a double linked list
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

    #TODO make this more efficient, maybe have a direction attribute
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

        
class Node:

    def __init__(self,item,left=None,right=None):
        self.item = item
        self.left = left
        self.right = right
        
#TODO split into Sum Tree and MaxHeap    
class SumTree:

    #num_leafs should be a power of 2
    def __init__(self,num_leafs,alpha):        
        self.num_leafs = num_leafs
        self.alpha = alpha
        #number of internal nodes is num_leafs - 1
        self.size = 2*self.num_leafs - 1
        self.base_index = self.num_leafs - 1
        self.arr = np.zeros(self.size,dtype=np.float32)
        
    def update(self,index,value):
        real_index = self.base_index + index
        old_value = self.arr[real_index]
        self.arr[real_index] = value**self.alpha
        
        #value_change = (value**self.alpha) - old_value
        
        #old_top = self.arr[0]
        #TODO use bitshift here?
        parent_index = (real_index-1)//2
        #in the end parent_index should be (0-1)//2 = -1 < 0
        while parent_index >= 0:
            left_child = parent_index*2+1
            right_child = parent_index*2+2
            self.arr[parent_index] = self.arr[left_child] + self.arr[right_child]                         
            #self.arr[parent_index] += value_change                        
            #if self.arr[parent_index] != (self.arr[left_child] + self.arr[right_child]):
            #        print("SUM ERROR IN TREE")
            parent_index = (parent_index-1)//2

        #if np.isnan(self.arr[0]):
        #    print("NaN detected")
        #    print(old_top)
        #    print(value)
        #    print(value_change)
        #    print()
    
    def sample(self,n):
        p_total = self.arr[0]
        p_step = p_total/n
        indices = np.zeros(n,dtype=int)
        p_values = np.zeros([n,1],dtype=np.float32)
        
        for i in range(n):
            v_min = i*p_step
            v_max = (i+1)*p_step
            #if v_max > 10e10:
            #    print(v_max)
            #if v_min < -10e10:
            #    print(v_min)
            #if v_max < v_min:
            #    print("error vmax < vmin")
            #value = 0
            #try:
            value = np.random.uniform(low=v_min,high=v_max)
            #except OverflowError:
            #    print("overflow error detected")
            #    print(v_max)
            #    print(v_min)
            #    print(p_total)
            #    print(p_step)
            #    print("overflow error end")
            #    print()
                
            
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
            #if p_total == 0.0:
            #    print("error, ptotal is 0")
            #if self.arr[curr_index] == 0:
            #    zero_found = False
            #    first_zero = None
            #    last_non_zero = None
            #    for j in range(self.num_leafs):
            #        test_index = self.base_index + j
            #        test_value = self.arr[test_index]
            #        if not zero_found:
            #            if test_value == 0:
            #                zero_found = True
            #                first_zero = test_index
            #        else:
            #            if test_value != 0:
            #                last_non_zero = test_index
            #    print("first_zero: {}  last_non_zero: {}".format(first_zero,last_non_zero))

            #    print("error: p_value is 0")
            #    print("sample value: {}".format(value))
            #    curr_index = 0
            #    low = 0
            #    high = p_total
            #    while curr_index < self.base_index:
            #        left_child = curr_index*2 + 1
            #        right_child = curr_index*2 + 2

            #        border = low + self.arr[left_child]
            #        print("low: {}  high: {}  border: {}".format(low,high,border))
            #        print("left: {}  right: {}".format(left_child,right_child))
            #        print("vleft: {}  vright: {}".format(self.arr[left_child],self.arr[right_child]))
            #        if self.arr[left_child] == 0:
            #            print("error: left child is zero here")
            #        if value <= border:
            #            high = border
            #            curr_index = left_child
            #        else:
            #            low = border
            #            curr_index = right_child

            p_values[i] = self.arr[curr_index]/p_total
            
        return indices, p_values
        
        
class MaxHeap:

    def __init__(self,num_leafs,alpha):       
        self.num_leafs = num_leafs
        #number of internal nodes is num_leafs - 1
        self.size = 2*self.num_leafs - 1
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


