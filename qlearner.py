import tensorflow as tf
import numpy as np
from qnetwork import QNetwork

class QLearner:

    def __init__(self,config,num_actions,width,height,num_channels,memory_size,load_model=None,target_network_update_tau=None):
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)

        self.num_actions = num_actions
        self.width = width
        self.height = height
        self.num_channels = num_channels
        self.memory_size = memory_size
        self.target_network_update_tau = target_network_update_tau    

        layers = config["layers"]
        self.clip_max = config["clip_max"]
        self.clip_grad_norm = None
        if "clip_grad_norm" in config:
            self.clip_grad_norm = config["clip_grad_norm"]
            print("Clipping gradient norm to {}".format(self.clip_grad_norm))
        self.lr = config["learning_rate"]
        self.rms_decay = config["rms_decay"] 
        self.gamma = config["gamma"]
        self.loss_type = config["loss"]
        self.optimizer_type = config["optimizer"]

        #placeholders
        with self.graph.as_default():
            self.state_train_placeholder = tf.placeholder(tf.float32,[None,self.width,self.height,self.num_channels])
            self.state_target_placeholder = tf.placeholder(tf.float32,shape=[None,self.width,self.height,self.num_channels])
            self.action_index_placeholder = tf.placeholder(tf.int32,shape=[None])
            self.reward_placeholder = tf.placeholder(tf.float32,shape=[None,1])
            self.terminal_placeholder = tf.placeholder(tf.float32,shape=[None,1])
            self.beta_placeholder = tf.placeholder(tf.float32,shape=[])
            self.p_placeholder = tf.placeholder(tf.float32,shape=[None,1])

        #create q networks
        self.train_network = QNetwork(layers, "train", self.graph, self.num_actions,self.state_train_placeholder,trainable=True)  
        self.target_network = QNetwork(layers, "target", self.graph, self.num_actions,self.state_target_placeholder,trainable=False)  
        self.add_training_ops()
        self.create_target_update_operations(tau=self.target_network_update_tau)
        self.add_saver()

        if not load_model is None:
            self.load_model(load_model) 
            self.variables_initialized = True
        else:
            with self.graph.as_default():
                self.init_op = tf.global_variables_initializer() 
            self.run_operations(self.init_op)
            self.update_target_network()
            self.variables_initialized = True
        

    def add_tensorboard_ops(self,path):
        with self.graph.as_default():
            summary_loss = tf.summary.scalar('loss',self.loss)
            summary_q_values = tf.summary.histogram('q_values',self.train_network.q_values)

            #for v in self.train_network.variables:
            #    tf.summary.histogram(v.name,v)
            
            #gradient norms
            for g in self.gradients:
                tf.summary.scalar(g[1].name,tf.reduce_sum(tf.square(g[0])))

            self.summary_op = tf.summary.merge_all()
            self.summary_writer = tf.summary.FileWriter(path,self.graph)	


    def train_step(self,s,a,r,s2,t,p_values,beta,write_summary=False):
        state_concat = np.concatenate((s,s2))  
        feed_dict = {
            self.state_train_placeholder : state_concat,
            self.state_target_placeholder : s2,
            self.action_index_placeholder : a,
            self.reward_placeholder : r,
            self.terminal_placeholder : t,
            self.beta_placeholder : beta,
            self.p_placeholder : p_values
        }

        train_ops = [self.global_step,self.loss,self.td,self.train_op]
        if write_summary:
            train_ops.append(self.summary_op)
        
        #train_ops.extend(self.debug_output)
        
        results = self.run_operations(train_ops,feed_dict=feed_dict)
        step = results[0]
        loss = results[1]
        td = results[2]
        
        #for j in range(3,len(train_ops)):
        #    print(self.debug_output_name[j-3])
        #    print(results[j])

        
        if write_summary:
            summary = results[-1]
            self.summary_writer.add_summary(summary, global_step=step)
            
        return (step,loss,td)

    def compute_q(self,s):
        feed_dict = {
            self.state_target_placeholder : s
        }

        q_op = self.target_network.q_values
        q_values = self.run_operations(q_op,feed_dict=feed_dict)
        return q_values  

    def compute_action(self,s):
        q_values = self.compute_q(s)
        #print(q_values)
        q_max = np.argmax(q_values,axis=1)
        return q_max

    def add_training_ops(self):
        with self.graph.as_default():
            #self.debug_output = []
            #self.debug_output_name = []
            train_q_values = self.train_network.q_values
            target_q_values = self.target_network.q_values
            #self.debug_output.append(target_q_values)
            #self.debug_output_name.append("target_q_values")
            action_one_hot = tf.one_hot(self.action_index_placeholder,self.num_actions,dtype=tf.float32)
            #self.debug_output.append(self.reward_placeholder)
            #self.debug_output_name.append("rewards_placeholder")
            #self.debug_output.append(self.action_index_placeholder)
            #self.debug_output_name.append("action_index")
            #self.debug_output.append(action_one_hot)
            #self.debug_output_name.append("action_one_hot")
            non_terminal = tf.subtract(tf.constant(1.0),self.terminal_placeholder)

            train_q_values_split = tf.split(train_q_values,2)
            train_q_values_1 = tf.multiply(train_q_values_split[0],action_one_hot)
            train_q_values_2 = train_q_values_split[1]
            
            #compute importance sampling weights        
            iw = tf.pow(tf.multiply((1.0/self.memory_size),tf.reciprocal(self.p_placeholder)),self.beta_placeholder)
            iw_max = tf.reduce_max(iw)
            iw = tf.divide(iw,iw_max)

            r = tf.multiply(action_one_hot,self.reward_placeholder) 
            #self.debug_output.append(r)
            #self.debug_output_name.append("rewards_one_hot")
            next_action_index = tf.argmax(train_q_values_2,axis=1,output_type=tf.int32) 
            #self.debug_output.append(next_action_index)
            #self.debug_output_name.append("next_action_index")
            row_indices = tf.range(tf.shape(next_action_index)[0])
            next_action_index = tf.stack([row_indices,next_action_index],axis=1)
            #next_action_one_hot = tf.one_hot(next_action_index,self.num_actions,dtype=tf.float32) 
            #next_action_q_value = tf.multiply(next_action_one_hot,target_q_values)
            next_action_q_value = tf.gather_nd(target_q_values,next_action_index)
            #self.debug_output.append(next_action_q_value)
            #self.debug_output_name.append("next_action_q_value")
            next_action_q_value = tf.expand_dims(next_action_q_value,axis=1)
            next_action_q_value = tf.multiply(action_one_hot,next_action_q_value)
            next_action_q_value = tf.multiply(non_terminal,next_action_q_value)
            #self.debug_output.append(next_action_q_value)
            #self.debug_output_name.append("next_action_q_value_terminal_action")
            targets = tf.add(r,tf.multiply(self.gamma,next_action_q_value)) 
            #self.debug_output.append(targets)
            #self.debug_output_name.append("targets")
            if self.loss_type == "mse":
                td = tf.subtract(targets,train_q_values_1)
                self.td = tf.reduce_max(tf.abs(td),axis=1)
                td = tf.multiply(td,iw)
                td_clipped = tf.clip_by_value(td,(-1)*self.clip_max,self.clip_max)	
                self.loss = tf.nn.l2_loss(td_clipped) 	
            else:
                self.loss = tf.losses.huber_loss(train_q_values_1,targets,delta=self.clip_max)

            
            ####TODO DEBUG####
            ###add l2 loss####
            #losses = []
            #for v in self.variables:
            #    losses.append(tf.nn.l2_loss(v))
            #l2_loss = tf.multiply(l2_lambda,tf.add_n(losses))
            #self.losses.append(l2_loss)
            #return l2_loss
            ##################

            #TODO changed learning rate here
            #TODO changed epsilon to 1e-10 from 0.01
            self.global_step = tf.Variable(0,trainable=False,name='global_step')
            
            if self.optimizer_type == "rms":
                self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr,decay=self.rms_decay,momentum=0,epsilon=1e-10,centered=True)
            elif self.optimizer_type == "adam":
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr,epsilon=1.5e-4)
            
            self.gradients = self.optimizer.compute_gradients(self.loss)
            if not self.clip_grad_norm is None:
                tmp_gradients = []
                for g in self.gradients:
                    tmp_gradients.append((tf.clip_by_norm(g[0],self.clip_grad_norm),g[1]))
                self.gradients = tmp_gradients

            self.train_op = self.optimizer.apply_gradients(self.gradients,global_step=self.global_step)
            #######
            #DEBUG#
            #tmpvars = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            #for tmpvar in tmpvars:
            #    print(tmpvar.name)
        

    def create_target_update_operations(self,tau=None):
        self.target_update_operations = []
        with self.graph.as_default():
            for i in range(len(self.train_network.variables)):
                var1 = self.target_network.variables[i]
                var2 = self.train_network.variables[i]
                update_op = None
                if not tau is None:
                    update_op = var1.assign(tf.add(tf.multiply(var2,tau),tf.multiply(var1,1-tau)))
                else:
                    update_op = var1.assign(var2)
                self.target_update_operations.append(update_op)

    def update_target_network(self):
        self.run_operations(self.target_update_operations)

    def run_operations(self,ops,feed_dict={}):
        return self.session.run(ops,feed_dict=feed_dict)

    def add_saver(self):
        """
        Adds an operation to save the weights of this neural network model.
        """

        with self.graph.as_default():
            self.saver = tf.train.Saver(save_relative_paths=True)
        return self.saver

    def save_model(self,filename):
        """
        Save the weights of this neural network model.

        Parameters
        ----------
        path : str
            directory to save model in
        """

        if self.variables_initialized:
            result = self.saver.save(self.session,filename)
            return result
        else:
            print("Error: can't save model if variables are not initialized")
            return None


    def load_model(self,filename):
        """
        Load the weights of a saved neural network model.

        Parameters
        ----------
        filename : str
            base filename of the saved model
        """

        self.saver.restore(self.session,filename)
        self.variables_initialized = True
