import tensorflow as tf

#class representing a neural network computing q-values
class QNetwork:

    #possible activation functions
    activations = {
        "relu" : tf.nn.relu,
        "sigmoid" : tf.sigmoid,
        "tanh" : tf.tanh
    }

    def __init__(self,config,name,graph,num_actions,input_placeholder,trainable=True):
        #name prefix for tensor operations of this neural network
        self.name = name
        self.graph = graph
        #number of actions the agent can perform
        self.num_actions = num_actions
        #if false, variables in this graph are not trained
        self.trainable = trainable

        self.input_placeholder = input_placeholder
        #output tensor of this neural network
        #contains the q-values and has shape [batch_size,num_actions]
        self.q_values = None
        self.variables = []
        self.next_name_id = 0

        self.DEFAULT_WEIGHT_INITIALIZER = "variance_scaling" 
        self.DEFAULT_BIAS_INITIALIZER = "zeros" 

        self.create_model_from_config(config)

    #creates a neural network based on a json object specifying the architecture
    def create_model_from_config(self,config):
        with self.graph.as_default():
            with tf.name_scope(self.name):
                current_input = self.input_placeholder
                current_input_a = None
                current_input_v = None
                for i in range(len(config)):
                    layer = config[i]
                    layer_type = layer["type"]
                    
                    #add convolutional layers
                    #since we use "VALID" padding, a strides value should divide width-filter_width and height_filter_height
                    if layer_type == "conv":
                        num_filters = layer["num_filters"]
                        filter_sizes = layer["filter_sizes"]
                        strides = layer["strides"]
                        activation = layer["activation"]
    
                        for j in range(len(num_filters)):
                            real_strides = [1,strides[j],strides[j],1]
                            current_input = self.add_conv_layer(current_input,num_filters[j],filter_sizes[j],strides=real_strides,activation=activation)
    
                        #reshape to [batch_size,x]
                        current_shape = self.get_tensor_shape(current_input)
                        new_size = 1 
                        for j in range(1,len(current_shape)):
                            new_size *= current_shape[j]
                        current_input = tf.reshape(current_input,[-1,new_size])		
                    
                    #add a fully connected layer
                    elif layer_type == "dense":
                        sizes = layer["sizes"]
                        activation = layer["activation"]
    
                        for j in range(len(sizes)):
                            current_input = self.add_dense_layer(current_input,sizes[j],activation=activation)
                            
                    #add to streams of fully connected layers
                    #used for Dueling Network Architecture
                    elif layer_type == "sep_dense":
                        current_input_a = current_input
                        current_input_v = current_input
                        sizes = layer["sizes"]
                        activation = layer["activation"]
                        for j in range(len(sizes)):
                            current_input_a = self.add_dense_layer(current_input_a,sizes[j],activation=activation)
                            current_input_v = self.add_dense_layer(current_input_v,sizes[j],activation=activation)
               
                if (current_input_a is None) or (current_input_v is None):
                    current_input_a = current_input
                    current_input_v = current_input
                
                output_a = self.add_dense_layer(current_input_a,self.num_actions)
                output_v = self.add_dense_layer(current_input_v,1)
                
                #combine the two streams of the dueling architecture into a single q-value  
                mean_a = tf.reduce_mean(output_a,axis=1,keepdims=True,name="mean_a")
                action_advantage = tf.subtract(output_a,mean_a)
                self.q_values = tf.add(action_advantage,output_v)
    
   
    def get_initial_value(self, name, shape):
        with self.graph.as_default():
                if name == "zeros":
                    return tf.zeros(shape)
                elif name == "random_normal":
                    return tf.random_normal(shape)
                elif name == "variance_scaling":
                    return tf.variance_scaling_initializer()(shape)
                elif name == "glorot_uniform":
                    return tf.glorot_uniform_initializer()(shape)
                elif name == "glorot_normal":
                    return tf.glorot_normal_initializer()(shape)
                elif name == "uniform_unit_scaling":
                    return tf.uniform_unit_scaling_initializer()(shape)
                else:
                    print("Error: unknown initializer")

    #adds a fully connected layer to the given input and applies an activation function (if provided)
    def add_dense_layer(self,inputs,size,use_bias=True,activation=None,weight_initializer=None,bias_initializer=None):
        with self.graph.as_default():
                if weight_initializer is None:
                    weight_initializer = self.DEFAULT_WEIGHT_INITIALIZER
                if bias_initializer is None:
                    bias_initializer = self.DEFAULT_BIAS_INITIALIZER

                input_shape = self.get_tensor_shape(inputs)
                #reshape input to [batch_size,x], i.e. one vector per example in the batch
                if len(input_shape) > 2:
                    #vector size is product of the sizes of all dimensions (except first)
                    new_size = 1
                    for i in range(1,len(input_shape)):
                        new_size *= input_shape[i]

                    input_shape = [-1,new_size]
                    inputs = tf.reshape(inputs,input_shape)

                #create weight matrix and bias vector and initialization operations
                shape = [input_shape[-1],size]
                bias_shape = [size]
                weights = tf.Variable(self.get_initial_value(weight_initializer,shape),trainable=self.trainable,name=self.get_name("dense_weights"))
                bias = tf.Variable(self.get_initial_value(bias_initializer,bias_shape),trainable=self.trainable,name=self.get_name("dense_bias"))

                #add weight matrix and bias vector to variable list
                self.add_variable(weights)
                self.add_variable(bias)

                output = tf.nn.bias_add(tf.matmul(inputs,weights,name=self.get_name("dense_matmul")),bias,name=self.get_name("dense_bias_add"))
                return self.add_activation(output,activation)

    #adds a convolutional layer to the given input
    #since we use "VALID" padding, the strides values should divide width-filter_width and height_filter_height respectively, otherwise some elements of the input will not be used
    def add_conv_layer(self,inputs,num_filters,filter_size,strides=[1,1,1,1],activation=None,weight_initializer=None,bias_initializer=None):
        with self.graph.as_default():
                if weight_initializer is None:
                    weight_initializer = self.DEFAULT_WEIGHT_INITIALIZER
                if bias_initializer is None:
                    bias_initializer = self.DEFAULT_BIAS_INITIALIZER

                input_shape = self.get_tensor_shape(inputs)
                if len(input_shape) == 3:
                    inputs = tf.expand_dims(inputs,3)

                filter_shape = [filter_size[0],filter_size[1],input_shape[3],num_filters]
                bias_shape = [num_filters]
                filter_weights = tf.Variable(self.get_initial_value(weight_initializer,filter_shape),trainable=self.trainable,name=self.get_name("filter_weights"))
                bias = tf.Variable(self.get_initial_value(bias_initializer,bias_shape),trainable=self.trainable,name=self.get_name("filter_bias"))
                self.add_variable(filter_weights)
                self.add_variable(bias)

                conv = tf.nn.conv2d(inputs,filter_weights,strides=strides,padding="VALID",name=self.get_name("conv"))
                activ = self.add_activation(tf.nn.bias_add(conv,bias,name=self.get_name("bias_add")),activation)

                return activ
    
    #applies the given activation function to the input tensor
    def add_activation(self,inputs,activation):
        with self.graph.as_default():
                if activation is None:
                    return inputs
                elif activation not in QNetwork.activations:
                    print("Warning: unknown activation function {}".format(activation))
                    return inputs
                else:
                    return QNetwork.activations[activation](inputs)
        
    def add_variable(self, variable):
        self.variables.append(variable)

    def add_variables(self,variables):
        self.variables.extend(variables)

    def get_tensor_shape(self,tensor):
        return tensor.get_shape().as_list()

    def get_name(self,base):
        name = base + "_" + str(self.next_name_id)
        self.next_name_id += 1
        return name

