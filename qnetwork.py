import tensorflow as tf

class QNetwork:

    activations = {
        "relu" : tf.nn.relu,
        "sigmoid" : tf.sigmoid,
        "tanh" : tf.tanh
    }

    def __init__(self,config,name,graph,num_actions,input_placeholder,trainable=True):
        self.name = name
        self.graph = graph
        self.num_actions = num_actions
        self.trainable = trainable

        self.input_placeholder = input_placeholder
        self.q_values = None
        self.variables = []
        self.next_name_id = 0

        self.DEFAULT_WEIGHT_INITIALIZER = "variance_scaling" 
        self.DEFAULT_BIAS_INITIALIZER = "zeros" 

        self.create_model_from_config(config)

    def create_model_from_config(self,config):
        with self.graph.as_default():
            with tf.name_scope(self.name):
                current_input = self.input_placeholder
                current_input_a = None
                current_input_v = None
                for i in range(len(config)):
                    layer = config[i]
                    layer_type = layer["type"]
                    
                    if layer_type == "conv":
                        num_filters = layer["num_filters"]
                        filter_sizes = layer["filter_sizes"]
                        strides = layer["strides"]
                        activation = layer["activation"]
    
                        for j in range(len(num_filters)):
                            real_strides = [1,strides[j],strides[j],1]
                            current_input = self.add_conv_layer(current_input,num_filters[j],filter_sizes[j],strides=real_strides,activation=activation)
    
                        #TODO does this produce the correct result, i.e. keep batch size fixed
                        current_shape = self.get_tensor_shape(current_input)
                        new_size = 1 
                        for j in range(1,len(current_shape)):
                            new_size *= current_shape[j]
                        current_input = tf.reshape(current_input,[-1,new_size])		
    
                    elif layer_type == "dense":
                        sizes = layer["sizes"]
                        activation = layer["activation"]
    
                        for j in range(len(sizes)):
                            current_input = self.add_dense_layer(current_input,sizes[j],activation=activation)

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

                #result here should be a vector
                output_a = self.add_dense_layer(current_input_a,self.num_actions)
                output_v = self.add_dense_layer(current_input_v,1)
    
                mean_a = tf.reduce_mean(output_a,axis=1,keepdims=True,name="mean_a")
                action_advantage = tf.subtract(output_a,mean_a)
                self.q_values = tf.add(action_advantage,output_v)
    

    def get_initial_value(self, name, shape):
        """
        Returns an initialization operation.

        Parameters
        ----------
        name : str
            string denoting the initialization method to be used
        shape: List
            list of integers defining the shape of the initial value
        """

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


    def add_dense_layer(self,inputs,size,use_bias=True,activation=None,weight_initializer=None,bias_initializer=None):
        """
        Adds a dense (or fully connected) layer to a neural network model.

        Reshapes the input to 2 dimensions. If no weight or bias initializer is passed, the initializers defined by self.DEFAULT_WEIGHT_INITIALIZER or self.DEFAULT_BIAS_INITIALIZER are used.

        Parameters
        ----------
        inputs : tensorflow.Tensor
            input of the dense layer to be added
        size : int
            number of units in the dense layer
        use_bias : bool, optional
            if true, the units use a bias term (default is True)
        activation : str, optional
            string denoting a activation function (default is None)
        weight_initializer : str, optional
            string denoting a weight initialization function (default is None)
        bias_initializer : str, optional
            string denoting a bias initialization function (default is None)
        """

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

    def add_conv_layer(self,inputs,num_filters,filter_size,strides=[1,1,1,1],activation=None,weight_initializer=None,bias_initializer=None):
        """
        Adds a convolutional layer to a neural network model.

        Special version of convolution for 3 dimensional inputs in NLP tasks.
        The input is seen as a batch of sentences, where each sentences is a sequence of word vectors.
        For each filter size k, windows of k consecutive word vectors are convolved.
        If no weight or bias initializer is passed, the initializers defined by self.DEFAULT_WEIGHT_INITIALIZER or self.DEFAULT_BIAS_INITIALIZER are used.

        Parameters
        ----------
        inputs : tensorflow.Tensor
            input of the convolutional layer to be added
        num_filters : int
            number of filters to use
        filter_sizes : List
            list of integers indicating the sizes of sliding windows
        strides : List
            list of integers indicating the strides for each window dimension
        activation : str, optional
            string denoting a activation function (default is None)
        use_max_pool : bool, optional
            if true, max pooling is applied after convolution (default is True)
        kernel_initializer : str, optional
            string denoting a weight initialization function (default is None)
        bias_initializer : str, optional
            string denoting a bias initialization function (default is None)
        """

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

    def add_activation(self,inputs,activation):
        """
        Applies the given activation function to the input tensor.

        Parameters
        ----------
        inputs: tensorflow.Tensor
            tensor to apply activation function to
        activation: str or None
            string denoting the name of the activation function
        """

        with self.graph.as_default():
                if activation is None:
                    return inputs
                elif activation not in QNetwork.activations:
                    print("Warning: unknown activation function {}".format(activation))
                    return inputs
                else:
                    return QNetwork.activations[activation](inputs)
        
    def add_variable(self, variable):
        """
        Adds a variable to this object's set of variables (self.variables).

        Parameters
        ----------
        variable : tensorflow.Tensor
            variable tensor
        """

        self.variables.append(variable)

    def add_variables(self,variables):
        """
        Adds a list of variables to this object's set of variables (self.variables).

        Parameters
        ----------
        variables : List of tensorflow.Tensor
            list of svariable tensors
        """

        self.variables.extend(variables)

    def get_tensor_shape(self,tensor):
        """
        Returns the shape of a tensor as a list.

        Parameters
        ----------
        tensor : tensorflow.Tensor
            tensor to return shape of
        """

        return tensor.get_shape().as_list()

    def get_name(self,base):
        """
        Returns a str obtained by concatenating the input base str with a unique integer index.

        Parameters
        ----------
        base : str
            base string name
        """

        name = base + "_" + str(self.next_name_id)
        self.next_name_id += 1
        return name

