import tensorflow as tf
import os

os.chdir(os.path.dirname(__file__))


# Following Attention Mechanism will return the context vector and attention weights over the time axis.
class BahdanauAttention(tf.keras.Model):  # Global Attention

    def get_config(self):
        pass

    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)  # a Linear layer for the EfficientB7_Encoder outputs
        self.W2 = tf.keras.layers.Dense(units)  # a Linear layer for the the hidden state produced by the decoder in
        # the previous time step
        self.V = tf.keras.layers.Dense(1)  # the last layer (tanh layer) (aggregate the previous two layers) to obtain
        # alignment scores.

    def call(self, features, hidden):
        # features (EfficientB7_Encoder_output) shape == (batch_size, 324, embedding_dim)
        # Hidden size is number of features of the hidden state for RNN. Hidden dimension determines the feature vector
        # size of the hidden state.
        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        # Dimensions expansion process was done due to preserve mathematical conditions in the last layer (V)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # W1 output shape: (batch_size, 324, units)
        # W2 output shape: (batch_size, 1, units)
        # attention_hidden_layer shape == (batch_size, 324, units)
        attention_hidden_layer = (tf.nn.tanh(self.W1(features) +
                                             self.W2(hidden_with_time_axis)))

        # alignment scores shape == (batch_size, 324, 1)
        score = self.V(attention_hidden_layer)

        # Finding Probability using Softmax
        # attention_weights shape == (batch_size, 324, 1)
        attention_weights = tf.nn.softmax(score, axis=1)  # axis is the dimension that softmax would be performed on.

        # Giving weights to the different pixels in the image --> calculating of the context vector
        context_vector = attention_weights * features  # shape == (batch_size, 324, embedding_dim)
        context_vector = tf.reduce_sum(context_vector, axis=1)  # Computes the sum of elements across dimensions of a
        # tensor.
        # context_vector shape after reduce_sum == (batch_size, embedding_dim)

        return context_vector, attention_weights


class RNN_Decoder(tf.keras.Model):
    # The encoder output(i.e. 'features'), hidden state(initialized to 0) and the decoder input (i.e. 'x') is passed
    # to the decoder.
    def get_config(self):
        pass

    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units
        """
        * tf.keras.layers.Embedding     - Turns positive integers (indexes) into dense vectors of fixed size
        - input_dim	    Integer. Size of the vocabulary, i.e. maximum integer index + 1.
        - output_dim	Integer. Dimension of the dense embedding.
        """
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        """
        * tf.compat.v1.keras.layers.CuDNNLSTM -  Fast LSTM implementation backed by cuDNN.
        NVIDIA CUDA Deep Neural Network (cuDNN) is a GPU-accelerated library of primitives for deep neural networks. 
        It provides highly tuned implementations of routines arising frequently in DNN applications.
        
        - units 	Positive integer, dimensionality of the output space. 
                    units is the size of the LSTM's hidden state (which is also the size of the output if no 
                    projection is used) (it is also the size the LSTM's cell state).
                    
        - return_sequences	    Boolean. Whether to return the last output. in the output sequence, or the full sequence.
                                By default, the return_sequences is set to False in Keras RNN layers, and this means 
                                the RNN layer will only return the last hidden state output.
                                In other cases, we need the full sequence as the output. Setting return_sequences to 
                                True is necessary.
                                
        - return_state	        Boolean. Whether to return the last state in addition to the output.
                                for LSTM, hidden state and cell state are not the same.
                                In Keras we can output RNN's last cell state in addition to its hidden states by setting
                                return_state to True.
                                
        Generally, we do not need to access the cell state unless we are developing sophisticated models where 
        subsequent layers may need to have their cell state initialized with the final cell state of another layer, 
        such as in an encoder-decoder model.
        if both return_sequences and return_state are set to True, the output of the LSTM will have three components
          (a<1...T>, a<T>, c<T>)     each one has the shape (#Samples, #LSTM units).
          
        - recurrent_initializer 	Initializer for the recurrent_kernel weights matrix, used for the linear 
                                    transformation of the recurrent state.
        
        Note: Initializations define the way to set the initial random weights of Keras layers.
        -----
        Glorot normal initializer, also called Xavier normal initializer.
        It draws samples from a truncated normal distribution centered on 0 with stddev = sqrt(2 / (fan_in + fan_out)) 
        where fan_in is the number of input units in the weight tensor 
        and fan_out is the number of output units in the weight tensor. 
        """
        self.lstm = tf.compat.v1.keras.layers.CuDNNLSTM(units,
                                                        return_sequences=True,
                                                        return_state=True,
                                                        recurrent_initializer='glorot_uniform')

        self.fc1 = tf.keras.layers.Dense(self.units)  # the first FC layer to make a prediction of the lstm output - it
        # should have num of units equal to the hidden size of the lstm

        """
        * tf.keras.layers.Dropout 
        The Dropout layer randomly sets input units to 0 with a frequency of rate at each step during training time,
        which helps prevent overfitting.
        
        - rate	        Float between 0 and 1. Fraction of the input units to drop.
        
        - noise_shape	1D integer tensor representing the shape of the binary dropout mask that will be multiplied with
                        the input. For instance, if your inputs have shape (batch_size, timesteps, features) and you 
                        want the dropout mask to be the same for all timesteps, you can use:
                        noise_shape=(batch_size, 1, features).
                        
        - seed	        A Python integer to use as random seed.
        """
        self.dropout = tf.keras.layers.Dropout(0.5, noise_shape=None, seed=None)

        """
        * tf.keras.layers.BatchNormalization    - Layer that normalizes its inputs.
        Batch normalization applies a transformation that maintains the mean output close to 0 and the output standard 
        deviation close to 1.
        
        Importantly, batch normalization works differently during training and during inference:
        
        - During training :
        (i.e. when using fit() or when calling the layer/model with the argument training=True), the layer normalizes 
        its output using the mean and standard deviation of the current batch of inputs. That is to say, for each 
        channel being normalized, the layer returns (batch - mean(batch)) / (var(batch) + epsilon) * gamma + beta
        
        - During inference :
        (i.e. when using evaluate() or predict() or when calling the layer/model with the argument training=False 
        (which is the default), the layer normalizes its output using a moving average of the mean and standard 
        deviation of the batches it has seen during training. That is to say, it returns: 
        (batch - self.moving_mean) / (self.moving_var + epsilon) * gamma + beta.
        self.moving_mean and self.moving_var are non-trainable variables that are updated each time the layer in called 
        in training mode, as such:
            moving_mean = moving_mean * momentum + mean(batch) * (1 - momentum)
            moving_var = moving_var * momentum + var(batch) * (1 - momentum)
        It will be less accurate to use the mean and variance of the input batch during inference as likely the size is 
        much smaller than what you used during training, the law of large numbers is playing a role here.
            
        - axis	    Integer or a list of integers, the axis that should be normalized (typically the features axis).
        
        - momentum	Momentum for the moving average.
        - epsilon	Small float added to variance to avoid dividing by zero.
        - center	If True, add offset of beta to normalized tensor. If False, beta is ignored.
        - scale	    If True, multiply by gamma. If False, gamma is not used. 
        - beta_initializer	            Initializer for the beta weight.
        - gamma_initializer	            Initializer for the gamma weight.
        - moving_mean_initializer	    Initializer for the moving mean.
        - moving_variance_initializer   Initializer for the moving variance.
        
        * Regularizers allow you to apply penalties on layer parameters or layer activity during optimization. 
          These penalties are summed into the loss function that the network optimizes.
          
        - beta_regularizer	            Optional regularizer for the beta weight.
        - gamma_regularizer	            Optional regularizer for the gamma weight.
        
        * Classes from the tf.keras.constraints module allow setting constraints (eg. non-negativity) on model 
          parameters during training. They are per-variable projection functions applied to the target variable after 
          each gradient update (when using fit()).
          
        - beta_constraint	    Optional constraint for the beta weight.
        - gamma_constraint	    Optional constraint for the gamma weight.
        """
        self.batchNormalization = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True,
                                                                     scale=True, beta_initializer='zeros',
                                                                     gamma_initializer='ones',
                                                                     moving_mean_initializer='zeros',
                                                                     moving_variance_initializer='ones',
                                                                     beta_regularizer=None, gamma_regularizer=None,
                                                                     beta_constraint=None, gamma_constraint=None)

        self.fc2 = tf.keras.layers.Dense(vocab_size)  # the second FC layer to make a prediction of the lstm output - it
        # should have num of units equal to the vocab size.

        self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden):
        # defining attention as a separate model
        context_vector, attention_weights = self.attention(features, hidden)

        # x shape after passing through embedding == (batch_size, len(x), embedding_dim)
        # we will process one word at a time, so the shape will be (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # Attention outputs (i.e. context vector) must be concatenated with the embedding of the previous predicted word
        # to be fed into the lstm along with the previous hidden state
        """
        * tf.concat    - Concatenates tensors along one dimension.
        Negative axis are interpreted as counting from the end of the rank
        """
        # context vector shape: (batch_size, embedding_dim)
        # embedding layer shape: (batch_size, 1, embedding_dim)
        # we have to expand the dimensions of the context vector.
        # x shape after concatenation == (batch_size, 1, 2*embedding_dim)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the LSTM
        # The layer returns the hidden state for each input time step, then separately, the hidden state output for the
        # last time step and the cell state for the last input time step.
        """ An example of the output of an LSTM ,passing a sequence of 3 elements through an lstm
                    [array([[[-0.02145359],
                            [-0.0540871 ],
                            [-0.09228823]]], dtype=float32),
                     array([[-0.09228823]], dtype=float32),
                     array([[-0.19803026]], dtype=float32)]
            first list: hidden states for each time step,
            second list: (one value) the hidden state for the last time step,
            third list: (one value) the cell state for the last time step.
        Note: when we pass just one element through the lstm then the output will be:
            last hidden state, last hidden state, last cell state (first and second output are the same now)
        """
        output, last_hidden_state, last_cell_state = self.lstm(x)
        # output_shape == (batch_size, max_length, hidden_size), max_length is the maximum length of captions in our
        # dataset(len of the longest sequence), but since we are feeding one word at a time so the shape will be:
        # (batch_size, 1, hidden_size)

        # Time to pass the output of the lstm through two FC layers to make a prediction
        x = self.fc1(output)  # shape == (batch_size, max_length = 1, hidden_size); hidden_size = fc1_units

        x = tf.reshape(x, (-1, x.shape[2]))  # shape == (batch_size * max_length = 1, hidden_size)

        # Adding Dropout and BatchNorm Layers
        x = self.dropout(x)  # shape == (batch_size, hidden_size)
        x = self.batchNormalization(x)  # shape == (batch_size, hidden_size)

        x = self.fc2(x)  # shape == (batch_size, vocab_size)

        return x, last_hidden_state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))


"""  Model Complexity    - total trainable parameters: 10,450,450

rnn__decoder/embedding/embeddings:0 	             (10001, 256)
rnn__decoder/cu_dnnlstm/kernel:0 	                 (512, 2048)
rnn__decoder/cu_dnnlstm/recurrent_kernel:0 	         (512, 2048)
rnn__decoder/cu_dnnlstm/bias:0 	                     (4096,)
rnn__decoder/dense_1/kernel:0 	                     (512, 512)
rnn__decoder/dense_1/bias:0 	                     (512,)
rnn__decoder/batch_normalization/gamma:0 	         (512,)
rnn__decoder/batch_normalization/beta:0 	         (512,)
rnn__decoder/dense_2/kernel:0 	                     (512, 10001)
rnn__decoder/dense_2/bias:0 	                     (10001,)
rnn__decoder/bahdanau_attention/dense_3/kernel:0 	 (256, 512)
rnn__decoder/bahdanau_attention/dense_3/bias:0 	     (512,)
rnn__decoder/bahdanau_attention/dense_4/kernel:0 	 (512, 512)
rnn__decoder/bahdanau_attention/dense_4/bias:0 	     (512,)
rnn__decoder/bahdanau_attention/dense_5/kernel:0 	 (512, 1)
rnn__decoder/bahdanau_attention/dense_5/bias:0 	     (1,)

"""