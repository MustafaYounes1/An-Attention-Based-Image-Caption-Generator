import tensorflow as tf


def nucleus_sampling(logits, p=0.9):
    sorted_logits = tf.sort(logits,
                            direction='DESCENDING')  # Sorts a tensor. [direction: ('ASCENDING' or 'DESCENDING')]
    sorted_indices = tf.argsort(logits, direction='DESCENDING')  # Returns the indices of a tensor that give its sorted
    # order along an axis. (axis: The default is -1, which sorts the last axis.)

    """
     * tf.math.reduce_sum(input_tensor, axis): Computes the sum of elements across dimensions of a tensor. (axis: The 
       dimensions to reduce. If None (the default), reduces all dimensions.) 
    
    *  tf.nn.softmax(logits, axis=None): Computes softmax activations. 
       [softmax = tf.exp(logits) / tf.math.reduce_sum(tf.exp(logits), axis)] axis: axis	The dimension softmax would be 
       performed on. The default is -1 which indicates the last dimension. 
     
    *  tf.math.cumsum(x, axis=0): Compute the cumulative sum of the tensor x along axis(default: 0). x = tf.constant([
       2, 4, 6, 8]) tf.math.cumsum(x) --> [2, 6, 12, 20] 
    """

    cumulative_probs = tf.math.cumsum(tf.nn.softmax(sorted_logits))
    t_sorted_indices_to_remove = cumulative_probs > p

    ''' Shift the indices to the right to keep also the first token (max prob) above the threshold (Sometimes, 
        it may be under the threshold)'''

    indices = tf.range(1, tf.shape(logits)[0], 1)
    """
     * tf.scatter_nd(indices, updates, shape, name=None) Scatter updates into a new tensor according to indices
       Creates a new tensor by applying sparse updates to individual values or slices within a tensor (initially zero 
       for numeric, empty for string, False for boolean) of the given shape according to indices.
       
                indices = tf.constant([[4], [3], [1], [7]])
                updates = tf.constant([9, 10, 11, 12])
                shape = tf.constant([8])
                scatter = tf.scatter_nd(indices, updates, shape)
                
                    --> tf.Tensor([ 0 11  0 10  9  0  0 12], shape=(8,), dtype=int32)
  """
    sorted_indices_to_remove = tf.scatter_nd(tf.expand_dims(indices, 1), t_sorted_indices_to_remove[:-1], logits.shape)
    """
     * tf.boolean_mask(tensor, mask, axis=None, name='boolean_mask') Apply boolean mask to tensor.

            tensor = [0, 1, 2, 3]  # 1-D example
            mask = np.array([True, False, True, False])
            tf.boolean_mask(tensor, mask)
            
                --> array([0, 2])
  """
    indices_to_remove = tf.boolean_mask(sorted_indices, sorted_indices_to_remove)
    t = tf.ones(tf.shape(indices_to_remove)[0], dtype=tf.bool)
    to_remove = tf.scatter_nd(tf.expand_dims(indices_to_remove, 1), t, logits.shape)
    """
     * tf.where(condition, x=None, y=None, name=None)
       Return the elements, either from x or y, depending on the condition. If element in condition is True, this 
       function will return element in x at the same position, otherwise, it will return element in y.

            condition = tf.Variable(np.array([[True, False, False],[False, True, False],[True, True, True]]), 
                                   dtype = tf.bool, name = 'condition')
            x = tf.Variable(np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]]), dtype = tf.float32, name = 'x')
            y = tf.Variable(np.array([[11, 12, 13],[14, 15, 16],[17, 18, 19]]), dtype = tf.float32, name = 'y')
            r = tf.where(condition, x, y)
            
                  --> [array([[ 1., 12., 13.],
                              [14.,  5., 16.],
                              [ 7.,  8.,  9.]], dtype=float32)]
  """
    logits = tf.where(
        to_remove,
        tf.ones_like(logits, dtype=logits.dtype) * -1e10,  # tf.ones_like(input, dtype=None, name=None) Creates a
        # tensor of all ones that has the same shape as the input.
        logits
    )

    # tf.random.categorical(logits, num_samples, dtype=None, seed=None, name=None) Draws samples from a categorical
    # distribution. (same as drawing from a multinomial distribution)
    sample = tf.random.categorical(tf.expand_dims(logits, 0), num_samples=1, dtype=tf.int32)  # output shape = (1, 1)

    return tf.reduce_sum(sample)  # output shape = ()


def top_k_sampling_with_temperature(logits, k=25, temperature=0.8):
    values, _ = tf.math.top_k(logits, k=k)  # Finds values and indices of the k largest entries for the last dimension.
    min_value = tf.math.reduce_min(values)  # Computes the tf.math.minimum of elements across dimensions of a tensor.
    # (axis: If None (the default), reduces all dimensions)
    logits = tf.where(
        logits < min_value,
        tf.ones_like(logits, dtype=logits.dtype) * -1e10,
        logits)

    logits = logits / temperature

    sample = tf.random.categorical(tf.expand_dims(logits, 0), num_samples=1, dtype=tf.int32)
    return tf.reduce_sum(sample)
