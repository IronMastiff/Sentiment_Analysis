import tensorflow as tf
import numpy as np
import tensorflow.contrib.rnn as rnn

def create_graph( vocab_to_int, graph ):

    n_words = len( vocab_to_int ) + 1 # Add 1 because we use 0"s for padding, dictionary started at 1


    # Add nodes to the graph
    with graph.as_default():
        inputs_ = tf.placeholder( tf.int32, [None, None], name = 'inputs' )
        labels_ = tf.placeholder( tf.int32, [None, None], name = 'labels' )
        keep_prob = tf.placeholder( tf.float32, name = 'keep_prob' )

    return inputs_, labels_, keep_prob, n_words

def embedding( n_words, embed_size, inputs_, graph ):
    with graph.as_default():
        embedding = tf.Variable( tf.random_uniform( ( n_words, embed_size ), -1, 1 ) )
        embed = tf.nn.embedding_lookup( embedding, inputs_ )    # tf.nn.embedding_lookup 选取tensor张量中的指定的元素

    return embed

def LSTM_cell( lstm_size, keep_prob, lstm_layers, batch_size, embed, labels_, learning_rate, graph ):
    with graph.as_default():
        # Your basic LSTM cell
        lstm = rnn.BasicLSTMCell( lstm_size )

        # Add dropout to the cell
        drop = rnn.DropoutWrapper( lstm, output_keep_prob = keep_prob )

        # Stack up multiple LSTM layers, for deep learning
        cell = rnn.MultiRNNCell( [drop] * lstm_layers )

        # Gretting an initial state of all zeros
        initial_state = cell.zero_state( batch_size, tf.float32 )

        outputs, final_state = tf.nn.dynamic_rnn( cell, embed, initial_state = initial_state )

        predictions = tf.contrib.layers.fully_connected( outputs[:, -1], 1, activation_fn = tf.sigmoid )

        cost = tf.losses.mean_squared_error( labels_, predictions )

        optimizer = tf.train.AdamOptimizer( learning_rate ).minimize( cost )

        return predictions, cost, optimizer, initial_state, final_state, cell

def Validation( predictions, graph, labels_ ):
    with graph.as_default():
        correct_pred = tf.equal( tf.cast( tf.round( predictions ), tf.int32 ), labels_ )    # tf.round 四舍五入， tf.cast 去除小数点后的数字
        accuracy = tf.reduce_mean( tf.cast( correct_pred, tf.float32 ) )

    return accuracy
