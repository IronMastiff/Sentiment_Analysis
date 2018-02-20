import tensorflow as tf
import numpy as np

import utils
import data_preprocessing
import tfgraph
import test

REVIEWS_DIR = './data/reviews.txt'
LABELS_DIR = './data/labels.txt'

LSTM_SIZE = 512
LSTM_LAYERS = 1
BATCH_SIZE = 128
LEARNING_RATE = 0.001
SEQ_LEN = 200

# Size of the embedding vectors ( number of units in the embedding layer )
EMBED_SIZE = 300
EPOCH = 10

# Creat the graph object
graph = tf.Graph()

with tf.Session( graph = graph ) as sess:

    reviews, labels, all_text, words = data_preprocessing.data_preprocession( REVIEWS_DIR,
                                                                              LABELS_DIR )

    reviews_ints, vocab_to_int = data_preprocessing.encoding_words( words, reviews )

    labels = data_preprocessing.encoding_labels( labels )

    reviews_ints, labels = data_preprocessing.non_zero(reviews_ints, labels)

    features = data_preprocessing.drop_dat( reviews_ints, SEQ_LEN )

    train_x, train_y, val_x, val_y, test_x, test_y = \
        data_preprocessing.classification_dataset( features, labels, split_frac = 8.0 )

    inputs_, labels_, keep_prob, n_words = tfgraph.create_graph( vocab_to_int, graph )

    embed = tfgraph.embedding( n_words, EMBED_SIZE, inputs_, graph )

    predictions, cost, optimizer, initial_state, final_state, cell = tfgraph.LSTM_cell( LSTM_SIZE,
                                                                  keep_prob,
                                                                  LSTM_LAYERS,
                                                                  BATCH_SIZE,
                                                                  embed, labels_,
                                                                  LEARNING_RATE, graph )

    accuracy = tfgraph.Validation( predictions, graph, labels_ )

    sess.run(tf.global_variables_initializer())

    with graph.as_default():
        saver = tf.train.Saver()

    iteration = 1
    for e in range( EPOCH ):
        state = sess.run( initial_state )

        for i, ( x, y ) in enumerate( utils.get_batches( train_x, train_y, BATCH_SIZE ), 1 ):
            feed = {inputs_ : x,
                    labels_ : y[:, None],
                    keep_prob : 0.5,
                    initial_state : state}
            loss, state, _ = sess.run( [cost, final_state, optimizer], feed_dict = feed )

            if iteration % 5 == 0:
                print( "Epoch: {} / {}" . format( e, EPOCH ),
                       "Iteration: {}" . format( iteration ),
                       "Train loss: {:.3f}" . format( loss ) )

            if iteration % 25 == 0:
                val_acc = []
                val_state = sess.run( cell.zero_state( BATCH_SIZE, tf.float32 ) )
                for x, y in utils.get_batches( val_x, val_y, batch_size = BATCH_SIZE ):
                    feed = {inputs_ : x,
                            labels_ : y[:, None],
                            keep_prob : 1,
                            initial_state : val_state}
                    batch_acc, val_state = sess.run( [accuracy, final_state], feed_dict = feed )
                    val_acc.append( batch_acc )
                print( "Val acc: {:.3f}" . format( np.mean( val_acc ) ) )
            iteration += 1
        saver.save( sess, 'checkpoints/sentment.ckpt' )

test.test( graph, cell, batch_size, accurecy )