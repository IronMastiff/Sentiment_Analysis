import utils

import numpy as np
import tensorflow as tf

def test( graph, cell, batch_size, accurecy ):
    test_acc = []
    with tf.Session( graph = graph ) as sess:
        saver.restore( sess, tf.train.latest_checkpoint( 'checkpoints' ) )
        test_state = sess.run( cell.zero_state( batch_size, tf.float32 ) )
        for i, ( x, y ) in enumerate( utils.get_batchs( test_x, test_y, batch_size ), 1 ):
            feed = {inputs_ : x,
                    labels_ : y[: None],
                    keep_prob : 1,
                    initial_state : test_state}
            batch_acc, test_state = sess.run( [accurecy, final_state], feed_dict = feed )
            test_acc.append( batch_acc )
        print( "Test accuracy: {:.3f}" . format( np.mean( test_acc ) ) )

