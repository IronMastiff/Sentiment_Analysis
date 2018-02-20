import numpy as np
import tensorflow as tf

def load_data( review_dir, labels_dir ):
    with open( review_dir, 'r' ) as f:
        reviews = f.read()

    with open( labels_dir, 'r' ) as f:
        labels = f.read()

    return reviews, labels

def get_batches( x, y, batch_size = 100 ):

    n_batches = len( x ) // batch_size    # //整除
    x, y = x[: n_batches * batch_size], y[: n_batches * batch_size]
    for i in range( 0, len( x ), batch_size ):
        yield x[i : i + batch_size], y[i : i + batch_size]

if __name__ == "__main__":
    review_dir = './data/reviews.txt'
    lablels_dir = './data/labels.txt'

    reviews, labels = load_data( review_dir, lablels_dir )

    print( reviews[: 2000] )
    print( labels[: 2000] )

