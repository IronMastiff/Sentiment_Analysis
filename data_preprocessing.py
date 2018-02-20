import utils
import numpy as np

from string import punctuation
from collections import Counter

REVIEWS_DIR = './data/reviews.txt'
LABLES_DIR = './data/labels.txt'

def data_preprocession( review_dir, labels_dir ):

    reviews, labels = utils.load_data( review_dir, labels_dir )

    all_text = ''.join( [c for c in reviews if c not in punctuation])    # 声明一个字符串， 如果数据集中字符串元素不为标点符号则加入新的字符串

    reviews = all_text.split( '\n' )    # 返回一个列表

    all_text = ''.join( reviews )    # 返回一个字符串

    words = all_text.split()    # 把字符串按空格划分成一个个单词

    return reviews, labels, all_text, words

def encoding_words( words, reviews ):
    counts = Counter( words )    # 返回元组，列表内容作为索引，列表内容个数作为内容
    vocab = sorted( counts, key = counts.get, reverse = True )    # sorted( , , reverse = True )把数组按从大到小顺序排列
    vocab_to_int = { word : i for i, word in enumerate( vocab, 1 ) }    # enumerate( vocab, start = 1 ) 把数组转换成索引序列,索引从1开始

    reviews_ints = []
    for each in reviews:
        reviews_ints.append( [vocab_to_int[word] for word in each.split()] )

    return reviews_ints, vocab_to_int


def encoding_labels( labels ):
    labels = labels.split( '\n' )
    labels = np.array( [1 if each == 'positive' else 0 for each in labels] )

    return labels

def non_zero( reviews_ints, labels ):
    non_zero_idx = [i for i, review in enumerate( reviews_ints ) if len( review ) != 0]
    reviews_ints = [reviews_ints[i] for i in non_zero_idx]
    labels = np.array( [labels[i] for i in non_zero_idx] )

    return reviews_ints, labels

def drop_dat( reviews_ints, seq_len ):
    features = np.zeros( ( len( reviews_ints ), seq_len ), dtype = int )
    for i, row in enumerate( reviews_ints ):
        features[i, -len( row ) :] = np.array( row )[: seq_len]

    return features

def classification_dataset(features, labels, split_frac=8.0):
    split_idx = int(len(features) * 0.8)
    train_x, preprocessing_val_x = features[: split_idx], features[split_idx:]
    train_y, preprocessing_val_y = labels[: split_idx], labels[split_idx:]

    test_idx = int(len(preprocessing_val_x) * 0.5)
    val_x, test_x = preprocessing_val_x[: test_idx], preprocessing_val_x[test_idx:]
    val_y, test_y = preprocessing_val_y[: test_idx], preprocessing_val_y[test_idx:]

    return train_x, train_y, val_x, val_y, test_x, test_y


if __name__ == '__main__':
    reviews, labels, all_text, words = data_preprocession( REVIEWS_DIR, LABLES_DIR )

    reviews_ints, vocab_to_int = encoding_words( words, reviews = reviews )

    reviews_lens = Counter( [len( x ) for x in reviews_ints] )

    print( "Zero-length reviews: {}" . format( reviews_lens[0] ) )
    print( "Maximun review length: {}" . format( max( reviews_lens ) ) )    # 返回元组中的索引最大值

    # review_lens = Counter( [len( x ) for x in review_ints] )
    # print( )