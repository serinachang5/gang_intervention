from keras.utils import np_utils
from sklearn.model_selection import StratifiedKFold
import subprocess

import cPickle as pickle
import numpy as np

def flatten(X, context_size):
    if X == None:
        return None, None
    X_tcs = []
    y_tcs = []
    for X_c in X:
        # pad atleast context_size - 1 zeros in the beginning
        # X_c = X_c.tolist()
        X_c = [0 for _ in xrange(context_size - 1)] + X_c
        s = 0
        while s + context_size < len(X_c):
            X_tcs.append(X_c[s:s + context_size])
            y_tcs.append(X_c[s + context_size])
            s += 1

    X_tcs = np.asarray(X_tcs)
    y_tcs = np.asarray(y_tcs)
    '''
    https://github.com/fchollet/keras/issues/483
    The trick to fix issue with the error expecting a 3D input when using
    sparse_categorical_crossentropy is to format outputs in a sparse 3-dimensional way.
    So instead of formatting the output like this:
    y_indices_naive = [1,5,4300,...]
    is should be formatted this way:
    y_indices_naive = [[1,], [5,] , [4300,],...]
    That will make Keras happy and it'll trained the model as expected.
    '''
    y_tcs = np.expand_dims(y_tcs, 1)
    return X_tcs, y_tcs

def add_pad_token(X, pad_token_idx, max_len):
    _X = [np.asarray([pad_token_idx for _ in xrange(max_len - len(indices))] + indices) for indices in X]
    return np.asarray(_X)

def parse_line(line, mode):
    x_y = line.split('<:>')
    seq = [int(ch) for ch in x_y[0].split(',')]
    chars = [int(ch) for ch in x_y[1].split(',')]
    window = [int(ch) for ch in x_y[2].split(',')]
    y = None
    if mode == 'clf':
        y = [int(x_y[3])]
    elif mode == 'seq2seq':
        y = seq
    return seq, chars, window, y

def get_line_count(fname):
    p = subprocess.Popen(['wc', '-l', fname], stdout = subprocess.PIPE,
                                              stderr = subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])

class Corpus:
    # in-memory data
    def __init__(self, data_file, mode):
        self.X_seq = []
        self.X_char = []
        self.X_window = []
        self.y = []
        self.read_data(data_file, mode)
        print 'Number of lines in %s: %d' % (data_file, len(self.X_seq))

    def read_data(self, data_file, mode):

        with open(data_file, 'r') as fh:
            for line in fh:
                seq, chars, window, y = parse_line(line, mode)
                self.X_seq.append(seq)
                self.X_char.append(chars)
                self.X_window.append(window)
                self.y.append(y)

        self.X_seq = np.asarray(self.X_seq)
        self.X_char = np.asarray(self.X_char)
        self.X_window = np.asarray(self.X_window)
        self.y = np.asarray(self.y)

class Generator:
    # streaming data
    def __init__(self, data_file, mode, max_len, nclasses, pad_token_idx):
        self.data_file = data_file
        self.mode = mode
        self.max_len = max_len
        self.nclasses = nclasses
        self.pad_token_idx = pad_token_idx
        self.wc = get_line_count(self.data_file)
        print 'Number of lines in %s: %d' % (self.data_file, self.wc)

    def get_mini_batch(self, batch_size):

        # mini-batch generator
        X = []
        y = []
        while True:
            done = False
            with open(self.data_file) as fh:
                for line in fh:
                    X_c, _, y_c = parse_line(line, self.mode, self.max_len, self.nclasses, self.pad_token_idx)
                    X.append(X_c)
                    y.append(y_c)
                    if len(X) == batch_size:
                        done = True
                        X = np.asarray(X)
                        y = np.asarray(y)
                        yield (X, y)
                    if done:
                        done = False
                        X = []
                        y = []

    def get_example(self):
        # mini-batch generator
        while True:
            with open(self.data_file) as fh:
                for line in fh:
                    X_c, _, y_c = parse_line(line, self.mode, self.max_len, self.nclasses, self.pad_token_idx)
                    yield (np.asarray([X_c]), np.asarray([y_c]))

class TweetCorpus:

    def __init__(self, train_file = None, val_file = None, test_file = None, unld_train_file = None, unld_val_file = None, dictionaries_file = None, char_file = None, splex_scores_file = None, tweet_tags_file = None):

        self.W, self.token2idx, self.label2idx, self.counts, self.class_weights, self.seq_max_len = pickle.load(open(dictionaries_file, 'rb'))
        self.W_c, self.char2idx, self.char_max_len = pickle.load(open(char_file, 'rb'))
        self.W_s = pickle.load(open(splex_scores_file, 'rb'))
        self.W_t = pickle.load(open(tweet_tags_file, 'rb'))

        self.pad_token_idx = self.token2idx['__PAD__']
        self.idx2token = {v:k for k, v in self.token2idx.iteritems()}
        self.pad_char_idx = self.char2idx['__PAD__']

        if train_file is None:
            self.tr_data = None
        else:
            self.tr_data = Corpus(train_file, 'clf')

        if val_file is None:
            self.val_data = None
        else:
            self.val_data = Corpus(val_file, 'clf')

        if test_file is None:
            self.te_data = None
        else:
            self.te_data = Corpus(test_file, 'clf')

        if unld_train_file is None:
            self.unld_tr_data = None
        else:
            self.unld_tr_data = Corpus(unld_train_file, 'lm')

        if unld_val_file is None:
            self.unld_val_data = None
        else:
            self.unld_val_data = Corpus(unld_val_file, 'lm')

    def get_data_for_classification(self):
        X_tr = None
        y_tr = None
        if self.tr_data is not None:
            X_tr_seq = add_pad_token(self.tr_data.X_seq, self.pad_token_idx, self.seq_max_len)
            X_tr_char = add_pad_token(self.tr_data.X_char, self.pad_char_idx, self.char_max_len)
            for x in X_tr_char:
                if len(x) != self.char_max_len:
                    print x, len(x)
            X_tr_window = self.tr_data.X_window
            X_tr = [X_tr_seq, X_tr_char, X_tr_window]
            y_tr = np_utils.to_categorical(self.tr_data.y, len(self.label2idx))

        X_val = None
        y_val = None
        if self.val_data is not None:
            X_val_seq = add_pad_token(self.val_data.X_seq, self.pad_token_idx, self.seq_max_len)
            X_val_char = add_pad_token(self.val_data.X_char, self.pad_char_idx, self.char_max_len)
            X_val_window = self.val_data.X_window
            X_val = [X_val_seq, X_val_char, X_val_window]
            y_val = np_utils.to_categorical(self.val_data.y, len(self.label2idx))

        X_te = None
        y_te = None
        if self.te_data is not None:
            X_te_seq = add_pad_token(self.te_data.X_seq, self.pad_token_idx, self.seq_max_len)
            X_te_char = add_pad_token(self.te_data.X_char, self.pad_char_idx, self.char_max_len)
            X_te_window = self.te_data.X_window
            X_te = [X_te_seq, X_te_char, X_te_window]
            y_te = np_utils.to_categorical(self.te_data.y, len(self.label2idx))

        return X_tr, X_val, X_te, y_tr, y_val, y_te

    def get_data_for_cross_validation(self, folds = 3):
        X_tr, X_val, _, y_tr, y_val, _ = self.get_data_for_classification()
        X_tr = np.concatenate((X_tr[0], X_tr[1], X_tr[2]), axis = 1)
        X_val = np.concatenate((X_val[0], X_val[1], X_val[2]), axis = 1)
        # combine X_train, X_val and use the combined dataset for cross validation
        X_train = np.concatenate((X_tr, X_val), axis = 0)
        y_train = np.concatenate((y_tr, y_val), axis = 0)
        skf = StratifiedKFold(shuffle = True, n_splits = folds)
        for train_index, test_index in skf.split(X_train, np.argmax(y_train, axis = 1)):
            this_X_train = [X_train[train_index, : self.seq_max_len],
                            X_train[train_index, self.seq_max_len : self.seq_max_len+self.char_max_len,
                            X_train[train_index, self.seq_max_len+self.char_max_len :]]]
            print len(this_X_train[0])
            this_X_test = [X_train[test_index, : self.seq_max_len],
                           X_train[test_index, self.seq_max_len : self.seq_max_len+self.char_max_len,
                           X_train[test_index, self.seq_max_len+self.char_max_len : ]]]
            yield this_X_train, this_X_test, y_train[train_index], y_train[test_index]

    def get_data_for_lm(self, truncate = False, context_size = 10):
        if truncate:
            # After truncation, X will have shape [-1, context_size] and y will have shape [-1,1]
            X_tr, y_tr = flatten(self.unld_tr_data.X_seq, context_size)
            X_val, y_val = flatten(self.unld_val_data.X_seq, context_size)
            return X_tr, X_val, y_tr, y_val
        else:
            _X = add_pad_token(self.unld_tr_data.X_seq, self.pad_token_idx, self.max_len)
            X_tr = _X[:, :-1]
            y_tr = _X[:, 1:]
            y_tr = np.expand_dims(y_tr, 2)
            _X = add_pad_token(self.unld_val_data.X_seq, self.pad_token_idx, self.max_len)
            X_val = _X[:, :-1]
            y_val = _X[:, 1:]
            y_val = np.expand_dims(y_val, 2)
            return X_tr, X_val, y_tr, y_val

    def get_data_for_seq2seq(self):
        pass

    def get_class_names(self):

        class_names = []
        idx2label = {v:k for k, v in self.label2idx.iteritems()}
        for idx in xrange(len(idx2label)):
            class_names.append(idx2label[idx])

        return class_names
