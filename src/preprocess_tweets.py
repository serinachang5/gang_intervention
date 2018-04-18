import argparse
import codecs
from collections import defaultdict
import datetime
from gensim.models import KeyedVectors
import operator
import os
import cPickle as pickle
from data_utils.utils import unicode_csv_reader2, parse_line, get_delimiter, datum_to_string, delete_files
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def read_stop_chars(stop_chars_file):
    if stop_chars_file is None:
        return None
    stop_chars = []
    with codecs.open(stop_chars_file, 'r', encoding = 'utf-8') as fh:
        for line in fh:
            ch = line.split('\t')[0].strip()
            stop_chars.append(ch)
    return set(stop_chars)

class TweetPreprocessor:

    def __init__(self, stop_chars, time_stamp, seq_max_len, char_max_len, normalize, add_ss_markers):

        self.stop_chars = stop_chars
        self.time_stamp = time_stamp
        self.seq_max_len = seq_max_len
        self.char_max_len = char_max_len
        self.normalize = normalize
        self.add_ss_markers = add_ss_markers
        self.token2idx = {}
        self.char2idx = {}
        self.user2idx = {}
        # add start and stop markers
        if self.add_ss_markers:
            self.token2idx['<'] = 1
            self.token2idx['>'] = 2

        self.label2idx = {}
        self.len_dict = defaultdict(int)
        self.class2count = defaultdict(int)

    def store_tweets(self, data_files, parser, text_column, label_column, tweet_id_column, user_name_column, time_column, encoding):
        if data_files is None:
            return

        user2idx = {}
        user2tweets = {}
        for df in data_files:
            delimiter = get_delimiter(df)
            with open(df, 'r') as fhr:
                reader = unicode_csv_reader2(fhr, encoding, delimiter = delimiter)
                for row in reader:
                    X_c, y_c, tweet_id, user_name, time_created = parser(row, text_column, label_column, tweet_id_column, user_name_column, time_column, self.seq_max_len, stop_chars = self.stop_chars, normalize = self.normalize, add_ss_markers = self.add_ss_markers)
                    if X_c is not None:
                        if user_name in user2idx:
                            user2tweets[user_name].append((X_c, time_created))
                        else:
                            user2idx[user_name] = len(user2idx)
                            user2tweets[user_name] = [(X_c, time_created)]

        users = sorted(user2idx.items(), key=lambda x: x[1])
        self.user2first = {}
        self.tweets = [None]
        user_first = 1
        for user,_ in users:
            user_tweets = sorted(user2tweets[user], key=lambda x: x[1])
            self.user2first[user] = user_first
            user_first += len(user_tweets)
            self.tweets.extend(user_tweets)
        print self.tweets[:3]

    def get_window(self, user_name, time_created):
        window = []
        if user_name not in self.user2first:
            pass
        else:
            user_first = self.user2first[user_name] # id of user's first tweet
            window_end = user_first
            while window_end < len(self.tweets) and self.tweets[window_end][1] < time_created:
                window_end += 1
            # window_end is the id of the first tweet after time_created
            if window_end == user_first: # all tweets come after tweet in question
                pass
            else:
                window_start = window_end-1
                while window_start > user_first and (window_end - window_start) < 5: # up to the 5 most recent tweets
                    window_start -= 1
                # window start == user_first OR window is 5 tweets
                window = range(window_start, window_end)
        to_str = []
        for i in range(5):
            if i < len(window):
                to_str.append(str(window[i]))
            else:
                to_str.append('0')
        return to_str

    def read_data(self, output_files_dir, data_file, parser, text_column, label_column, tweet_id_column, user_name_column, time_column, encoding, is_train = False):

        if data_file is None:
            return

        _, fname = os.path.split(data_file)
        dot_index = fname.rindex('.')
        fname_wo_ext = fname[:dot_index]

        if not os.path.exists(os.path.join(output_files_dir, 'extra_files')):
            os.makedirs(os.path.join(output_files_dir, 'extra_files'))

        indices_file = os.path.join(output_files_dir, '_' + fname_wo_ext + '_' + self.time_stamp + '.txt')
        dropped_tweets_file = os.path.join(output_files_dir, 'extra_files', fname_wo_ext + '_' + self.time_stamp + '_dropped.txt')
        delimiter = get_delimiter(data_file)

        with open(data_file, 'r') as fhr, open(indices_file, 'w') as fhw1, codecs.open(dropped_tweets_file, 'w', encoding = encoding) as fhw2:
            reader = unicode_csv_reader2(fhr, encoding, delimiter = delimiter)
            for row in reader:
                X_c, y_c, tweet_id, user_name, time_created = parser(row, text_column, label_column, tweet_id_column, user_name_column, time_column, self.seq_max_len, stop_chars = self.stop_chars, normalize = self.normalize, add_ss_markers = self.add_ss_markers)
                if X_c is None:
                    fhw2.write(tweet_id)
                    fhw2.write('\n')
                    continue
                tok_ids = self.update_token2idx(X_c)
                if y_c is not None:
                    y_id = self.update_label2idx(y_c)
                else:
                    y_id = ''

                # update class dictionary
                if is_train and y_id != '':
                    self.class2count[int(y_id)] += 1

                char_ids = self.update_char2idx(' '.join(X_c), self.char_max_len)
                window_ids = self.get_window(user_name, time_created)

                fhw1.write(datum_to_string(tok_ids, char_ids, window_ids, y_id, tweet_id))
                fhw1.write('\n')

        return indices_file

    def update_token2idx(self, tweet):

        if tweet is None:
            return None

        idx = []

        self.len_dict[len(tweet)] += 1

        for token in tweet:
            if token not in self.token2idx:
                self.token2idx[token] = len(self.token2idx) + 1
            idx.append(str(self.token2idx[token]))

        return idx

    def update_char2idx(self, tweet, max_len):

        if tweet is None:
            return None

        idx = []

        index = 0
        while index < len(tweet) and len(idx) <= max_len:
            if tweet[index:index + len('__URL__')] == '__URL__':
                if '__URL__' not in self.char2idx:
                    self.char2idx['__URL__'] = len(self.char2idx) + 1
                idx.append(str(self.char2idx['__URL__']))
                index = index + len('__URL__')
            elif tweet[index:index + len('__USER_HANDLE__')] == '__USER_HANDLE__':
                if '__USER_HANDLE__' not in self.char2idx:
                    self.char2idx['__USER_HANDLE__'] = len(self.char2idx) + 1
                idx.append(str(self.char2idx['__USER_HANDLE__']))
                index = index + len('__USER_HANDLE__')
            elif tweet[index:index + len('__RT__')] == '__RT__':
                if '__RT__' not in self.char2idx:
                    self.char2idx['__RT__'] = len(self.char2idx) + 1
                idx.append(str(self.char2idx['__RT__']))
                index = index + len('__RT__')
            else:
                ch = tweet[index]
                if ch not in self.char2idx:
                    self.char2idx[ch] = len(self.char2idx) + 1
                idx.append(str(self.char2idx[ch]))
                index += 1

        return idx

    def update_user2idx(self, user_name):
        if user_name not in self.user2idx:
            self.user2idx[user_name] = len(self.user2idx)

        return str(self.user2idx[user_name])

    def update_label2idx(self, label):

        if label is None:
            return None

        if label not in self.label2idx:
            self.label2idx[label] = len(self.label2idx)

        return str(self.label2idx[label])

    def get_class_weights(self):

        self.class_weights = {}

        total = 0.0
        for cls in self.class2count.keys():
            total += (float(1) / self.class2count[cls])

        if total <= 0.0:
            return

        K = float(1) / total

        n_classes = len(self.label2idx)
        for i in xrange(n_classes):
            self.class_weights[i] = (K / self.class2count[i])

    def get_onehot_vectors(self):
        W = np.zeros((len(self.token2idx), len(self.token2idx)))
        for ii in xrange(len(W)):
            W[ii][ii] = 1
        return W

    def load_w2v(self, embedding_file):
        word2vec = KeyedVectors.load_word2vec_format(embedding_file, binary = True)
        print('Found %s word vectors of word2vec' % len(word2vec.vocab))
        return word2vec

    def get_dense_embeddings(self, w2v_file = None, emoji_file = None, splex_file = None, append_splex = False, emb_dim = 300):
        w2v = None
        if w2v_file is not None:
            w2v = self.load_w2v(w2v_file)

        unicode_tokens = None
        unicode_embs = None
        unicode_token2idx = None
        if emoji_file is not None:
            unicode_tokens, unicode_embs = pickle.load(open(emoji_file, "rb"))
            unicode_token2idx = {v:k for k, v in enumerate(unicode_tokens)}
        splex = None
        sp_dim = 0
        scalers = []
        if (splex_file is not None) and append_splex:
            splex = pickle.load(open(splex_file, 'rb'))
            scores = np.array(splex.values())
            sp_dim = scores.shape[1]
            by_class = scores.T
            for i in range(sp_dim):
                scaler = MinMaxScaler()
                scaler.fit(by_class[i].reshape(-1,1))
                scalers.append(scaler)

        W = np.zeros((len(self.token2idx), emb_dim+sp_dim))
        print 'Token W-shape:', W.shape

        w2v_miss_count = 0
        emoji_miss_count = 0
        splex_miss_count = 0
        for token in self.token2idx:
            if w2v is not None and token in w2v.vocab:
                emb = w2v.word_vec(token)
            elif unicode_tokens is not None and token in unicode_tokens:
                emb = unicode_embs[unicode_token2idx[token]]
                w2v_miss_count += 1
            else:
                w2v_miss_count += 1
                emoji_miss_count += 1
                emb = np.random.uniform(-0.25, 0.25, emb_dim)
            if (splex is not None) and append_splex:
                if token.lower() in splex:
                    scores = splex[token.lower()]
                    scaled_scores = np.zeros(sp_dim, dtype=np.float)
                    for i in range(sp_dim):
                        scaled_scores[i] = scalers[i].transform(scores[i].reshape(1,-1))[0]
                    emb = np.concatenate((emb, scaled_scores), axis = 0)
                else:
                    splex_miss_count += 1
                    emb = np.concatenate((emb, np.zeros(sp_dim)), axis = 0)
            else:
                splex_miss_count += 1
            W[self.token2idx[token]] = emb

        # just make sure the embedding corresponding to padding token is al zero
        W[self.token2idx['__PAD__']] = np.zeros(emb_dim+sp_dim, dtype = 'float32')

        print 'Number of tokens in vocabulary:', len(self.token2idx.keys())
        print 'Number of token embeddings found:', (len(self.token2idx.keys()) - w2v_miss_count)
        print 'Number of emoji embeddings found:', (w2v_miss_count - emoji_miss_count)
        print 'Number of splex scores appended:', (len(self.token2idx.keys()) - splex_miss_count)

        return W

    def get_char_embeddings(self, char_file, emb_dim = 100):
        c2v = self.load_w2v(char_file)

        W = np.zeros((len(self.char2idx)+1, emb_dim)) # plus one for PAD

        char_miss_count = 0
        for char in self.char2idx:
            if char in c2v.vocab:
                emb = c2v.word_vec(char)
                W[self.char2idx[char]] = emb
            else:
                print 'Missed char:', char
                char_miss_count += 1
                W[self.char2idx[char]] = np.random.uniform(-0.25, 0.25, emb_dim)

        W[self.token2idx['__PAD__']] = np.zeros(emb_dim, dtype = 'float32')

        print 'Number of characters in vocabulary:', len(self.char2idx.keys())
        print 'Number of character embeddings found:', (len(self.char2idx.keys()) - char_miss_count)

        return W

    def get_splex_scores(self, splex_file):
        splex = pickle.load(open(splex_file, 'rb'))
        scores = np.array(splex.values())
        sp_dim = scores.shape[1]
        by_class = scores.T
        scalers = []
        for i in range(sp_dim):
            scaler = MinMaxScaler()
            scaler.fit(by_class[i].reshape(-1,1))
            scalers.append(scaler)

        W = np.zeros((len(self.token2idx), sp_dim))

        splex_miss_count = 0
        for tok in self.token2idx:
            tok_lower = tok.lower()
            if tok_lower in splex:
                scores = splex[tok_lower]
                scaled_scores = []
                for i in range(sp_dim):
                    scaled_scores.append(scalers[i].transform(scores[i].reshape(1,-1))[0])
                W[self.token2idx[tok]] = scaled_scores
            else:
                splex_miss_count += 1
                W[self.token2idx[tok]] = np.zeros((sp_dim,))

        W[self.token2idx['__PAD__']] = np.zeros((sp_dim,))
        print 'Number of splex scores found:', (len(self.token2idx.keys()) - splex_miss_count)

        return W

    def get_tweet_tags(self, splex_file):
        splex = pickle.load(open(splex_file, 'rb'))
        scores = np.array(splex.values())
        sp_dim = scores.shape[1]
        by_class = scores.T
        scalers = []
        for i in range(sp_dim):
            scaler = MinMaxScaler()
            scaler.fit(by_class[i].reshape(-1,1))
            scalers.append(scaler)

        W = np.zeros((len(self.tweets), sp_dim))

        tweet_rep = 0
        for twe_i, tuple in enumerate(self.tweets):
            if tuple is None:
                W[twe_i] = np.zeros((sp_dim,))
                tweet_rep += 1
            else:
                tweet, date = tuple
                maxes = np.full(sp_dim, -1, dtype=np.float)
                for tok in tweet:
                    tok = tok.lower()
                    if tok in splex:
                        scores = splex[tok]
                        for i in range(sp_dim):
                            scaled_score = scalers[i].transform(scores[i].reshape(1,-1))[0]
                            maxes[i] = max(maxes[i], scaled_score)
                if maxes[0] < 0:
                    tweet_emb = np.zeros((sp_dim,))
                else:
                    tweet_rep += 1
                    tweet_emb = maxes
                W[twe_i] = tweet_emb

        print 'Number of tweets stored:', len(self.tweets)
        print 'Number of tweets represented:', tweet_rep

        return W

    def print_stats(self):
        print 'Number of classes: ', len(self.label2idx)
        print 'Length distribution: ', self.len_dict
        print 'Train set class distribution: ', self.class2count
        print 'Vocabulary:'
        _token_lst = sorted(self.token2idx.items(), key = operator.itemgetter(1))
        for w, idx in _token_lst:
            print w.encode('utf8'), idx, self.counts[w]

    def remap(self, labeled_files, unlabeled_files):

        if len(unlabeled_files + labeled_files) == 0:
            return

        _token2idx = {}
        _counts = {}
        _idx2c = defaultdict(int)
        for ifile in (labeled_files + unlabeled_files):
            with open(ifile, 'r') as fh:
                for line in fh:
                    x_y = line.strip().split('<:>')
                    idx_lst = x_y[0].split(',')
                    for idx in idx_lst:
                        _idx2c[int(idx)] += 1
                    # account for padding token
                    _idx2c[0] += max(0, self.seq_max_len - len(idx_lst))

        # replace rare words
        rare_wc = 0
        idx2c = {}
        for k, v in _idx2c.iteritems():
            if v > 2:
                idx2c[k] = v
            else:
                rare_wc += v

        # add the rare word as a token
        rare_word_id = len(self.token2idx) + 1
        idx2c[rare_word_id] = rare_wc

        s_idx2c = sorted(idx2c.items(), key = operator.itemgetter(1), reverse = True)

        # s_idx2c is a list of tuples
        # idx_map is a mapping from old index to new index
        idx_map = {}
        wc = 0
        for t in s_idx2c:
            idx_map[t[0]] = wc
            wc += 1

        # update token2idx based on new mapping
        for token in self.token2idx.keys():
            if self.token2idx[token] in idx_map:
                _token2idx[token] = idx_map[self.token2idx[token]]
                _counts[token] = idx2c[self.token2idx[token]]

        # save the below tokens in the map
        _token2idx['__UNK__'] = idx_map[rare_word_id]
        _token2idx['__PAD__'] = idx_map[0]
        _counts['__UNK__'] = rare_wc
        _counts['__PAD__'] = idx2c[0]

        self.token2idx = _token2idx
        self.counts = _counts
        _idx2token = {v:k for k, v in self.token2idx.iteritems()}

        tweets_dropped = defaultdict(int)
        for ifile in (labeled_files + unlabeled_files):
            # print 'Re-mapping file:', ifile
            fpath, fname = os.path.split(ifile)
            dot_index = fname.rindex('.')
            fname_wo_ext = fname[1:dot_index]
            with open(ifile, 'r') as fhr1, \
                open(os.path.join(fpath, fname_wo_ext + '.txt'), 'w') as fhw1, \
                open(os.path.join(fpath, 'extra_files', fname_wo_ext + '_pp.txt'), 'w') as fhw2:
                for line in fhr1:
                    x_y_tid = line.strip().split('<:>')
                    idx_lst = x_y_tid[0].split(',')
                    char_lst = x_y_tid[1].split(',')
                    window_lst = x_y_tid[2].split(',')
                    new_idx_lst = []
                    for idx in idx_lst:
                        idx = int(idx)
                        if idx in idx_map:
                            new_idx_lst.append(str(idx_map[idx]))
                        else:
                            new_idx_lst.append(str(idx_map[rare_word_id]))
                    if len(new_idx_lst) == 0:
                        tweets_dropped[fname] += 1
                        continue
                    else:
                        fhw2.write(','.join([x_y_tid[4], ' '.join([_idx2token[int(idx)].encode('utf8') for idx in new_idx_lst])]))
                        fhw2.write('\n')
                        fhw1.write(datum_to_string(new_idx_lst, char_lst, window_lst, x_y_tid[3], x_y_tid[4]))
                        fhw1.write('\n')

        # delete old files
        delete_files(labeled_files + unlabeled_files)

        print 'Tweet drop statistics:', tweets_dropped

def print_args(args):

    for k, v in args.iteritems():
        print k, v

def main(args):

    print_args(args)
    time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    stop_chars = read_stop_chars(args['stop_chars_file'])
    seq_max_len = 53
    char_max_len = 150

    unlabeled_files = []
    labeled_files = []
    tweet_preprocessor = TweetPreprocessor(stop_chars, time_stamp, seq_max_len = seq_max_len, char_max_len = char_max_len, normalize = args['normalize'], add_ss_markers = args['add_ss_markers'])
    if args['store_files'] is not None:
        print 'Storing data . . .'
        tweet_preprocessor.store_tweets(args['store_files'], parse_line, 'text', 'label', 'tweet_id', 'user_name', 'created_at', 'utf8')
    if args['store_files'] is None and args['train_file'] is not None:
        print 'Storing training data . . .'
        tweet_preprocessor.store_tweets([args['train_file']], parse_line, 'text', 'label', 'tweet_id', 'user_name', 'created_at', 'utf8')
    if args['train_file'] is not None:
        print 'Processing training set . . .'
        labeled_files.append(tweet_preprocessor.read_data(args['output_file_dir'], args['train_file'], parse_line, 'text', 'label', 'tweet_id', 'user_name', 'created_at', 'utf8', is_train = True))
    if args['val_file'] is not None:
        print 'Processing validation set . . .'
        labeled_files.append(tweet_preprocessor.read_data(args['output_file_dir'], args['val_file'], parse_line, 'text', 'label', 'tweet_id', 'user_name', 'created_at', 'utf8'))
    if args['test_file'] is not None:
        print 'Processing test set . . .'
        labeled_files.append(tweet_preprocessor.read_data(args['output_file_dir'], args['test_file'], parse_line, 'text', 'label', 'tweet_id', 'user_name', 'created_at', 'utf8'))
    if args['tweets_file_tr'] is not None:
        print 'Processing unlabeled train set . . .'
        unlabeled_files.append(tweet_preprocessor.read_data(args['output_file_dir'], args['tweets_file_tr'], parse_line, 'text', 'label', 'tweet_id', 'user_name', 'created_at', 'utf8'))
    if args['tweets_file_val'] is not None:
        print 'Processing unlabeled validation set . . .'
        unlabeled_files.append(tweet_preprocessor.read_data(args['output_file_dir'], args['tweets_file_val'], parse_line, 'text', 'label', 'tweet_id', 'user_name', 'created_at', 'utf8'))

    print 'Re-mapping token2idx. . .'
    tweet_preprocessor.remap(labeled_files, unlabeled_files)
    tweet_preprocessor.get_class_weights()

    # word-level
    if args['use_one_hot']:
        W = tweet_preprocessor.get_onehot_vectors()
    else:
        W = tweet_preprocessor.get_dense_embeddings(args['w2v_file'], args['emoji_file'], args['splex_file'], args['append_splex'], args['emb_dim'])
    # pickle.dump([W, tweet_preprocessor.token2idx, tweet_preprocessor.label2idx, tweet_preprocessor.counts, tweet_preprocessor.class_weights, tweet_preprocessor.max_len], open(os.path.join(args['output_file_dir'], 'dictionaries_' + time_stamp + '.p'), 'wb'))

    # char-level
    if args['char_file'] is not None:
        W = tweet_preprocessor.get_char_embeddings(args['char_file'])
        # pickle.dump(W, open(os.path.join(args['output_file_dir'], 'char_embeddings_' + time_stamp + '.p'), 'wb'))

    # splex scores
    if args['splex_file'] is not None:
        W = tweet_preprocessor.get_splex_scores(args['splex_file'])
        # pickle.dump(W, open(os.path.join(args['output_file_dir'], 'splex_scores_' + time_stamp + '.p'), 'wb'))
        W = tweet_preprocessor.get_tweet_tags(args['splex_file'])
        # pickle.dump(W, open(os.path.join(args['output_file_dir'], 'tweet_tags_' + time_stamp + '.p'), 'wb'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('-st', '--store_files', nargs = '+', type = str, default = None, help = 'set of tweets to store for emo profiles')
    parser.add_argument('-tr', '--train_file', type = str, default = None, help = 'labeled train set')
    parser.add_argument('-val', '--val_file', type = str, default = None, help = 'labeled validation set')
    parser.add_argument('-tst', '--test_file', type = str, default = None, help = 'labeled test set')
    parser.add_argument('-ofd', '--output_file_dir', type = str, default = None, help = 'directory where output files should be saved')

    parser.add_argument('-sch', '--stop_chars_file', type = str, default = None, help = 'file containing stop characters/words')
    parser.add_argument('-1h', '--use_one_hot', type = bool, default = False, help = 'If True, one hot vectors will be used instead of dense embeddings')

    parser.add_argument('-wfile', '--w2v_file', type = str, default = None, help = 'file containing pre-trained word2vec embeddings')
    parser.add_argument('-efile', '--emoji_file', type = str, default = None, help = 'file containing pre-trained emoji embeddings')
    parser.add_argument('-cfile', '--char_file', type = str, default = None, help = 'file containing pre-trained character embeddings')
    parser.add_argument('-lfile', '--splex_file', type = str, default = None, help = 'file containing pre-trained splex scores')
    parser.add_argument('-app', '--append_splex', type = bool, default = False, help = 'whether to append splex scores to word embeddings; ignored if splex is not provided')

    parser.add_argument('-unld_tr', '--tweets_file_tr', type = str, default = None, help = 'unlabeled tweets file to be used for training language model')
    parser.add_argument('-unld_val', '--tweets_file_val', type = str, default = None, help = 'unlabeled tweets file to be used for validating language model')
    parser.add_argument('-nor', '--normalize', type = bool, default = True, help = 'Normalize i.e. preprocess tweets')
    parser.add_argument('-amrks', '--add_ss_markers', type = bool, default = False, help = 'If True, start and stop markers will be added to the tweets')
    parser.add_argument('-edim', '--emb_dim', type = int, default = 300, help = 'embedding dimension')
    args = vars(parser.parse_args())

    main(args)
