import argparse
from data_utils.utils import unicode_csv_reader2, parse_line, get_delimiter, datum_to_string, delete_files
from gensim.models import Word2Vec, KeyedVectors
import json
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer

EMO_LEX = pickle.load(open('sentprop_lex_gnip.p', 'rb'))

DEBUGGING = True

def read_tweets(data_file, join=False, char_level=True):
    tweets = []
    delimiter = get_delimiter(data_file)
    with open(data_file, 'r') as fhr:
        reader = unicode_csv_reader2(fhr, encoding = 'utf8', delimiter = delimiter)
        for row in reader:
            parsed = parse_line(row, 'text', 'label', 'tweet_id', 'user_name', 'created_at',
                                      max_len=100, normalize=True)
            tweet = parsed[0]
            if tweet is not None:
                if join or char_level:
                    tweet = ' '.join(tweet)

                if char_level:
                    chars = []
                    index = 0
                    while index < len(tweet):
                        if tweet[index:index + len('__URL__')] == '__URL__':
                            chars.append('__URL__')
                            index = index + len('__URL__')
                        elif tweet[index:index + len('__USER_HANDLE__')] == '__USER_HANDLE__':
                            chars.append('__USER_HANDLE__')
                            index = index + len('__USER_HANDLE__')
                        elif tweet[index:index + len('__RT__')] == '__RT__':
                            chars.append('__RT__')
                            index = index + len('__RT__')
                        else:
                            ch = tweet[index]
                            chars.append(ch)
                            index += 1
                    tweets.append(chars)
                else:
                    tweets.append(tweet)
    print tweets[:5]
    return tweets

def train_w2v(D):
    print 'Training word2vec on {} texts...'.format(len(D))
    model = Word2Vec(D, size=100, iter=20, workers=4)
    w2v = model.wv
    print 'Vocab size:', len(w2v.vocab)
    return w2v

def make_ppmi_embs(D, dim=300):
    mat, vocab = get_ppmi(D)
    print 'PPMI for word0, 0-20:', mat[0][:20]
    u,s,v = np.linalg.svd(mat)
    print 'Computed SVD'
    print 'Emb for word0, up to dim20:', u[0][:20]
    embs = {}
    for i, word in enumerate(vocab):
        ui = u[i]
        embs[word] = (ui[:dim]).tolist()
    print 'Embedding dim:', len(embs['you'])
    return embs

def get_ppmi(D):
    count_model = CountVectorizer(lowercase=True, max_features=20000)
    counts = count_model.fit_transform(D) # counts is (n,v) - need to keep sparse
    counts.data = np.fmin(np.ones(counts.data.shape), counts.data) # cap at 1
    n,v = counts.shape
    print 'n = {}, v = {}'.format(n,v)
    vocab = sorted(count_model.vocabulary_.items(), key=lambda x: x[1])
    vocab = [x[0] for x in vocab]
    print 'First 10 words in vocab:', vocab[:10]

    coo = (counts.T).dot(counts)
    coo.setdiag(1) # fill same word co-occurence to 1 so every word has at least one coo

    marginalized = coo.sum(axis=0) # num of coo per x
    prob_norm = coo.sum() # all coo
    print 'Prob_norm:', prob_norm
    row_mat = np.ones((v, v), dtype=np.float)
    for i in range(v):
        prob = marginalized[0,i] / prob_norm
        row_mat[i,:] = prob
    col_mat = row_mat.T
    joint = coo.toarray() / prob_norm

    P = joint / (row_mat * col_mat) # elementwise
    with np.errstate(divide='ignore'): # ignore 0
        P = np.fmax(np.zeros((v, v), dtype=np.float), np.log(P))
    print 'Computed PPMI:', P.shape
    return P, vocab

def check_emoji():
    emoji_file = 'emoji_embeddings_300.p'
    unicode_tokens, unicode_embs = pickle.load(open(emoji_file, "rb"))
    for tok in unicode_tokens:
        print tok.encode('utf8')

def update_user_dict(data_file, current_dict):
    delimiter = get_delimiter(data_file)
    with open(data_file, 'r') as fhr:
        reader = unicode_csv_reader2(fhr, encoding = 'utf8', delimiter = delimiter)
        for row in reader:
            X_c, y_c, _, user_name, time = parse_line(row, 'text', 'label', 'tweet_id', 'user_name', 'created_at',
                                         max_len=100, normalize=True, word_level=True)
            if X_c is not None:
                tweet_tuple = (X_c, time, y_c)
                if user_name in current_dict:
                    current_dict[user_name].append(tweet_tuple)
                else:
                    current_dict[user_name] = [tweet_tuple]
    return current_dict

def compute_aff_profiles(lex_path, u2t, cheat):
    print cheat
    lex = pickle.load(open(lex_path, "rb"))
    print len(lex)

    user2embs = {}
    global_counts = np.zeros(3)

    for user in u2t:
        l_counts = np.zeros(3)
        s_counts = np.zeros(3)
        for (tweet, time, label) in u2t[user]:
            if label == 'Loss':
                idx = 0
            elif label == 'Aggression':
                idx = 1
            else:
                idx = 2
            l_counts[idx] += 1

            for tok in tweet:
                if tok.encode('utf8') in lex:
                    idx = np.argmax(lex[tok.encode('utf8')])
                    s_counts[idx] += 1
                    global_counts[idx] += 1

        proportions = [x/sum(l_counts) for x in l_counts]
        if sum(s_counts) > 0:
            sent_profile = [x/sum(s_counts) for x in s_counts]
        else:
            sent_profile = 'global'

        if cheat:
            user2embs[user] = proportions
        else:
            user2embs[user] = sent_profile

    if not cheat:
        global_profile = [x/sum(global_counts) for x in global_counts]
        for user in user2embs:
            if user2embs[user] == 'global':
                user2embs[user] = global_profile

    return user2embs

def compute_aff_tweets(lex_path, u2t):
    lex = pickle.load(open(lex_path, "rb"))
    print len(lex)

    user2embs = {}
    for user, tuples in u2t.items():
        sent_scores = []
        sorted_tuples = sorted(tuples, key=lambda t: t[1])
        for (tweet, time, _) in sorted_tuples:
            max_l = 0
            max_a = 0
            max_o = 0
            for tok in tweet:
                if tok.encode('utf8') in lex:
                    scores = lex[tok.encode('utf8')]
                    l, a, o = [x/sum(scores) for x in scores]
                    max_l = max(max_l, l)
                    max_a = max(max_a, a)
                    max_o = max(max_o, o)
            sent_scores.append((time, [max_l, max_a, max_o]))
        user2embs[user] = sent_scores
    return user2embs

def check_rep(rep_name):
    saved = pickle.load(open(rep_name, 'rb'))
    rep1 = saved[3]
    print rep1.shape
    print np.count_nonzero(rep1)
    # rep2 = saved[4]
    # print rep2.shape
    # print np.count_nonzero(rep2)
    #for arr in rep:
        #print np.count_nonzero(arr)

def main(args):
    fnames = args['data_files']

    if args['embedding_type'] == 'w2v':
        tweets = []
        for fn in fnames:
            print 'Reading data from {}...'.format(fn)
            tweets += read_tweets(fn, join=False)

        tokens = set()
        for tweet in tweets:
            for tok in tweet:
                tokens.add(tok)
        print 'Number of unique tokens:', len(tokens)

        save_file = args['save_file']
        w2v_save = save_file + '_w2v'
        w2v = train_w2v(tweets)
        w2v.save_word2vec_format(w2v_save, binary=True)
        print 'Saved w2v keyed vectors at', w2v_save

    elif args['embedding_type'] == 'svd':
        tweets = []
        for fn in fnames:
            print 'Reading data from {}...'.format(fn)
            tweets += read_tweets(fn, join=True)

        tokens = set()
        for tweet in tweets:
            for tok in tweet.split():
                tokens.add(tok)
        print 'Number of unique tokens:', len(tokens)

        save_file = args['save_file']
        svd_save = save_file + '_svd'
        embs = make_ppmi_embs(tweets)
        json.dump(embs, open(svd_save, 'wb'))
        print 'Saved word embeddings at', svd_save

    elif args['embedding_type'] == 'user':
        user2tweets = {}
        for fn in fnames:
            user2tweets = update_user_dict(fn, user2tweets)

        # sort tweets per user by time created
        for user,tuples in user2tweets.items():
            sorted_tuples = sorted(tuples, key=lambda t: t[1])
            user2tweets[user] = sorted_tuples
        # print [x[1] for x in (user2tweets['younggodumb'][:5])], '\n'
        # print [x[1] for x in (user2tweets['younggodumb'][-5:])], '\n'

        save_file = args['save_file']
        user_embs = compute_aff_profiles(args['sentprop'], user2tweets, args['cheat'])
        print user_embs['younggodumb']
        print user_embs['tyquanassassin']
        pickle.dump(user_embs, open(save_file, 'wb'))
        print 'Saved user embeddings at', save_file

    elif args['embedding_type'] == 'tweet':
        user2tweets = {}
        for fn in fnames:
            user2tweets = update_user_dict(fn, user2tweets)

        save_file = args['save_file']
        user_embs = compute_aff_tweets(args['sentprop'], user2tweets)
        print user_embs['younggodumb'][:5]
        print user_embs['tyquanassassin'][:5]
        pickle.dump(user_embs, open(save_file, 'wb'))
        print 'Saved tweet aff embeddings at', save_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('-l', '--data_files', nargs = '+', type = str, default = None, help = 'filenames for raw texts')
    parser.add_argument('-et', '--embedding_type', type = str, default = None, help = 'type of embedding: word or user')
    parser.add_argument('-ch', '--cheat', type = bool, default = False, help = 'return real label proportions for user embeddings')
    parser.add_argument('-s', '--save_file', type = str, default = None, help = 'filename for saving w2v keyed vectors')
    parser.add_argument('-sp', '--sentprop', type = str, default = None, help = 'path to sentprop lexicon')

    args = vars(parser.parse_args())
    #
    # main(args)

    c2v = KeyedVectors.load_word2vec_format('charembs300_nov27unlabeled_w2v', binary = True)
    print 'Found %s word vectors of word2vec' % len(c2v.vocab)
    print c2v.vocab

    tweets = []
    for fn in args['data_files']:
        print 'Reading data from {}...'.format(fn)
        tweets += read_tweets(fn, join=False)
        tokens = set()

    tokens = set()
    for tweet in tweets:
        for tok in tweet:
            tokens.add(tok)
    print 'Number of unique tokens:', len(tokens)

    for tok in tokens:
        if tok not in c2v.vocab:
            print 'Missing:', tok

    # D = ['Hi it is me',
    #      'how is it going',
    #      'It going well',
    #      'You']
    # print D
    # embs = make_ppmi_embs(D)
    # check_rep('./saved/best_prediction_2018_04_09_11_44_39.p')
    # check_emoji()
