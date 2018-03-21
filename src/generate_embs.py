import argparse
from data_utils.utils import unicode_csv_reader2, parse_line, get_delimiter, datum_to_string, delete_files
from gensim.models import Word2Vec
import numpy as np
import pickle

EMB_WIDTH = 100
EMO_LEX = pickle.load(open('sentprop_lex_gnip.p', 'rb'))

def read_tweets(data_file):
    tweets = []
    delimiter = get_delimiter(data_file)
    with open(data_file, 'r') as fhr:
        reader = unicode_csv_reader2(fhr, encoding = 'utf8', delimiter = delimiter)
        for row in reader:
            parsed = parse_line(row, 'text', 'label', 'tweet_id', 'user_name', 'created_at',
                                      max_len=100, normalize=True, word_level=True)
            X_c = parsed[0]
            if X_c is not None:
                tweets.append(X_c)
    print tweets[:5]
    return tweets

def train_w2v(D):
    print 'Training word2vec on {} texts...'.format(len(D))
    model = Word2Vec(D, size=EMB_WIDTH, min_count=10, iter=5, workers=4)
    w2v = model.wv
    print 'Vocab size:', len(w2v.vocab)
    return w2v

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

def main(args):
    fnames = args['data_files']

    if args['embedding_type'] == 'word':
        tweets = []
        for fn in fnames:
            print 'Reading data from {}...'.format(fn)
            tweets += read_tweets(fn)

        tokens = set()
        for tweet in tweets:
            for tok in tweet:
                tokens.add(tok)
        print 'Number of unique tokens:', len(tokens)

        save_file = args['save_file']
        w2v = train_w2v(tweets)
        w2v.save_word2vec_format(save_file, binary=True)
        print 'Saved w2v keyed vectors at', save_file

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('-l', '--data_files', nargs = '+', type = str, default = None, help = 'filenames for raw texts')
    parser.add_argument('-et', '--embedding_type', type = str, default = None, help = 'type of embedding: word or user')
    parser.add_argument('-ch', '--cheat', type = bool, default = False, help = 'return real label proportions for user embeddings')
    parser.add_argument('-s', '--save_file', type = str, default = None, help = 'filename for saving w2v keyed vectors')
    parser.add_argument('-sp', '--sentprop', type = str, default = None, help = 'path to sentprop lexicon')

    args = vars(parser.parse_args())

    main(args)


