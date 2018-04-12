import argparse
from data_utils.utils import unicode_csv_reader2, parse_line, get_delimiter, datum_to_string, delete_files
from gensim.models import KeyedVectors
import json
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression

def read_tuples(data_file):
    tuples = []
    delimiter = get_delimiter(data_file)
    with open(data_file, 'r') as fhr:
        reader = unicode_csv_reader2(fhr, encoding = 'utf8', delimiter = delimiter)
        for row in reader:
            X_c, y_c, _, _, _ = parse_line(row, 'text', 'label', 'tweet_id', 'user_name', 'created_at',
                                           max_len=100, normalize=True, word_level=True)
            if X_c is not None:
                text = ' '.join(X_c)
                tuples.append((text, y_c))
    # print tuples[:5]
    return tuples

def get_top_words(fnames, num_words, cutoff):
    tuples = []
    for fn in fnames:
        print 'Reading data from {}...'.format(fn)
        tuples += read_tuples(fn)

    loss = ''
    agg = ''
    other = ''
    all_tweets = []
    for tweet, label in tuples:
        all_tweets.append(tweet)
        if label == 'Loss':
            loss += tweet
        elif label == 'Aggression':
            agg += tweet
        else:
            other += tweet

    tfidf = TfidfVectorizer(lowercase=False)
    tfidf.fit(all_tweets)
    loss_arr, agg_arr, other_arr = tfidf.transform([loss, agg, other]).toarray()
    vocab = [tuple[0] for tuple in sorted(tfidf.vocabulary_.items(), key=lambda x: x[1])]
    print 'Length of Tfidf vocabulary:', len(vocab)
    assert(len(loss_arr) == len(vocab) and len(agg_arr) == len(vocab) and len(other_arr) == len(vocab))

    # ordered by indices of most important to least important tokens (by tfidf)
    loss_ordered = [tuple[0] for tuple in sorted(enumerate(loss_arr), key=lambda x: x[1], reverse=True)]
    agg_ordered = [tuple[0] for tuple in sorted(enumerate(agg_arr), key=lambda x: x[1], reverse=True)]
    other_ordered = [tuple[0] for tuple in sorted(enumerate(other_arr), key=lambda x: x[1], reverse=True)]

    idx = 0
    top_loss = []
    while idx < len(loss_ordered) and len(top_loss) < num_words:
        word_idx = loss_ordered[idx]
        if word_idx not in agg_ordered[:cutoff] and word_idx not in other_ordered[:cutoff]:
            top_loss.append(vocab[word_idx])
        idx += 1
    idx = 0
    top_agg = []
    while idx < len(agg_ordered) and len(top_agg) < num_words:
        word_idx = agg_ordered[idx]
        if word_idx not in loss_ordered[:cutoff] and word_idx not in other_ordered[:cutoff]:
            top_agg.append(vocab[word_idx])
        idx += 1
    idx = 0
    top_other = []
    while idx < len(other_ordered) and len(top_other) < num_words:
        word_idx = other_ordered[idx]
        if word_idx not in loss_ordered[:cutoff] and word_idx not in agg_ordered[:cutoff]:
            top_other.append(vocab[word_idx])
        idx += 1
    top_words = (top_loss, top_agg, top_other)
    return top_words

def load_w2v(wv_path):
    word2vec = KeyedVectors.load_word2vec_format(wv_path, binary = True)
    print('Found %s word vectors of word2vec' % len(word2vec.vocab))
    return word2vec

def eval_embeddings(wv_path, class_words):
    embs = load_w2v(wv_path)
    matrix = np.zeros((len(class_words), len(class_words)))
    for i,set_i in enumerate(class_words):
        for j,set_j in enumerate(class_words):
            sum_sim = 0
            count = 0
            for ti in set_i:
                for tj in set_j:
                    if ti in embs and tj in embs:
                        sum_sim += embs.similarity(ti, tj)
                        count += 1
            avg_sim = sum_sim/count
            matrix[i][j] = avg_sim
    return matrix

def write_embeddings(embs, ext): # embs is a dictionary of word to emb
    tokens = set()
    written_file = 'written_embs_' + ext + '.txt'
    lex_dump = 'lexicon_' + ext + '.p'
    with open(written_file, 'w') as f:
        for word in embs:
            tokens.add(word)
            result = word.encode('utf8') + '\t'
            emb = embs[word]
            result += ' '.join([str(x) for x in emb])
            result += '\n'
            f.write(result)
    print 'Saved', written_file
    pickle.dump(tokens, open(lex_dump, 'wb'))
    print 'Saved', lex_dump

def lex_embs_regression(lex, embs):
    print len(lex), len(embs)

    classes = ['loss', 'aggression', 'other']
    for idx,c in enumerate(classes):
        X = []
        y = []

        for word in embs:
            encoded = word.encode('utf8')
            if encoded in lex:
                emb = embs[word]
                X.append(emb)
                scores = lex[encoded]
                sent_score = scores[idx]/sum(scores)
                y.append([sent_score])

        reg = LinearRegression()
        reg.fit(X,y)
        print '{}: r-squared={}'.format(c, reg.score(X,y))

def qualitative(splex):
    loss_top, agg_top, other_top = pickle.load(open('seed_sets.p', 'rb'))
    print 'LOSS SEED SET'
    for word in loss_top:
        if word in splex:
            scores = splex[word]
            scores = [round(x/sum(scores),3) for x in scores]
            print '{}: {}'.format(word, scores)
    print 'AGG SEED SET'
    for word in agg_top:
        if word in splex:
            scores = splex[word]
            scores = [round(x/sum(scores),3) for x in scores]
            print '{}: {}'.format(word, scores)
    print 'OTHER SEED SET'
    for word in other_top:
        if word in splex:
            scores = splex[word]
            scores = [round(x/sum(scores),3) for x in scores]
            print '{}: {}'.format(word, scores)

def main(args):
    print 'TOP {} WORDS & UNIQUE FROM OTHER CLASSES\' TOP {}...'.format(args['num_words'], args['cutoff'])
    loss_top, agg_top, other_top = get_top_words(args['data_files'], args['num_words'], args['cutoff'])
    print 'LOSS:', loss_top
    print 'AGG:', agg_top
    print 'OTHER:', other_top
    seed_sets = (loss_top, agg_top, other_top)

    pickle.dump(seed_sets, open(args['save_file'], 'wb'))

    print 'Class to index:', {'Loss':0, 'Aggression':1, 'Other':2}
    print eval_embeddings(args['word2vec'], seed_sets)

    write_embeddings(args['word2vec'])

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description = '')
    # parser.add_argument('-l', '--data_files', nargs = '+', type = str, default = None, help = 'filenames for raw texts')
    # parser.add_argument('-n', '--num_words', type = int, default = 20, help = 'number of unique words')
    # parser.add_argument('-c', '--cutoff', type = int, default = 30, help = 'x, s.t. unique must be unique from classes\' top x')
    # parser.add_argument('-s', '--save_file', type = str, default = None, help = 'filename for saving seed sets')
    # parser.add_argument('-wv', '--word2vec', type = str, default = None, help = 'path to w2v keyed vectors')
    # parser.add_argument('-sp', '--sentprop', type = str, default = None, help = 'path to sentprop lexicon')
    #
    # args = vars(parser.parse_args())

    # wv_path = 'embs300_nov27unlabeled_2_w2v'
    # wv = load_w2v(wv_path)
    # embs = {}
    # for word in wv.vocab:
    #     embs[word] = wv[word]
    # write_embeddings(embs, 'w2v')
    svd_path = 'embs300_nov27unlabeled_svd'
    embs = json.load(open(svd_path, 'rb'))
    lex_path = 'splex_nov27unlabeled_svd.p'
    lex = pickle.load(open(lex_path, 'rb'))
    lex_embs_regression(lex, embs)
    # qualitative(lex)
    # embs = json.load(open('embs300_nov27unlabeled_svd', 'rb'))
    # write_embeddings(embs, 'nov27unlabeled_svd')