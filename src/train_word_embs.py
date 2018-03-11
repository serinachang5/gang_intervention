import argparse
from data_utils.utils import unicode_csv_reader2, parse_line, get_delimiter, datum_to_string, delete_files
from gensim.models import Word2Vec

EMB_WIDTH = 300

def read_data(data_file):
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
    model = Word2Vec(D, size=EMB_WIDTH, min_count=5, iter=5, workers=4)
    w2v = model.wv
    print 'Vocab size:', len(w2v.vocab)
    return w2v

def main(args):
    fnames = args['data_files']
    tweets = []
    for fn in fnames:
        print 'Reading data from {}...'.format(fn)
        tweets += read_data(fn)

    tokens = set()
    for tweet in tweets:
        for tok in tweet:
            tokens.add(tok)
    print 'Number of unique tokens:', len(tokens)

    save_file = args['save_file'] + '.h5'
    w2v = train_w2v(tweets)
    w2v.save_word2vec_format(save_file, binary=True)
    print 'Saved w2v keyed vectors at', save_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('-l', '--data_files', nargs = '+', type = str, default = None, help = 'filenames for raw texts')
    parser.add_argument('-s', '--save_file', type = str, default = None, help = 'filename for saving w2v keyed vectors')

    args = vars(parser.parse_args())

    main(args)


