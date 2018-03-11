import argparse
from data_utils.utils import unicode_csv_reader2, parse_line, get_delimiter, datum_to_string, delete_files
from gensim.models import Word2Vec

EMB_WIDTH = 300
MAX_VOCAB_SIZE = 20000

def read_data(data_file):
    tweets = []
    delimiter = get_delimiter(data_file)
    with open(data_file, 'r') as fhr:
        reader = unicode_csv_reader2(fhr, encoding = 'utf8', delimiter = delimiter)
        for row in reader:
            X_c, _, _, _ = parse_line(row, text_column="text", label_column="label",
                                      tweet_id_column="tweet_id", user_name_column="user_name",
                                      max_len=100, normalize=True, word_level=True)
            if X_c is not None:
                tweets.append(X_c)
    print tweets[:5]
    return tweets

def train_w2v(D):
    print "Training word2vec on {} texts...".format(len(D))
    model = Word2Vec(D, size=EMB_WIDTH, min_count=1)
    w2v = model.wv
    print "Vocab size:", len(w2v.vocab)
    return w2v

def main(args):
    data_file = args['data_file1']
    print "Reading data from {}...".format(data_file)
    tweets = read_data(data_file)

    tokens = set()
    for tweet in tweets:
        for tok in tweet:
            tokens.add(tok)
    print len(tokens)

    # data_file = args['data_file2']
    # if data_file is not None:
    #     print "Reading data from {}...".format(data_file)
    #     tweets.extend(read_data(data_file))
    #
    save_file = args['save_file'] + ".h5"
    w2v = train_w2v(tweets)
    w2v.save_word2vec_format(save_file, binary=True)
    print "Saved w2v keyed vectors at", save_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('-d1', '--data_file1', type = str, default = None, help = 'filename for raw texts')
    parser.add_argument('-d2', '--data_file2', type = str, default = None, help = 'filename for raw texts')
    parser.add_argument('-s', '--save_file', type = str, default = None, help = 'filename for saving w2v keyed vectors')

    args = vars(parser.parse_args())

    main(args)


