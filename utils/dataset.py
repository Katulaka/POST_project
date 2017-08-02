from __future__ import print_function

import os
import numpy as np
from itertools import chain

from vocab import Vocab
from gen_tags import gen_stags, gen_tags

def get_raw_data(data_path):

    print("Getting raw data from corpora")
    if not os.path.exists(data_path):
        try:
            os.makedirs(os.path.abspath(data_path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

        scp_path = ("scp -r login.eecs.berkeley.edu:" +
        "/project/eecs/nlp/corpora/EnglishTreebank/wsj/* ")
        os.system(scp_path + data_path)

def convert_data_flat(src_dir, words_out, tags_out, tag_type='stags'):
    """ If src dir is empty or not a file will result in empty file """

    gen_tags_fn = gen_stags if tag_type == 'stags' else gen_tags
    with open(tags_out, 'w') as t_file:
        with open(words_out, 'w') as w_file:
            for directory, _, filenames in os.walk(src_dir):
                for fname in filenames:
                    data_in = os.path.join(directory, fname)
                    print("Reading file %s" %(data_in))
                    for s_tags, s_words in gen_tags_fn(data_in):
                        print(s_tags, file=t_file)
                        print(s_words, file=w_file)

def textfile_to_vocab(fname, vocab_size=0, is_tag=False):

    with open(fname) as f:
        text = f.read().splitlines()
    tokens = map(lambda x: x.split(), text)
    tokens_flat_ = list(chain.from_iterable(tokens))
    if is_tag:
        tokens_flat = []
        for t in tokens_flat_:
            tokens_flat.extend(t.split('+'))
        return Vocab(tokens_flat, vocab_size), tokens

    return Vocab(tokens_flat_, vocab_size), tokens

def gen_dataset(w_file, t_file, w_vocab_size=0, t_vocab_size=0, max_len=10):

    w_vocab, words = textfile_to_vocab(w_file, w_vocab_size)
    t_vocab, tags = textfile_to_vocab(t_file, t_vocab_size, True)

    dataset = dict()
    if max_len > 0:
        indeces = np.where(map(lambda w: len(w) <= max_len, words))[0]
        words_ = np.array(words)[indeces]
        tags_ = np.array(tags)[indeces]
    else:
        words_ = words
        tags_ = tags

    dataset['word'] = w_vocab.to_ids(words_)
    dataset['tag'] = map(lambda x:
    t_vocab.to_ids(map(lambda y: y.split('+'), x)), tags_)
    return w_vocab, t_vocab, dataset
