from __future__ import print_function

from utils import *
from data import *
import copy
import numpy as np

# class Batcher(object):

def gen_sourcefile(text_w, text_t, max_len=10):

    indeces = [i for i in range(len(text_w)) if len(text_w[i])<=max_len]
    word_list = np.ndarray.tolist(np.array(text_w)[indeces])
    tag_list = np.ndarray.tolist(np.array(text_t)[indeces])

    with open('word_file', 'w') as f:
        f.write("\n".join([' '.join(s) for s in word_list]))

    with open('tag_file', 'w') as f:
        f.write("\n".join([' '.join(s) for s in tag_list]))

    return word_list, tag_list

def gen_dataset(w_file = 'words', t_file = 'tags', max_len=10, tvocab_size=-1, wvocab_size=-1):

    text = read_file(w_file)
    text_flat = list(flatten(text))
    word_vocab = Vocab(text_flat, wvocab_size)

    tags = read_file(t_file)
    tags_flat = list(flatten([tag.split('+') for tree in tags for tag in tree]))
    tag_vocab = Vocab(tags_flat, tvocab_size)

    vectors = dict()
    word_list, tag_list = gen_sourcefile(text, tags, max_len)
    vectors['word'] = lookup_fn(word_list, word_vocab)
    tag_list_mod = [[tag.split('+') for tag in tree] for tree in tag_list]
    vectors['tag'] = map(lambda tag_list: lookup_fn(tag_list, tag_vocab), tag_list_mod)

    return word_vocab, tag_vocab, vectors


def _generate_batch(data, batch_size=32):
    dn = np.array(data)
    dn = np.array_split(dn, len(dn)/batch_size)
    return dn

def generate_batch(data, batch_size=32):
    batch = dict()
    d_keys = data.keys()
    d_size = len(data[d_keys[0]])
    cond = True
    while cond: #TODO fix dataset to avoid mismatching from word and tag
        d_index = np.random.randint(d_size, size=batch_size)
        for key in d_keys:
            batch[key] = np.array(data[key])[d_index]
        lw = map(lambda x: len(x), batch['word'])
        lt = map(lambda x: len(x), batch['tag'])
        cond = len([i for i in range(len(lw)) if lw[i]!=lt[i]])>0
    return batch

def get_batch(train_data, tag_vocabulary_size, batch_size=32):

    def arr_dim(a): return 1 + arr_dim(a[0]) if (type(a) == list) else 0

    bv = generate_batch(train_data, batch_size)
    bv_w = copy.copy(bv['word'])
    bv_t = copy.copy(bv['tag'])
    if arr_dim(bv_t.tolist()) == 3:
        bv_t = [x for y in bv_t for x in y]

    seq_len_w = map(lambda x: len(x), bv_w)
    data_padding(bv_w)
    bv_w = np.vstack([np.expand_dims(x, 0) for x in bv_w])

    bv_t_go = add_go(bv_t)
    bv_t_eos = add_eos(bv_t)
    seq_len_t = map(lambda x: len(x), bv_t_go)
    bv_t_1hot = map(lambda x: to_onehot(x, max(seq_len_t),
                                        tag_vocabulary_size), bv_t_eos)
    bv_t_1hot = np.vstack([np.expand_dims(x, 0) for x in bv_t_1hot])
    data_padding(bv_t_go)
    bv_t_go = np.vstack([np.expand_dims(x, 0) for x in bv_t_go])

    return seq_len_w, seq_len_t, bv_w, bv_t, bv_t_go, bv_t_1hot
