from __future__ import print_function

from utils import *
from data import *
import copy
import numpy as np

class Batcher(object):

    def __init__(self, vocab_size, batch_size):
        self._vocab_size = vocab_size
        self._batch_size = batch_size

    def _generate_batch(self, data):
        dn = np.array(data)
        dn = np.array_split(dn, len(dn)/self._batch_size)
        return dn

    def generate_batch(self, data):
        batch = dict()
        d_keys = data.keys()
        d_size = len(data[d_keys[0]])
        cond = True
        while cond: #TODO fix dataset to avoid mismatching from word and tag
            d_index = np.random.randint(d_size, size=self._batch_size)
            for key in d_keys:
                batch[key] = np.array(data[key])[d_index]
            lw = map(lambda x: len(x), batch['word'])
            lt = map(lambda x: len(x), batch['tag'])
            cond = len([i for i in range(len(lw)) if lw[i]!=lt[i]])>0
        return batch

    def get_batch(self, train_data):

        def arr_dim(a):
            return 1 + arr_dim(a[0]) if (type(a) == list) else 0

        bv = self.generate_batch(train_data)
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
                                            self._vocab_size), bv_t_eos)
        bv_t_1hot = np.vstack([np.expand_dims(x, 0) for x in bv_t_1hot])
        data_padding(bv_t_go)
        bv_t_go = np.vstack([np.expand_dims(x, 0) for x in bv_t_go])

        return seq_len_w, seq_len_t, bv_w, bv_t, bv_t_go, bv_t_1hot
