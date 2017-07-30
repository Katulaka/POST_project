from __future__ import print_function

from data import *
import copy
import numpy as np

class Batcher(object):

    def __init__(self, data, vocab_size, batch_size):
        self._vocab_size = vocab_size
        self._batch_size = batch_size
        self._data = data

    def batch_valid(self, b_word, b_tag):
        return all(map(lambda x, y: len(x)==len(y), b_word, b_tag))

    def generate_batch(self):
        d_size = len(self._data['word'])
        d_index = np.random.randint(d_size, size=self._batch_size)
        b_word = np.array(self._data['word'])[d_index]
        b_tag = np.array(self._data['tag'])[d_index]
        return b_word, b_tag

    def get_batch(self):
        batch = dict()
        b_valid = False
        while not b_valid:
            batch['word'], batch['tag'] = self.generate_batch()
            b_valid = self.batch_valid(batch['word'], batch['tag'])
        return batch

    def generate_in_data(self, batch_data):
        seq_len = map(lambda x: len(x), batch_data)
        data_padding(batch_data)
        batch_data = np.vstack([np.expand_dims(x, 0) for x in batch_data])
        return seq_len, batch_data

    def generate_targets(self, max_len, batch_data):
        batch_1hot = map(lambda x: to_onehot(x, max_len, self._vocab_size),
                            batch_data)
        return np.vstack([np.expand_dims(x, 0) for x in batch_1hot])

    def next_batch(self):

        def arr_dim(a):
            return 1 + arr_dim(a[0]) if (type(a) == list) else 0

        bv = self.get_batch()

        bv_w = copy.copy(bv['word'])
        seq_len_w, bv_w = self.generate_in_data(bv_w)

        bv_t = copy.copy(bv['tag'])
        if arr_dim(bv_t.tolist()) == 3:
            bv_t = [x for y in bv_t for x in y]
        seq_len_t, bv_t_go = self.generate_in_data(add_go(bv_t))
        bv_t_1hot = self.generate_targets(max(seq_len_t), add_eos(bv_t))

        return seq_len_w, seq_len_t, bv_w, bv_t, bv_t_go, bv_t_1hot

    

        # def _generate_batch(self):
        #     dn = np.array(self._data)
        #     dn = np.array_split(dn, len(dn)/self._batch_size)
        #     return dn
