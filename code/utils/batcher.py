from __future__ import print_function

import copy
import numpy as np
from itertools import islice

from vocab import pad, add_go, add_eos, to_onehot


class Batcher(object):

    def __init__(self, data, vocab_size, batch_size, add_delim, split_tag_pos):
        self._vocab_size = vocab_size
        self._batch_size = batch_size
        self._data = data
        self._add_delim = add_delim
        self._split_tag_pos = split_tag_pos

    def get_batch_size(self):
        return self._batch_size

    def get_random_batch(self):
        batch = dict()
        d_size = len(self._data['words'])
        d_index = np.random.randint(d_size, size=self._batch_size)
        batch['words'] = np.array(self._data['words'])[d_index]
        batch['tags'] = np.array(self._data['tags'])[d_index]
        return batch

    def seq_len(self, batch_data):
        return map(lambda x: len(x), batch_data)

    def generate_in_data(self, batch_data):
        pad(batch_data)
        batch_data = np.vstack([np.expand_dims(x, 0) for x in batch_data])
        return batch_data

    def generate_targets(self, max_len, batch_data):
        batch_1hot = map(lambda x: to_onehot(x, max_len, self._vocab_size),
                            batch_data)
        return np.vstack([np.expand_dims(x, 0) for x in batch_1hot])

    def process_word_batch(self, bv_words):

        bv_w = copy.copy(bv_words)
        self._seq_len = self.seq_len(bv_w)
        bv_w = add_eos(add_go(bv_w)) if self._add_delim else bv_w
        seq_len_w = self.seq_len(bv_w)
        bv_w_pad = self.generate_in_data(bv_w)

        return seq_len_w, bv_w_pad

    def process_pos_batch(self, bv_tags):
        def arr_dim(arr):
            return 1 + arr_dim(arr[0]) if (type(arr) == list) else 0

        def flatten(arr):
            return [x for y in arr for x in y]

        bv_t = copy.copy(bv_tags)
        bv_pos = []
        if arr_dim(bv_t.tolist()) == 3:
            if self._split_tag_pos:
                bv_pos = [ [_b[-1] for _b in b] for b in bv_t]
                bv_pos = add_eos(add_go(bv_pos)) if self._add_delim else bv_pos
                bv_pos_pad = self.generate_in_data(bv_pos)
            bv_t = flatten(bv_t)
        return bv_t, bv_pos

    def process_tag_batch(self, bv_tags):

        bv_t_go = add_go(bv_tags)
        seq_len_t = self.seq_len(bv_t_go)
        bv_t_pad = self.generate_in_data(bv_t_go)
        bv_t_1hot = self.generate_targets(max(seq_len_t), add_eos(bv_tags))

        return seq_len_t, bv_t_pad, bv_t_1hot

    def process_batch(self, bv):
        # bv = self.get_random_batch()
        seq_len_w, bv_w_pad = self.process_word_batch(bv['words'])
        bv_t, bv_pos = self.process_pos_batch(bv['tags'])
        seq_len_t, bv_t_pad, bv_t_1hot = self.process_tag_batch(bv_t)
        return seq_len_w, seq_len_t, bv_w_pad, bv_pos, bv_t, bv_t_pad, bv_t_1hot

    def get_batch(self):
        d_size = len(self._data['words'])
        num_batches = np.ceil(float(d_size)/self._batch_size)
        words = np.array_split(self._data['words'], num_batches)
        tags = np.array_split(self._data['tags'], num_batches)
        return [dict(zip(('words','tags'),(w,t))) for w,t in zip(words, tags)]

    def restore(self, batch):
        it = iter(batch)
        return [x for x in (list(islice(it, n)) for n in self._seq_len)]
