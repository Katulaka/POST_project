from __future__ import print_function

import copy
import numpy as np
from itertools import islice

from vocab import pad, add_go, add_eos, to_onehot
from utils import operate_on_Narray, flatten_to_1D


class Batcher(object):

    def __init__(self, data, batch_size, reverse):
        self._data = data
        self._batch_size = batch_size
        self._revese = reverse
        self._d_size = len(data.values()[0])

    def get_batch_size(self):
        return self._batch_size

    def _get_pos(self, data, pos_id):
        return data[pos_id]

    def get_pos(self, batch_data, pos_id):
        return operate_on_Narray(batch_data, self._get_pos, pos_id)

    def seq_len(self, batch_data):
        return operate_on_Narray(batch_data, len)

    def get_random_batch(self):
        batch = dict()
        d_index = np.random.randint(self._d_size, size=self._batch_size)
        for k in self._data.keys():
            batch[k] = np.array(self._data[k])[d_index]
        return batch

    def get_permute_batch(self):
        batch = dict()
        num_batches = np.ceil(float(self._d_size)/self._batch_size)
        batch_permute = np.random.permutation(int(num_batches))
        batch = {k : np.array_split(self._data[k], num_batches)
                    for k in self._data.keys()}
        return [{k: batch[k][i] for k in self._data.keys()}
                    for i in batch_permute]


    def get_batch(self):
        num_batches = np.ceil(float(self._d_size)/self._batch_size)
        words = np.array_split(self._data['words'], num_batches)
        tags = np.array_split(self._data['tags'], num_batches)
        return [dict(zip(('words','tags'),(w,t))) for w,t in zip(words, tags)]

    def to_in_data_format(self, batch_data):
        pad(batch_data)
        batch_data = np.vstack([np.expand_dims(x, 0) for x in batch_data])
        return batch_data

    def process(self, bv):

        self._seq_len = self.seq_len(bv['words'].tolist())

        #Process words input batch
        bv_w = copy.deepcopy(bv['words'].tolist())
        bv_w_delim = add_eos(add_go(bv_w))
        seq_len_w = self.seq_len(bv_w_delim)
        max_len_w = max(seq_len_w)
        bv_w_pad = pad(bv_w_delim, max_len_w)
        bv_w_in = np.vstack(bv_w_pad)

        #Process tags input batch
        bv_t = copy.deepcopy(bv['tags'].tolist())
        bv_t_go = add_go(bv_t)
        seq_len_t = np.reshape(pad(self.seq_len(bv_t_go), max_len_w), [-1])
        max_len_t = max(seq_len_t)

        bv_t_pad = []
        for _bv_t in pad(bv_t_go, max_len_t):
            _bv_t = np.array(_bv_t)
            _bv_t.resize(max_len_w, max_len_t)
            bv_t_pad.append(_bv_t.tolist())
        bv_t_in = np.reshape(bv_t_pad, (-1, np.shape(bv_t_pad)[-1]))

        pos_id = 0 if self._revese else -1
        bv_pos = self.get_pos(bv_t, pos_id)
        bv_pos_delim = add_eos(add_go(bv_pos))
        bv_pos_pad = pad(bv_pos_delim, max_len_w)
        bv_pos_in = np.vstack(bv_pos_pad)

        bv_t_eos = flatten_to_1D(add_eos(bv_t))
        return seq_len_w, seq_len_t, bv_w_in, bv_pos_in, bv_t, bv_t_in, bv_t_eos


    def restore(self, batch):
        it = iter(batch)
        return [x for x in (list(islice(it, n)) for n in self._seq_len)]
