from __future__ import print_function

import copy
import numpy as np
from itertools import islice

from vocab import pad, add_go, add_eos, to_onehot
from utils import arr_dim, flattenNd, operate_on_Narray


class Batcher(object):

    def __init__(self, data, batch_size, reverse):
        self._data = data
        self._batch_size = batch_size
        self._revese = reverse

    def get_batch_size(self):
        return self._batch_size

    # def seq_len(self, batch_data):
    #     return map(lambda x: len(x), batch_data)

    def _get_pos(self, data, pos_id):
        return data[pos_id]

    def get_pos(self, batch_data, pos_id):
        return operate_on_Narray(batch_data, self._get_pos, pos_id)

    def seq_len(self, batch_data):
        return operate_on_Narray(batch_data, len)

    # def seq_max(self, seq_len):
    #     return operate_on_Narray(batch_data, max)

    def get_random_batch(self):
        batch = dict()
        d_size = len(self._data['words'])
        d_index = np.random.randint(d_size, size=self._batch_size)
        batch['words'] = np.array(self._data['words'])[d_index]
        batch['tags'] = np.array(self._data['tags'])[d_index]
        return batch

    def get_batch(self):
        d_size = len(self._data['words'])
        num_batches = np.ceil(float(d_size)/self._batch_size)
        words = np.array_split(self._data['words'], num_batches)
        tags = np.array_split(self._data['tags'], num_batches)
        return [dict(zip(('words','tags'),(w,t))) for w,t in zip(words, tags)]

    def to_in_data_format(self, batch_data):
        pad(batch_data)
        batch_data = np.vstack([np.expand_dims(x, 0) for x in batch_data])
        return batch_data

    # def generate_targets(self, max_len, batch_data):
    #     batch_1hot = map(lambda x: to_onehot(x, max_len, self._vocab_size),
    #                         batch_data)
    #     return np.vstack([np.expand_dims(x, 0) for x in batch_1hot])

    def process(self, bv):
        #Process words input batch
        bv_w = copy.deepcopy(bv['words'].tolist())
        self._seq_len = self.seq_len(bv_w)
        bv_w = add_eos(add_go(bv_w))
        seq_len_w = self.seq_len(bv_w)
        max_len_w = max(flattenNd(seq_len_w, arr_dim(seq_len_w)-1))
        bv_w_pad = pad(bv_w, max_len_w)
        bv_w_in = np.vstack(bv_w_pad)

        #Process tags input batch
        bv_t = copy.deepcopy(bv['tags'].tolist())
        bv_t_go = add_go(bv_t)
        seq_len_t = self.seq_len(bv_t_go)
        max_len_t = max(flattenNd(seq_len_t, arr_dim(seq_len_t)-1))
        _bv_t_pad = pad(bv_t_go, max_len_t)

        bv_t_pad = []
        for _bv_t in _bv_t_pad:
            _bv_t = np.array(_bv_t);
            _bv_t.resize(max_len_w, max_len_t);
            bv_t_pad.append(_bv_t.tolist())
        bv_t_pad = flattenNd(bv_t_pad, 1)
        bv_t_in = np.vstack([np.expand_dims(x, 0) for x in bv_t_pad])

        pos_id = 0 if self._revese else -1
        bv_pos = self.get_pos(bv_t, pos_id)
        bv_pos = add_eos(add_go(bv_pos))
        bv_pos_pad = pad(bv_pos, max_len_w)
        bv_pos_in = np.vstack(bv_pos_pad)

        bv_t_eos = flattenNd(add_eos(bv_t), 2)
        seq_len_t = pad(seq_len_t, max_len_w)
        seq_len_t = flattenNd(seq_len_t, arr_dim(seq_len_t)-1)
        # import pdb; pdb.set_trace()
        return seq_len_w, seq_len_t, bv_w_in, bv_pos_in, bv_t, bv_t_in, bv_t_eos


    def restore(self, batch):
        it = iter(batch)
        return [x for x in (list(islice(it, n)) for n in self._seq_len)]
