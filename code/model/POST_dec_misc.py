import numpy as np
from multiprocessing import Queue
from utils.utils import ProcessWithReturnValue


def multi_process_decode(config, vocab, batcher, t_op):
    batch_list = batcher.get_batch()

    decode_graph = tf.Graph()
    with tf.Session(graph=decode_graph) as sess:
        model = get_model(sess, config, decode_graph)

        decoded_trees = [0] * num_batches
        twrv = [0] * num_batches
        res_q = [Queue()] * num_batches
        for i, batch in enumerate(batch_list):
            beam_pair, word_tokens = decode_bs(sess, model, config, vocab,
                                                 batcher, batch, t_op)

            twrv[i] = ProcessWithReturnValue(target=decode_batch, name=i,
                                            res_q=res_q[i],
                                            args=(beam_pair, word_tokens,
                                                config.num_goals,
                                                config.time_out))
            twrv[i].start()

        for i in xrange(len(batch_list)):
            print ("[Process Debug] Waiting for Process[%d] to end"%i)
            _, _decoded_trees = twrv[i].join()
            print ("[Process Debug] Ended Process[%d]"%i)
            decoded_trees[i] = _decoded_trees

    decode_tags = to_mrg(decoded_trees)
    return decode_tags

def stats(config, vocab, batcher, t_op, data_file):
    stat_graph = tf.Graph()
    with tf.Session(graph=stat_graph) as sess:
        model = get_model(sess, config, stat_graph,)
        beam_rank = []
        batch_list = batcher.get_batch()
        len_batch_list = len(batch_list)
        for i, bv in enumerate(batch_list):
            # w_len, _, words, pos, tags, _, _ = batcher.process(bv)
            w_len, _, words, pos, tags, _, _, c_len, chars = batcher._process(bv)
            if config.use_pos:
                pos = model.pos_decode(sess, words, w_len, c_len, chars)

            bs = BeamSearch(model,
                            config.beam_size,
                            vocab['tags'].token_to_id('GO'),
                            vocab['tags'].token_to_id('EOS'),
                            config.dec_timesteps)
            words_cp = copy.copy(words)
            w_len_cp = copy.copy(w_len)
            pos_cp = copy.copy(pos)
            c_len_cp = copy.copy(c_len)
            chars_cp = copy.copy(chars)
            search_fn = bs.greedy_beam_search if config.greedy else bs.beam_search
            best_beams = search_fn(sess, words_cp, w_len_cp, pos_cp, c_len_cp, chars_cp)
            # best_beams = search_fn(sess, words_cp, w_len_cp, pos_cp)
            tags_cp = copy.copy(tags)
            tags_cp = [t for tc in tags_cp for t in tc]
            for dec_in, beam_res in zip(tags_cp, best_beams['tokens']):
                try:
                    beam_rank.append(beam_res.index(dec_in) + 1)
                except ValueError:
                    beam_rank.append(config.beam_size + 1)
            with open(data_file, 'w') as outfile:
                json.dump(beam_rank, outfile)
            print ("Finished batch %d/%d: Mean beam rank so for is %f" \
             %(i+1, len_batch_list, np.mean(beam_rank)))
    return np.mean(beam_rank)

def verify(t_vocab, batcher, t_op):
    decoded_tags = []
    for bv in batcher.get_batch():
        _, _, _, _, tags, _, _ = batcher.process(bv)
        verify_tags = t_op.combine_fn(t_vocab.to_tokens(tags))
        verify_pair = [[pair] for pair in zip(verify_tags, [1.]*len(tags))]
        for verify_tag in batcher.restore(verify_pair):
            tree = solve_tree_search(tags, 1, 100)
            decode_tags.append(to_mrg(decoded_trees))
    return decoded_tags
