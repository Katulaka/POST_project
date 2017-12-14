import copy
import tensorflow as tf
import json
import cPickle
import numpy as np
from multiprocessing import Queue


from astar.search import solve_tree_search
from utils.tags.tag_tree import convert_to_TagTree
from utils.tags.trees_to_tags import to_mrg
from beam.search import BeamSearch
from utils.utils import ProcessWithReturnValue

from POST_main import get_model

def decode_bs(sess, model, config, w_vocab, t_vocab, batcher, batch, t_op):

    bs = BeamSearch(model, config.beam_size, t_vocab.token_to_id('GO'),
                    t_vocab.token_to_id('EOS'), config.dec_timesteps)

    w_len, _, words, pos, _, _, _ = batcher.process(batch)

    words_cp = copy.copy(words)
    w_len_cp = copy.copy(w_len)
    pos_cp = copy.copy(pos)
    best_beams = bs.beam_search(sess, words_cp, w_len_cp, pos_cp)
    beam_tags = t_op.combine_fn(t_vocab.to_tokens(best_beams['tokens']))
    _beam_pair = map(lambda x, y: zip(x, y),
                                beam_tags, best_beams['scores'])
    beam_pair = batcher.restore(_beam_pair)
    _words = [s[1:s_len-1].tolist() for s, s_len in zip(words, w_len)]
    word_tokens = w_vocab.to_tokens(_words)
    return beam_pair, word_tokens

def decode_batch(beam_pair, word_tokens, num_goals, time_out):

    decode_trees = []
    num_sentences = len(word_tokens)
    for i, (beam_tag, sent) in enumerate(zip(beam_pair, word_tokens)):
        print ("Staring astar search for sentence %d /"
                " %d [tag length %d]" %
                (i+1, num_sentences, len(beam_tag)))

        if all(beam_tag):
            tags = convert_to_TagTree(beam_tag, sent)
            trees = solve_tree_search(tags, num_goals, time_out)
        else:
            trees = []
        decode_trees.append(trees)
    return decode_trees

def decode(config, w_vocab, t_vocab, batcher, t_op):

    batch_list = batcher.get_batch()

    decode_graph = tf.Graph()
    with tf.Session(graph=decode_graph) as sess:
        model = get_model(sess, config, decode_graph)

        if config.multi_processing:
            decoded_trees = [0] * num_batches
            twrv = [0] * num_batches
            res_q = [Queue()] * num_batches
            for i, batch in enumerate(batch_list):
                beam_pair, word_tokens = decode_bs(sess, model, config, w_vocab,
                                                t_vocab, batcher, batch, t_op)


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

        else:
            decoded_trees = []
            for batch in batch_list:
                beam_pair, word_tokens = decode_bs(sess, model, config,
                                                    w_vocab, t_vocab, batcher,
                                                    batch, t_op)

                decoded_trees.extend(decode_batch(beam_pair, word_tokens,
                                                config.num_goals,
                                                config.time_out))
                f = open('decode_trees', 'wb')
                cPickle.dump(decoded_trees, f, protocol=cPickle.HIGHEST_PROTOCOL)
                f.close()
                with open('decode_mrg', 'w') as outfile:
                    json.dump(to_mrg(decoded_trees), outfile)

        decode_tags = to_mrg(decoded_trees)
    return decode_tags

def stats(config, w_vocab, t_vocab, batcher, t_op, data_file):
    stat_graph = tf.Graph()
    with tf.Session(graph=stat_graph) as sess:
        model = get_model(sess, config, stat_graph,)
        beam_rank = []
        batch_list = batcher.get_batch()
        len_batch_list = len(batch_list)
        for i, bv in enumerate(batch_list):
            w_len, _, words, pos, tags, _, _ = batcher.process(bv)

            bs = BeamSearch(model,
                            config.beam_size,
                            t_vocab.token_to_id('GO'),
                            t_vocab.token_to_id('EOS'),
                            config.dec_timesteps)
            words_cp = copy.copy(words)
            w_len_cp = copy.copy(w_len)
            pos_cp = copy.copy(pos)
            search_fn = bs.greedy_beam_search if config.greedy else bs.beam_search
            best_beams = search_fn(sess, words_cp, w_len_cp, pos_cp)
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
