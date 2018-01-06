import copy
import tensorflow as tf
import json
import cPickle

from astar.search import solve_tree_search
from beam.search import BeamSearch

from POST_main import get_model

def decode_bs(sess, model, config, vocab, batcher, batch, t_op):

    bs = BeamSearch(model, config.beam_size, vocab['tags'].token_to_id('GO'),
                    vocab['tags'].token_to_id('EOS'), config.dec_timesteps)

    # words, w_len, chars, c_len, pos, _, _, _, _ = batcher.process(batch)
    #
    # w_cp = copy.copy(words)
    # wlen_cp = copy.copy(w_len)
    # pos_cp = copy.copy(pos)
    # clen_cp = copy.copy(c_len)
    # c_cp = copy.copy(chars)
    bv = batcher.process(batch)
    bv_cp = copy.copy(bv)

    # best_beams = bs.beam_search(sess, w_cp, wlen_cp, c_cp, clen_cp, pos_cp)
    best_beams = bs.beam_search(sess, bv_cp)
    beam_tags = t_op.combine_fn(vocab['tags'].to_tokens(best_beams['tokens']))
    _beam_pair = map(lambda x, y: zip(x, y), beam_tags, best_beams['scores'])
    beam_pair = batcher.restore(_beam_pair)
    words = [w[1:l-1].tolist() for w, l in zip(bv['word']['in'], bv['word']['len'])]
    word_tokens = vocab['words'].to_tokens(words)
    return beam_pair, word_tokens

def decode_batch(beam_pair, word_tokens, no_val_gap, num_goals, time_out):

    decode_trees = []
    num_sentences = len(word_tokens)
    for i, (beam_tag, sent) in enumerate(zip(beam_pair, word_tokens)):
        print ("Staring astar search for sentence %d /"
                " %d [tag length %d]" %(i+1, num_sentences, len(beam_tag)))

        if all(beam_tag):
            trees = solve_tree_search(beam_tag, sent, no_val_gap, num_goals, time_out)
        else:
            trees = []
        decode_trees.append(trees)
    return decode_trees

def decode(config, vocab, batcher, t_op):

    batch_list = batcher.get_batch()

    decode_graph = tf.Graph()
    with tf.Session(graph=decode_graph) as sess:
        model = get_model(sess, config, decode_graph)

        decoded_trees = []
        for batch in batch_list:
            beam_pair, word_tokens = decode_bs(sess, model, config, vocab,
                                                batcher, batch, t_op)

            decoded_trees.extend(decode_batch(beam_pair, word_tokens,
                                            config.no_val_gap,
                                            config.num_goals,
                                            config.time_out))
            import pdb; pdb.set_trace()
            # f = open('decode_trees', 'wb')
            # cPickle.dump(decoded_trees, f, protocol=cPickle.HIGHEST_PROTOCOL)
            # f.close()
            # with open('decode_mrg', 'w') as outfile:
            #     json.dump(trees_to_ptb(decoded_trees), outfile)

    # decode_tags = trees_to_ptb(decoded_trees)
    return decode_tags
