import copy
import tensorflow as tf
import json
import cPickle

from astar.search import solve_tree_search
from utils.tags.tag_tree import convert_to_TagTree
from utils.tags.trees_to_tags import to_mrg
from beam.search import BeamSearch

from POST_main import get_model

def decode_bs(sess, model, config, vocab, batcher, batch, t_op):

    bs = BeamSearch(model, config.beam_size, vocab['tags'].token_to_id('GO'),
                    vocab['tags'].token_to_id('EOS'), config.dec_timesteps)

    w_len, _, words, pos, _, _, _, c_len, chars = batcher._process(batch)
    if config.use_pos:
        pos = model.pos_decode(sess, words, w_len, c_len, chars)

    w_cp = copy.copy(words)
    wlen_cp = copy.copy(w_len)
    pos_cp = copy.copy(pos)
    clen_cp = copy.copy(c_len)
    c_cp = copy.copy(chars)

    best_beams = bs.beam_search(sess, w_cp, wlen_cp, pos_cp, clen_cp, c_cp)
    beam_tags = t_op.combine_fn(vocab['tags'].to_tokens(best_beams['tokens']))
    _beam_pair = map(lambda x, y: zip(x, y), beam_tags, best_beams['scores'])
    beam_pair = batcher.restore(_beam_pair)
    _words = [s[1:s_len-1].tolist() for s, s_len in zip(words, w_len)]
    word_tokens = vocab['words'].to_tokens(_words)
    return beam_pair, word_tokens

def decode_batch(beam_pair, word_tokens, num_goals, time_out):

    decode_trees = []
    num_sentences = len(word_tokens)
    for i, (beam_tag, sent) in enumerate(zip(beam_pair, word_tokens)):
        print ("Staring astar search for sentence %d /"
                " %d [tag length %d]" %(i+1, num_sentences, len(beam_tag)))

        if all(beam_tag):
            tags = convert_to_TagTree(beam_tag, sent)
            trees = solve_tree_search(tags, num_goals, time_out)
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
                                            config.num_goals,
                                            config.time_out))
            # f = open('decode_trees', 'wb')
            # cPickle.dump(decoded_trees, f, protocol=cPickle.HIGHEST_PROTOCOL)
            # f.close()
            # with open('decode_mrg', 'w') as outfile:
            #     json.dump(to_mrg(decoded_trees), outfile)

    decode_tags = to_mrg(decoded_trees)
    return decode_tags
