from __future__ import print_function

import copy
import json
import math
import os
import sys
import time
import tensorflow as tf
import numpy as np

from astar.search import solve_tree_search
from beam.search import BeamSearch
from POST_model import POSTModel


def get_model(session, config, special_tokens, add_pos_in, mode='decode'):
    """ Creates new model for restores existing model """
    start_time = time.time()

    model = POSTModel(config.batch_size, config.word_embedding_size,
                        config.tag_embedding_size, config.n_hidden_fw,
                        config.n_hidden_bw, config.n_hidden_lstm,
                        config.word_vocabulary_size,
                        config.tag_vocabulary_size, config.learning_rate,
                        config.learning_rate_decay_factor, add_pos_in, mode)
    model.build_graph(special_tokens)

    ckpt = tf.train.get_checkpoint_state(config.checkpoint_path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
        if mode == 'train':
            if config.learning_rate < model.learning_rate.eval():
                print('Re-setting learning rate to %f' % config.learning_rate)
                session.run(model.learning_rate.assign(config.learning_rate), [])
        print("Time to restore model: %.2f" % (time.time() - start_time))
    elif mode == 'train':
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        print("Time to create model: %.2f" % (time.time() - start_time))
    else:
        raise ValueError('Model not found to restore.')
        return None
    return model

def train(config, batcher, cp_path, special_tokens, add_pos_in):

    with tf.Session() as sess:
        model = get_model(sess, config, special_tokens, add_pos_in, 'train')

        # This is the training loop.
        step_time = 0.0
        loss = 0.0
        moving_avg_loss = 0.0 #TODO fix this moving_avg_loss
        current_step = 0
        decay = 0.999
        prev_losses = []

        while True:
            # Get a batch and make a step.
            start_time = time.time()
            bv = batcher.get_random_batch()
            w_len, t_len, words, pos, _, tags, targets = batcher.process(bv)
            step_loss, _  = model.step(sess, w_len, t_len, words, pos, tags, targets)
            step_time += (time.time() - start_time) / config.steps_per_checkpoint
            loss += step_loss / config.steps_per_checkpoint
            if moving_avg_loss == 0:
                moving_avg_loss = step_loss
            else:
                moving_avg_loss = (moving_avg_loss * decay + (1 - decay)*loss)
            current_step += 1

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % config.steps_per_checkpoint == 0:

                # Print statistics for the previous epoch.
                perplex = math.exp(loss) if loss < 300 else float('inf')
                print ("global step %d learning rate %f step-time %.2f"
                       " perplexity %.6f (loss %.6f)" % (model.global_step.eval(),
                        model.learning_rate.eval(), step_time, perplex, loss))

                # Decrease learning rate if no improvement
                # was seen over last 3 times.
                if len(prev_losses) > 2 and loss > max(prev_losses[-3:]) :
                    sess.run(model.learning_rate_decay_op)
                if model.learning_rate.eval() < 0.000001:
                    sess.run(model.learning_rate.assign(0.1), [])

                prev_losses.append(loss)
                # Save checkpoint and zero timer and loss.
                ckpt_path = os.path.join(config.checkpoint_path, cp_path)
                if not os.path.exists(config.checkpoint_path):
                    try:
                        os.makedirs(os.path.abspath(config.checkpoint_path))
                    except OSError as exc: # Guard against race condition
                        if exc.errno != errno.EEXIST:
                            raise
                model.saver.save(sess, ckpt_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0
                sys.stdout.flush()


# def decode(config, w_vocab, t_vocab, batcher, t_op, add_pos_in, _tags):
def decode(config, w_vocab, t_vocab, batcher, t_op, add_pos_in):
    with tf.Session() as sess:
        model = get_model(sess, config, w_vocab.get_ctrl_tokens(), add_pos_in)

        decoded_tags = []
        orig_tags = []
        for bv in batcher.get_batch():
            w_len, _, words, pos, _, _, _ = batcher.process(bv)

            bs = BeamSearch(model,
                            config.beam_size,
                            t_vocab.token_to_id('GO'),
                            t_vocab.token_to_id('EOS'),
                            config.dec_timesteps)
            words_cp = copy.copy(words)
            w_len_cp = copy.copy(w_len)
            pos_cp = copy.copy(pos)
            import pdb; pdb.set_trace()
            best_beams = bs.beam_search(sess, words_cp, w_len_cp, pos_cp)
            beam_tags = t_op.combine_fn(t_vocab.to_tokens(best_beams['tokens']))
            beam_pair = map(lambda x, y: zip(x, y), beam_tags, best_beams['scores'])
            for i, beam_tag in enumerate(batcher.restore(beam_pair)):
                print ("Staring astar search for sentence %d / %d [tag length %d]"
                % (i+1, batcher.get_batch_size(),len(beam_tag)))
                path, tree, new_tag = solve_tree_search(beam_tag, 1)
                decoded_tags.append(new_tag)

    return decoded_tags

def stats(config, w_vocab, t_vocab, batcher, t_op, add_pos_in, data_file):

    with tf.Session() as sess:
        model = get_model(sess, config, w_vocab.get_ctrl_tokens(), add_pos_in)
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
            best_beams = bs.beam_search(sess, words_cp, w_len_cp, pos_cp)
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
#TODO
    decoded_tags = []
    for bv in batcher.get_batch():
        _, _, _, _, tags, _, _ = batcher.process(bv)
        import pdb; pdb.set_trace()
        verify_tags = t_op.combine_fn(t_vocab.to_tokens(tags))
        verify_pair = [[pair] for pair in zip(verify_tags, [1.]*len(tags))]
        for verify_tag in batcher.restore(verify_pair):
            path, tree, new_tag = solve_tree_search(verify_tag, 1)
            decoded_tags.append(new_tag)
    return decoded_tags
