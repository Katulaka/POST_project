from __future__ import print_function

import NN_model as nnModel
import beam_search as beam_search
import time
import math
import sys
import os
import tensorflow as tf
import copy
import numpy as np


TAGS_ONLY = False

def create_model(session, config):

    start_time = time.time()

    model = nnModel.NNModel(
            config.batch_size, config.word_embedding_size,
            config.tag_embedding_size,
            config.n_hidden_fw, config.n_hidden_bw, config.n_hidden_lstm,
            config.word_vocabulary_size, config.tag_vocabulary_size,
            config.num_steps,
            config.learning_rate, config.learning_rate_decay_factor,
            config.max_gradient_norm)

    # model.build_graph()
    ckpt = tf.train.get_checkpoint_state(config.checkpoint_path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
        if config.learning_rate < model.learning_rate.eval():
            print('Re-setting learning rate to %f' % config.learning_rate)
            session.run(model.learning_rate.assign(config.learning_rate), [])
        end_time = time.time()
        print("Time to restore Gen_RNN model: %.2f" % (end_time - start_time))
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        end_time = time.time()
        print("Time to create Gen_RNN model: %.2f" % (end_time - start_time))

    return model


def restore_model(session, config):

    start_time = time.time()

    model = nnModel.NNModel(
            config.batch_size, config.word_embedding_size, config.tag_embedding_size,
            config.n_hidden_fw, config.n_hidden_bw, config.n_hidden_lstm,
            config.word_vocabulary_size, config.tag_vocabulary_size, config.num_steps,
            config.learning_rate, config.learning_rate_decay_factor, config.max_gradient_norm)

    # model.build_graph()
    ckpt = tf.train.get_checkpoint_state(config.checkpoint_path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
        end_time = time.time()
        print("Time to restore Gen_RNN model: %.2f" % (end_time - start_time))
        return model

    print ("Error")
    return


def train(config, train_set, cp_path):

    with tf.Session() as sess:
        model = create_model(sess, config)

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        moving_average_loss = 0.0
        current_step = 0
        decay=0.999
        previous_losses = []

        step_loss_summary = tf.Summary()
        writer = tf.summary.FileWriter("../logs/", sess.graph)

        while True:
            # Get a batch and make a step.
            start_time = time.time()
            word_seq_len, tag_seq_len, words_in, tags_in, tags_in_1hot = \
                model.get_batch(train_set, config.tag_vocabulary_size, config.batch_size)
            pred, step_loss, _  = model.step(sess, word_seq_len, tag_seq_len, words_in, tags_in, tags_in_1hot)
            step_time += (time.time() - start_time) / config.steps_per_checkpoint
            loss += step_loss / config.steps_per_checkpoint
            if moving_average_loss == 0:
                moving_average_loss = step_loss
            else:
                moving_average_loss = moving_average_loss * decay + (1 - decay) * loss
            #     moving_average_loss = min(running_avg_loss, 12)
            # moving_average_loss += step_loss #TODO fix this moving_average_loss
            current_step += 1

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % config.steps_per_checkpoint == 0:

                bucket_value = step_loss_summary.value.add()
                bucket_value.tag = "loss"
                bucket_value.simple_value = float(loss)
                writer.add_summary(step_loss_summary, current_step)

                # Print statistics for the previous epoch.
                perplexity = math.exp(loss) if loss < 300 else float('inf')
                print ("global step %d learning rate %f step-time %.2f perplexity "
                       "%.6f" % (model.global_step.eval(), model.learning_rate.eval(),
                                 step_time, perplexity))

                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and \
                    loss > max(previous_losses[-3:]) :
                    #and model.learning_rate.eval() > 0.000001:
                    sess.run(model.learning_rate_decay_op)
                if model.learning_rate.eval() < 0.000001:
                    sess.run(model.learning_rate.assign(0.1), [])

                previous_losses.append(loss)
                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join(config.checkpoint_path, cp_path) #TODO add checkpoint path generation
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0
                sys.stdout.flush()

from collections import Counter

def decode_tags_only(config, train_set, reverse_dict):
    with tf.Session() as sess:
        model = restore_model(sess, config)
        guesses = Counter()
        while(True):
            word_seq_len, tag_seq_len, words_in, tags_in, tags_in_1hot = \
                model.get_batch(train_set, config.tag_vocabulary_size,
                        config.batch_size)
            predicted_tags = decode_one_tag(sess, model, word_seq_len, words_in)
            true_tags = np.squeeze(tags_in[:,1:2], axis=1)
            guesses += Counter(zip(predicted_tags, true_tags)) # Add counts of (predicted, true) pairs
            import pdb; pdb.set_trace()

def decode_one_tag(sess, model, word_seq_len, words_in):
    input_feed = {
        model.word_inputs: words_in,
        model.word_seq_lens: word_seq_len
    }
    output_feed = model.logits
    results = sess.run(output_feed, input_feed)
    return np.argmax(results, axis=1)

def decode(config, train_set, reverse_dict):

    if TAGS_ONLY:
        decode_tags_only(config, train_set, reverse_dict)
        return

    with tf.Session() as sess:
        model = restore_model(sess, config)
        while(True): # for i in xrange(config.batch_size):
            word_seq_len, tag_seq_len, words_in, tags_in, tags_in_1hot = \
                model.get_batch(train_set, config.tag_vocabulary_size,
                        config.batch_size)
            bs = beam_search.BeamSearch(model, config.beam_size,
                      1, #SOS
                      2, #EOS
                      config.dec_timesteps)

            words_in_cp = copy.copy(words_in)
            word_seq_len_cp = copy.copy(word_seq_len)
            best_beam = bs.BeamSearch(sess, words_in_cp, word_seq_len_cp)
            print (best_beam)
            import pdb; pdb.set_trace()

            tmp = [[[(map(lambda x: reverse_dict['tag'][x],g[0][1:-1]), g[1])
                    for g in bi] for bi in b]
                        for b in best_beam ]
    return True
