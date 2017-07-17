from __future__ import print_function

import NN_model as nnModel
import beam_search as beam_search
import time
# import os
import math
import sys
import tensorflow as tf
import copy


def create_model(session, config):

    start_time = time.time()

    model = nnModel.NNModel(
            config.batch_size, config.word_embedding_size, config.tag_embedding_size,
            config.n_hidden_fw, config.n_hidden_bw, config.n_hidden_lstm,
            config.word_vocabulary_size, config.tag_vocabulary_size, config.num_steps,
            config.learning_rate, config.learning_rate_decay_factor, config.max_gradient_norm)

    # print("Creating %d layers of %d units." % (gen_config.num_layers, gen_config.size))
    ckpt = tf.train.get_checkpoint_state(config.checkpoint_path) #TODO verify this
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

    # print("Creating %d layers of %d units." % (gen_config.num_layers, gen_config.size))
    ckpt = tf.train.get_checkpoint_state(config.checkpoint_path) #TODO verify this
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
        if config.learning_rate < model.learning_rate.eval():
            print('Re-setting learning rate to %f' % config.learning_rate)
            session.run(model.learning_rate.assign(config.learning_rate), [])
        end_time = time.time()
        print("Time to restore Gen_RNN model: %.2f" % (end_time - start_time))
    else:
        print ("Error")
        return
    return model


def train(config, train_set):

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
                print ("global step %d learning rate %.6f step-time %.2f perplexity "
                       "%.6f" % (model.global_step.eval(), model.learning_rate.eval(),
                                 step_time, perplexity))

                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                # Save checkpoint and zero timer and loss.
                # checkpoint_path = os.path.join() #TODO add checkpoint path generation
                model.saver.save(sess, config.checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0
                sys.stdout.flush()


def decode(config, train_set):

    with tf.Session() as sess:
        model = restore_model(sess, config)

        while(True):
    # for _ in xrange(FLAGS.decode_batches_per_ckpt):
            word_seq_len, tag_seq_len, words_in, tags_in, tags_in_1hot = \
                model.get_batch(train_set, config.tag_vocabulary_size,
                        config.batch_size)
            for i in xrange(config.batch_size):
                bs = beam_search.BeamSearch(model, config.beam_size,
                          1, #SOS
                          2, #EOS
                          config.dec_timesteps)

                words_in_cp = copy.copy(words_in)
                word_seq_len_cp = copy.copy(word_seq_len)
                best_beam = bs.BeamSearch(sess, words_in_cp, word_seq_len_cp)
    #   decode_output = [int(t) for t in best_beam.tokens[1:]]
    #   self._DecodeBatch(
    #       origin_words_in[i], origin_tags_in[i], decode_output)
    return True
