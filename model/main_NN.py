from __future__ import print_function

import NN_model as nnModel
import utils.data_preproc as dp

import time
# import os
import math
import sys
import tensorflow as tf


def create_model(session, config):

    start_time = time.time()
    
    model = nnModel.NNModel(
            config.batch_size, config.word_embedding_size, config.tag_embedding_size, 
            config.n_hidden_fw, config.n_hidden_bw, config.n_hidden_lstm, 
            config.word_vocabulary_size, config.tag_vocabulary_size, config.num_steps,
            config.learning_rate, config.learning_rate_decay_factor, config.max_gradient_norm)

    ckpt = tf.train.get_checkpoint_state(config.checkpoint_path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
        if config.learning_rate < model.learning_rate.eval():       
            print('Re-setting learning rate to %f' % config.learning_rate)
            session.run(model.learning_rate.assign(config.learning_rate), [])
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    
    end_time = time.time()
    print("Time to create Gen_RNN model: %.2f" % (end_time - start_time))

    return model


def train(config):

    with tf.Session() as sess:
        # print("Creating %d layers of %d units." % (gen_config.num_layers, gen_config.size))
        model = create_model(sess, config)

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        moving_average_loss = 0.0
        current_step = 0
        previous_losses = []

        step_loss_summary = tf.Summary()
        writer = tf.summary.FileWriter("../logs/", sess.graph)
        
        train_set = dp.gen_data(config.train_dir)  # TODO

        while True:

            # Get a batch and make a step.
            start_time = time.time()
            word_seq_lens, tag_seq_lens, word_inputs, tag_inputs, y = model.get_batch(train_set, config.tag_vocabulary_size, config.batch_size)

            pred, step_loss, _ = model.step(sess, word_seq_lens[0], tag_seq_lens[0], word_inputs[0], tag_inputs[0], y[0]) #TODO

            step_time += (time.time() - start_time) / config.steps_per_checkpoint
            loss += step_loss / config.steps_per_checkpoint
            moving_average_loss += step_loss
            current_step += 1

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % config.steps_per_checkpoint == 0:

                bucket_value = step_loss_summary.value.add()
                bucket_value.tag = "loss"
                bucket_value.simple_value = float(loss)
                writer.add_summary(step_loss_summary, current_step)

                # Print statistics for the previous epoch.
                perplexity = math.exp(loss) if loss < 300 else float('inf')
                print ("global step %d learning rate %.4f step-time %.2f perplexity "
                       "%.6f" % (model.global_step.eval(), model.learning_rate.eval(),
                                 step_time, perplexity))
                # Decrease learning rate if no improvement was seen over last 3 times.
                # step_tracker = model.global_step.eval()

                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                # Save checkpoint and zero timer and loss.
                # checkpoint_path = os.path.join()
                model.saver.save(sess, config.checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0
                sys.stdout.flush()


def gen_toy_data(batch_size,tag_vocabulary_size,word_vocabulary_size, max_len = 3):
    import numpy as np
    ex = np.random.randint(tag_vocabulary_size, size=(max_len*max_len*batch_size))
    labels = np.zeros((max_len*max_len*batch_size, tag_vocabulary_size))
    labels[np.arange(max_len*max_len*batch_size), ex] = 1
    w_in = np.random.randint(word_vocabulary_size, size=(batch_size, max_len))
    t_in = np.random.randint(tag_vocabulary_size, size=(batch_size, max_len, max_len))
    w_s_len = [max_len] * batch_size
    t_s_len = [max_len] * (max_len*batch_size)

    return t_s_len, w_s_len, t_in, w_in, labels


#    loss_sum = tf.summary.scalar("loss", loss)
#    summary_op = tf.summary.merge_all()
#    
#    
#        
#    logs_path = '/Users/katia.patkin/Berkeley/Research/BiRNN/tmp'    
#    writer = tf.summary.FileWriter(logdir=logs_path, graph= sess.graph)    
#
#    training_epochs = 10
#    num_examples    = 40
#    display_step    = 1
#
#    
#    for epoch in range(training_epochs):
#    
#        avg_loss = 0.
#        total_batch = num_examples/batch_size
#
#        for i in range(total_batch):
#            t_s_len, w_s_len, t_in, w_in, labels = gen_toy_data(batch_size,tag_vocabulary_size,word_vocabulary_size,
#                                                               max_len = 3)
#                             
#            feed_dict =  {word_inputs : w_in, tag_inputs : t_in, word_seq_lens : w_s_len,
#                           tag_seq_lens: t_s_len, y : labels}
#
#            [ap, l] = sess.run([summary_op, loss], feed_dict = feed_dict)
#
#            avg_loss += l/total_batch
#            if (epoch+1) % display_step == 0:
#                print("Epoch:", '%04d' % (epoch+1), "loss=", "{:.9f}".format(avg_loss))
