from __future__ import print_function

import NN_model as nnModel
import time
# import os
import math
import sys
import tensorflow as tf



def _Train(model, data_batcher):
  """Runs model training."""
  with tf.device('/cpu:0'):
    model.build_graph()
    saver = tf.train.Saver()
    # Train dir is different from log_root to avoid summary directory
    # conflict with Supervisor.
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir)
    sv = tf.train.Supervisor(logdir=FLAGS.log_root,
                             is_chief=True,
                             saver=saver,
                             summary_op=None,
                             save_summaries_secs=60,
                             save_model_secs=FLAGS.checkpoint_secs,
                             global_step=model.global_step)
    sess = sv.prepare_or_wait_for_session(config=tf.ConfigProto(
        allow_soft_placement=True))
    running_avg_loss = 0
    step = 0
    while not sv.should_stop() and step < FLAGS.max_run_steps:
      (article_batch, abstract_batch, targets, article_lens, abstract_lens,
       loss_weights, _, _) = data_batcher.NextBatch()
      (_, summaries, loss, train_step) = model.run_train_step(
          sess, article_batch, abstract_batch, targets, article_lens,
          abstract_lens, loss_weights)

      summary_writer.add_summary(summaries, train_step)
      running_avg_loss = _RunningAvgLoss(
          running_avg_loss, loss, summary_writer, train_step)
      step += 1
      if step % 100 == 0:
        summary_writer.flush()
    sv.Stop()
    return running_avg_loss


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


def train(config, train_set):

    with tf.Session() as sess:
        model = create_model(sess, config)

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        moving_average_loss = 0.0
        current_step = 0
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
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                # Save checkpoint and zero timer and loss.
                # checkpoint_path = os.path.join() #TODO add checkpoint path generation
                model.saver.save(sess, config.checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0
                sys.stdout.flush()

# def evaluate(config):
#
#     model = create_model(sess, config)
#
#
# def _Decode(self, saver, sess):
#   """Restore a checkpoint and decode it.
#   Args:
#     saver: Tensorflow checkpoint saver.
#     sess: Tensorflow session.
#   Returns:
#     If success, returns true, otherwise, false.
#   """
#   ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
#   if not (ckpt_state and ckpt_state.model_checkpoint_path):
#     tf.logging.info('No model to decode yet at %s', FLAGS.log_root)
#     return False
#
#   tf.logging.info('checkpoint path %s', ckpt_state.model_checkpoint_path)
#   ckpt_path = os.path.join(
#       FLAGS.log_root, os.path.basename(ckpt_state.model_checkpoint_path))
#   tf.logging.info('renamed checkpoint path %s', ckpt_path)
#   saver.restore(sess, ckpt_path)
#
#   self._decode_io.ResetFiles()
#   for _ in xrange(FLAGS.decode_batches_per_ckpt):
#     (article_batch, _, _, article_lens, _, _, origin_articles,
#      origin_abstracts) = self._batch_reader.NextBatch()
#     for i in xrange(self._hps.batch_size):
#       bs = beam_search.BeamSearch(
#           self._model, self._hps.batch_size,
#           self._vocab.WordToId(data.SENTENCE_START),
#           self._vocab.WordToId(data.SENTENCE_END),
#           self._hps.dec_timesteps)
#
#       article_batch_cp = article_batch.copy()
#       article_batch_cp[:] = article_batch[i:i+1]
#       article_lens_cp = article_lens.copy()
#       article_lens_cp[:] = article_lens[i:i+1]
#       best_beam = bs.BeamSearch(sess, article_batch_cp, article_lens_cp)[0]
#       decode_output = [int(t) for t in best_beam.tokens[1:]]
#       self._DecodeBatch(
#           origin_articles[i], origin_abstracts[i], decode_output)
#   return True
#

#    loss_sum = tf.summary.scalar("loss", loss)
#    summary_op = tf.summary.merge_all()

#    logs_path = '/Users/katia.patkin/Berkeley/Research/BiRNN/tmp'
#    writer = tf.summary.FileWriter(logdir=logs_path, graph= sess.graph)
#
#    training_epochs = 10
#    num_examples    = 40
#    display_step    = 1
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