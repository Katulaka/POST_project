import math
import os
import sys
import time
import tensorflow as tf
import numpy as np

from POST_main import get_model

def save_ckpt(sess, model, ckpt_path, ckpt_dir):
    ckpt_path = os.path.join(ckpt_path, ckpt_dir)
    if not os.path.exists(ckpt_path):
        try:
            os.makedirs(os.path.abspath(ckpt_path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    model.saver.save(sess, ckpt_path, global_step=model.global_step)

def _train(config, batcher):

    step_time, loss = 0.0, 0.0
    train_graph = tf.Graph()
    with tf.Session(graph = train_graph) as sess:
        model = get_model(sess, config, train_graph, 'train')
        current_step = model._step.eval()
        for i in range(config.num_epochs):
            for bv in batcher.get_permute_batch():
                start_time = time.time()
                # w_in, w_len, c_in, c_len, pos, _, t_in, t_len, trgts = batcher.process(bv)
                # step_loss, _, _  = model.step(sess, w_in, w_len, c_in, c_len,
                # pos, t_in, t_len, trgts)
                step_loss, _, _  = model.step(sess, batcher._process(bv))
                current_step += 1
                step_time += (time.time() - start_time) / config.steps_per_ckpt
                loss += step_loss / config.steps_per_ckpt
                # Once in a while, we save checkpoint, print statistics
                if current_step % config.steps_per_ckpt == 0:
                     # Print statistics for the previous epoch.
                    perplex = math.exp(loss) if loss < 300 else float('inf')
                    # Save checkpoint and zero timer and loss.
                    save_ckpt(sess, model, config.ckpt_path, config.ckpt_dir)
                    print ("[[train_model:]] step %d learning rate %f step-time %.3f"
                               " perplexity %.6f (loss %.6f)" %
                               (current_step, model.learning_rate.eval(),
                               step_time, perplex, loss))
                    if current_step == 20:
                        sess.run(model.learning_rate_decay_op)
                    step_time, loss = 0.0, 0.0
                    sys.stdout.flush()

def _dev(config, batcher):

    step_time, tot_loss = 0.0, 0.0
    dev_graph = tf.Graph()
    with tf.Session(graph=dev_graph) as sess:
        model = get_model(sess, config, dev_graph)
        current_step =  0
        for bv in batcher.get_permute_batch():
            start_time = time.time()
            # w_in, w_len, c_in, c_len, pos, _, t_in, t_len, trgts = batcher.process(bv)
            # step_loss = model.dev_step(sess, w_in, w_len, c_in, c_len, pos,
            #                             t_in, t_len, trgts)
            step_loss = model.dev_step(sess, batcher._process(bv))
            current_step += 1
            step_time = (time.time() - start_time) / current_step
            tot_loss += step_loss
            loss = tot_loss / current_step
            perplex = math.exp(loss) if loss < 300 else float('inf')
            print ("[[train_model(dev):]] step %d step-time %.3f perplexity %.6f (loss %.6f)" %
                   (current_step, step_time, perplex, loss))
            sys.stdout.flush()
        return loss

def train(config, batcher_train, batcher_dev):

    # This is the training loop.
    dev_losses = [np.inf]
    while dev_losses[-1] > config.th_loss:
        _train(config, batcher_train)
        dev_losses.append(_dev(config, batcher_dev))
