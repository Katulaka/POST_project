import math
import os
import sys
import time
import tensorflow as tf
import numpy as np

from POST_main import get_model

def _train(config, batcher, cp_path):

    step_time, loss = 0.0, 0.0
    train_graph = tf.Graph()

    with tf.Session(graph=train_graph) as sess:
        model = get_model(sess, config, train_graph, 'train')

        current_step =  model.global_step.eval()
        for i in range(config.num_epochs):
            for bv in batcher.get_permute_batch():
                start_time = time.time()
                w_len, t_len, words, pos, _, tags, targets = batcher.process(bv)
                step_loss, _  = model.step(sess, w_len, t_len, words, pos, tags,
                                            targets)
                step_time += (time.time() - start_time)\
                                 / config.steps_per_checkpoint
                loss += step_loss / config.steps_per_checkpoint
                current_step += 1
                # Once in a while, we save checkpoint, print statistics
                if current_step % config.steps_per_checkpoint == 0:
                    # Print statistics for the previous epoch.
                    perplex = math.exp(loss) if loss < 300 else float('inf')
                    print ("global step %d learning rate %f step-time %.2f"
                           " perplexity %.6f (loss %.6f)" %
                           (model.global_step.eval(),
                           model.learning_rate.eval(),
                           step_time, perplex, loss))
                    if model.global_step.eval() == 20:
                        sess.run(model.learning_rate_decay_op)

                    # Save checkpoint and zero timer and loss.
                    ckpt_path = os.path.join(config.checkpoint_path, cp_path)
                    if not os.path.exists(config.checkpoint_path):
                        try:
                            os.makedirs(os.path.abspath(config.checkpoint_path))
                        except OSError as exc: # Guard against race condition
                            if exc.errno != errno.EEXIST:
                                raise
                    model.saver.save(sess, ckpt_path,
                                        global_step=model.global_step)
                    step_time, loss = 0.0, 0.0
                    sys.stdout.flush()

def _eval(config, batcher, cp_path):
    step_time, loss = 0.0, 0.0
    eval_graph = tf.Graph()
    current_step =  0

    with tf.Session(graph=eval_graph) as sess:
        model = get_model(sess, config, eval_graph)

        for bv in batcher.get_permute_batch():
            start_time = time.time()
            w_len, t_len, words, pos, _, tags, targets = batcher.process(bv)
            step_loss = model.eval_step(sess, w_len, t_len, words, pos, tags,
                                        targets)
            current_step += 1
            step_time += (time.time() - start_time) / current_step
                             # / config.steps_per_checkpoint
            loss += step_loss / current_step
            # config.steps_per_checkpoint
            perplex = math.exp(loss) if loss < 300 else float('inf')
            print ("global step %d step-time %.2f"
                    " perplexity %.6f (loss %.6f)" %
                   (current_step, step_time, perplex, loss))
            sys.stdout.flush()
        return loss

def train_eval(config, batcher_train, batcher_test, cp_path):

    # This is the training loop.
    eval_loss = np.inf
    eval_losses = []
    while eval_loss > config.th_loss:
        _train(config, batcher_train, cp_path)

        eval_loss = _eval(config, batcher_test, cp_path)

        eval_losses.append(eval_loss)
