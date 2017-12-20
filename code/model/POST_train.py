import math
import os
import sys
import time
import tensorflow as tf
import numpy as np

from POST_main import get_model

def save_checkpoints(sess, model, checkpoint_path, cp_path):
    ckpt_path = os.path.join(checkpoint_path, cp_path)
    if not os.path.exists(checkpoint_path):
        try:
            os.makedirs(os.path.abspath(checkpoint_path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    model.saver.save(sess, ckpt_path, global_step=model.global_step)

def _train(config, batcher):

    step_time, loss = 0.0, 0.0
    train_graph = tf.Graph()
    with tf.Session(graph = train_graph) as sess:
        model = get_model(sess, config, train_graph, 'train')
        current_step =  model.pos_step.eval() if config.pos else model.suptag_step.eval()
        for i in range(config.num_epochs):
            for bv in batcher.get_permute_batch():
                start_time = time.time()
                w_len, t_len, words, pos, _, tags, targets = batcher.process(bv)
                if not config.pos:
                    pos_pred = model.pos_decode(sess, words, w_len)
                step_loss, _, _  = model.step(sess, w_len, t_len, words, pos,
                                                tags, targets)
                step_time += (time.time() - start_time)\
                                 / config.steps_per_checkpoint
                loss += step_loss / config.steps_per_checkpoint
                current_step += 1
                # Once in a while, we save checkpoint, print statistics
                if current_step % config.steps_per_checkpoint == 0:
                    # Print statistics for the previous epoch.
                    perplex = math.exp(loss) if loss < 300 else float('inf')
                    print ("step %d learning rate %f step-time %.2f"
                           " perplexity %.6f (loss %.6f)" %
                           (current_step,
                           model.learning_rate.eval(),
                           step_time, perplex, loss))
                    if current_step == 20:
                        sess.run(model.learning_rate_decay_op)

                    # Save checkpoint and zero timer and loss.
                    save_checkpoints(sess, model, config.checkpoint_path,
                                        config.cp_dir)
                    step_time, loss = 0.0, 0.0
                    sys.stdout.flush()

def _eval(config, batcher):
    step_time, tot_loss = 0.0, 0.0
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
            step_time = (time.time() - start_time) / current_step
            tot_loss += step_loss
            loss = avg_loss = tot_loss / current_step
            perplex = math.exp(loss) if loss < 300 else float('inf')
            print ("global step %d step-time %.2f"
                    " perplexity %.6f (loss %.6f)" %
                   (current_step, step_time, perplex, loss))
            sys.stdout.flush()
        return loss

def train_eval(config, batcher_train, batcher_test):

    # This is the training loop.
    eval_loss = np.inf
    eval_losses = []
    while eval_loss > config.th_loss:
        _train(config, batcher_train)

        if not config.pos: #TODO maybe add eval for the training of POS?
            eval_loss = _eval(config, batcher_test)
            eval_losses.append(eval_loss)
