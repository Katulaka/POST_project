import os

class Config(object):

    batch_size = 32
    word_embedding_size = 128
    tag_embedding_size = 128
    n_hidden_fw = 128
    n_hidden_bw = 128
    n_hidden_lstm = 256
    word_vocabulary_size = 48368
    tag_vocabulary_size = 11390
    learning_rate = 0.1
    learning_rate_decay_factor = 0.1
    steps_per_checkpoint = 10
    beam_size = 5
    dec_timesteps = 20

    checkpoint_path = 'checkpoints'

    w_attn = True
    #use regular loss or modified with "covarience" relation
    reg_loss = False #TODO
    no_val_gap = False

    #training loss threshold to stop
    th_loss = 0.1
    #training num epochs before evaluting the dev loss
    num_epochs = 1

    #Astar search time-out
    time_out = 1000.
    #Astar search number of goals
    num_goals = 1
    # use multi-processing when doing astar search
    multi_processing = False

    #beam search or greedy beam search
    greedy = False

    src_dir = '../gold_data'
    ds_dir = 'data'
    ds_fname = 'data.txt'
    gold_fname = 'gold'
    gold_file = os.path.join(os.getcwd(), ds_dir, gold_fname)
    ds_file = os.path.join(os.getcwd(), ds_dir, ds_fname)
