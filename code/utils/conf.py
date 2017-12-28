import os

class Config(object):

    class ModelParms(object):
        batch_size = 32
        #embedding size
        dim_word = 128
        dim_tag = 64
        dim_char = 32
        dim_pos = 64
        #NN dims
        char_hidden = 32
        pos_hidden = 128
        word_hidden = 128
        tag_hidden = 256
        #vocab size
        nwords = 0
        ntags = 0
        nchars = 0
        #learning rate
        lr = 0.1
        lr_decay_factor = 0.1
        #use attention
        attn = True
        #use modified loss with "covarience" relation
        comb_loss = False

    steps_per_checkpoint = 10
    beam_size = 5
    dec_timesteps = 20

    checkpoint_path = 'checkpoints'

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
