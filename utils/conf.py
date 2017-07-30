class Config(object):

    batch_size = 2
    word_embedding_size = 128
    tag_embedding_size = 128
    n_hidden_fw = 128
    n_hidden_bw = 128
    n_hidden_lstm = 256
    word_vocabulary_size = 48368
    tag_vocabulary_size = 11390
    num_steps = 100001
    learning_rate = 0.1
    learning_rate_decay_factor = 0.5
    max_gradient_norm = 5.0
    steps_per_checkpoint = 10
    train_dir = 'proc_data/'
    checkpoint_path = 'checkpoints/'
    src_dir = 'raw_data/wsj'
    dest_dir = 'proc_data'
    beam_size = 5
    dec_timesteps = 20 #TODO


   # training_epochs = 10
   # num_examples    = 40
   # display_step    = 1
