class Config(object):

    batch_size = 2
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
    dec_timesteps = 20 #TODO
    # src_dir = '../raw_data/wsj'
    src_dir = '../gold_data'
    train_dir = 'data'
    checkpoint_path = 'checkpoints'
