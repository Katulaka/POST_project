class Config(object):
   
    batch_size = 100
    word_embedding_size = 128
    tag_embedding_size = 128
    n_hidden_fw = 128
    n_hidden_bw = 128
    n_hidden_lstm = 256
    word_vocabulary_size = 100000
    tag_vocabulary_size = 100000
    num_steps = 100001
    learning_rate = 0.1
    learning_rate_decay_factor = 0.5
    max_gradient_norm = 5.0
    train_dir = '/Users/katia.patkin/Berkeley/Research/BiRNN/proc_data/'
    steps_per_checkpoint = 100
    checkpoint_path = '/Users/katia.patkin/Berkeley/Research/BiRNN/checkpoints/'
    src_dir = 'raw_data/wsj/'
    dest_dir = 'proc_data'


   # training_epochs = 10
   # num_examples    = 40
   # display_step    = 1


