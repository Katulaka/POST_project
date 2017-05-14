class config(object):
   
    batch_size             = 100 
    word_embedding_size    = 128 
    tag_embedding_size     = 128 
    n_hidden_fw            = 128
    n_hidden_bw            = 128
    n_hidden_lstm          = 256
    word_vocabulary_size   = 100000
    tag_vocabulary_size    = 100000
    num_steps              = 100001
    learning_rate          = 0.1
    learning_rate_decay_factor = #TODO

    train_dir               = #TODO
    steps_per_checkpoint    = #TODO
    checkpoint_path         = #TODO

    training_epochs = 10
    num_examples    = 40
    display_step    = 1


