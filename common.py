class Config:

    minibatch_size = 50
    keep_prob = 0.8
    nr_epoch = 30
    
    embedding_size = 200
    learning_rate_base = 1e-3
    learning_rate_decay = 0.96
    show_interval = 5

    # transformer
    rand_seed = None
    is_mask = True
    is_layer_norm = True
    is_positional = True
    regularizer = 0.0015
    stack_num = 3
    attention_dtype = 'dot'

    # fc layer
    layer1_size = 512
    # datasets SST2
    num_class  = 2
    vocab_size = 16789
    max_sentence_len = 35
    data_path_root   = r'./data/SST2'
cf = Config()


