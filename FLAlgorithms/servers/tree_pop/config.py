

def get_configs():
    paras = {  # 'fusion_ways': [ 'mul', 'cat', 'max'],
        'data_name': 'data_set_maxout_text',
        'fusion_ways': ['add', 'mul', 'cat', 'max', 'avg'],
        'fused_nb_feats':128, ## 64
        'nb_view': 2,
        'pop_size': 10,
        'nb_iters': 10,
        'result_save_dir': 'NEW-MM-ENAS' + '128-10-1-video-' + 'result',
        'gpu_list': [0,1,2,3,4],
        'epochs': 5,  ## 10
        'batch_size': 64,
        'patience': 10,
        'is_remove': False,
        'crossover_rate': 0.9,
        'mutation_rate': 0.2,
        'noisy': True,
        'max_len':40,
        'image_size': {
            'w': 224, 'h': 224, 'c': 3},
        'classes': 23,
        'split_data' :[2,4,6,8,10],
        'fusion_L' : 8,
        'fusion_C' : 512,
    }
    return paras