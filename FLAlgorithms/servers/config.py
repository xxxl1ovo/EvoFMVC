
def get_configs():
    paras = {  # 'fusion_ways': [ 'mul', 'cat', 'max'],
        'data_name': 'animal_result',
        'fusion_ways': ['add', 'mul','cat', 'max', 'avg'],
        'fused_nb_feats': 1024,
        'nb_view': 5,
        'pop_size': 28,
        'nb_iters': 20,
        'result_save_dir': 'EDF-True' + '-128-5' + 'result',
        'gpu_list': [0],
        'epochs': 100,
        'batch_size': 16,
        'patience': 10,
        'is_remove': False,
        'crossover_rate': 0.9,
        'mutation_rate': 0.2,
        'noisy': True,
        'max_len': 40,
        'image_size': {
            'w': 230, 'h': 230, 'c': 3},
        'classes': 200,
    }
    return paras
