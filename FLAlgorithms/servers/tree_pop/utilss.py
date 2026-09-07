import os
import re

import torch
from torch.utils.data import DataLoader, TensorDataset

from FLAlgorithms.servers.tree_pop import config


paras = config.get_configs()
data_name = paras['data_name']
idxx = 0
FITNESS_FILENAME = 'fitness.csv'
HISTORY_FILENAME = 'population_history.csv'
INDIVIDUAL_RESULT_DIR = 'individuals'
_active_result_save_dir = None


def sanitize_filename(value):
    value = str(value).strip()
    value = re.sub(r'[^A-Za-z0-9_.-]+', '_', value)
    return value.strip('._') or 'run'


def get_result_save_dir(dataset=None):
    if 'AAAI_EVO_RESULT_DIR' in os.environ:
        return os.environ['AAAI_EVO_RESULT_DIR']
    run_name = os.environ.get('AAAI_EVO_RUN_NAME')
    if run_name is None:
        run_name = dataset if dataset is not None else paras['result_save_dir']
    return os.path.join('results', sanitize_filename(run_name))


def set_active_result_save_dir(result_save_dir):
    global _active_result_save_dir
    _active_result_save_dir = result_save_dir


def get_active_result_save_dir():
    return _active_result_save_dir or get_result_save_dir()


def get_fitness_file(result_save_dir=None):
    return os.path.join(result_save_dir or get_active_result_save_dir(), FITNESS_FILENAME)


def get_history_file(result_save_dir=None):
    return os.path.join(result_save_dir or get_active_result_save_dir(), HISTORY_FILENAME)


def get_individual_result_dir(result_save_dir=None):
    return os.path.join(result_save_dir or get_active_result_save_dir(), INDIVIDUAL_RESULT_DIR)


def get_individual_result_file(code, result_save_dir=None):
    code_str = '+'.join([str(i) for i in code])
    return os.path.join(get_individual_result_dir(result_save_dir), f'{code_str}.csv')


def get_nb_view_by_individal_code(code):
    return (len(code) + 1) // 2


def write_result_file(line, fn=None):
    if fn is None:
        fn = get_fitness_file()
    dirname = os.path.dirname(fn)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(fn, 'a+') as f:
        f.write(line)
        f.write('\n')
        f.flush()


def load_result(result_fn=None):
    if result_fn is None:
        result_fn = get_fitness_file()
    shared_code_acc = {}
    with open(result_fn) as f:
        for item in f.readlines():
            items = item.strip().split(',')
            if len(items) < 2 or items[0] in shared_code_acc:
                continue
            shared_code_acc[items[0]] = float(items[1])
    return shared_code_acc


def list2str(list1):
    return '-'.join([str(i) for i in list1])


def sign_sqrt(x):
    return torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-10)


def l2_norm(x):
    return torch.nn.functional.normalize(x, p=2, dim=-1)


def create_data_loader(batch_size, shuffle, data, labels=None):
    tensors = list(data)
    if labels is not None:
        tensors.append(torch.tensor(labels))
    dataset = TensorDataset(*tensors)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataset, data_loader
