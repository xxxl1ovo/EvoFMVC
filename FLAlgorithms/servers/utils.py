
import torch
import torch.nn.functional as F
import os

from FLAlgorithms.servers import config
from FLAlgorithms.servers.tree_pop import utilss

paras = config.get_configs()
data_name = paras['data_name']
idxx = 0

def get_nb_view_by_individal_code(code):
    nb_view = (len(code) + 1) // 2
    return nb_view


def write_result_file(str, fn=None):
    if fn is None:
        fn = utilss.get_fitness_file()
    dirname = os.path.dirname(fn)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(fn, 'a+') as f:
        f.write(str)
        f.write('\n')
        f.flush()


def load_result(result_fn=None):
    if result_fn is None:
        result_fn = utilss.get_fitness_file()
    shared_code_acc = {}
    shared_code_acc_set = set()
    with open(result_fn) as f:
        for item in f.readlines():
            items = item.strip().split(',')
            if items[0] not in shared_code_acc_set:
                shared_code_acc[items[0]] = float(items[1])
    return shared_code_acc


def list2str(list1):
    return '-'.join([str(i) for i in list1])

def sign_sqrt(x):
    return torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-10)

def l2_norm(x):
    return F.normalize(x, p=2, dim=-1)
