import torch
import numpy as np
import argparse
import random
from data import dataloader
from FLAlgorithms.servers import serveravg
import multiprocessing as mp



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda:{}".format(5) if torch.cuda.is_available() else "cpu")
print('device:', device)

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default='NUS', choices=["Animal", "Scene", "Aloi", "BBCSpor"
                                                                                    "ts", "100Leaves", "Yale", "NUS"])

parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--learning_rate", type=float, default=0.0001)

parser.add_argument("--client_cluster", type=float, default=0.1)
parser.add_argument("--client_orth", type=float, default=0.01)
parser.add_argument("--server_contrastive", type=float, default=0.15)#
parser.add_argument("--server_gram", type=float, default=0.001)

parser.add_argument("--feature_dim_ae", default=512, type=int)
parser.add_argument("--num_ae_epochs", type=int, default=20)
parser.add_argument("--num_global_iters", type=int, default=10)  #########10
parser.add_argument("--local_epochs", type=int, default=5)
parser.add_argument("--cl_epochs", type=int, default=5)


args = parser.parse_args()


def main(dataset, feature_dim_ae, learning_rate,
         num_ae_epochs, num_glob_iters,
         local_epochs, cl_epochs, batch_size,client_cluster,client_orth,server_contrastive,server_gram):



    for i in range(1, 2):
        seed = 10
        setup_seed(seed)

        print("---------------Running time:------------", i)
        data, dims, view, data_size, class_num = dataloader.load_data(dataset)
        data_size = int(data_size / batch_size) * batch_size



        print(f'The number of views: {view}')


        server = serveravg.FedAvg(batch_size, device, dataset, learning_rate, feature_dim_ae,
                                  num_ae_epochs, num_glob_iters, local_epochs, cl_epochs,
                                  data, dims, view, data_size, class_num,
                                  client_cluster,client_orth,server_contrastive,server_gram)

        server.train()



if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main(
        dataset=args.dataset,
        feature_dim_ae=args.feature_dim_ae,
        learning_rate=args.learning_rate,
        num_glob_iters=args.num_global_iters,
        local_epochs=args.local_epochs,
        num_ae_epochs=args.num_ae_epochs,
        cl_epochs=args.cl_epochs,
        batch_size=args.batch_size,
        client_cluster=args.client_cluster,
        client_orth =args.client_orth,
        server_contrastive =args.server_contrastive,
        server_gram =args.server_gram )




