import torch
from torch.utils.data import DataLoader

from FLAlgorithms.trainmodel import trainmodels


class User:

    def __init__(self, batch_size, device, id, dim,
                    feature_dim_ae, train_data,
                    data_size, learning_rate, local_epochs, class_num,
                                  ):
        self.batch_size = batch_size
        self.device = device
        self.id = id  # integer
        self.dim = dim
        self.feature_dim_ae = feature_dim_ae
        self.train_data = train_data
        self.data_size = data_size
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.class_num = class_num



        self.G_C = []
        self.G_E = []
        self.G_C_Glo = []
        self.G_E_Glo = []
        self.Z_list = []
        self.Z_tensor = []

        self.generator = torch.Generator().manual_seed(10)
        self.shuffle = False
        self.data_loader = DataLoader(train_data, batch_size=batch_size,
                                      shuffle=self.shuffle, drop_last=True,
                                      generator=self.generator)
        self.iter_data_loader = iter(self.data_loader)

        self.model = trainmodels.MyLocalModel(dim, feature_dim_ae, data_size,class_num).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate,
                                             weight_decay=0)



    def get_next_train_batch(self):
        try:
            (X, y) = next(self.iter_data_loader)
        except StopIteration:
            self.iter_data_loader = iter(self.data_loader)
            (X, y) = next(self.iter_data_loader)
        return (X.to(self.device), y.to(self.device))

