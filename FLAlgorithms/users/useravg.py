import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from FLAlgorithms.users.userbase import User





class ClusterLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(self, logits):
        prob = F.softmax(logits, dim=1)
        target = prob / prob.sum(dim=0, keepdim=True).sqrt()
        target = target / target.sum(dim=1, keepdim=True)
        return self.kl_loss(prob.log(), target.detach())

class UserAVG(User):
    def __init__(self, batch_size, device, id, dim,
                    feature_dim_ae, train_data,
                    data_size, learning_rate, local_epochs, class_num,client_cluster,client_orth
                                  ):
        super().__init__(batch_size, device, id, dim,
                    feature_dim_ae, train_data,
                    data_size, learning_rate, local_epochs, class_num,
                                  )

        self.cluster_loss_fn = ClusterLoss()
        self.client_cluster=client_cluster
        self.client_orth =client_orth

    def train_ae(self, epoch):
        criterion = torch.nn.MSELoss()
        loss = 0
        self.local_list = []

        generator = torch.Generator().manual_seed(10)
        self.data_loader = DataLoader(self.train_data, batch_size=self.batch_size,
                                      shuffle=self.shuffle, drop_last=True,
                                      generator=generator)
        for batch_idx, (xs, _) in enumerate(self.data_loader):
            xs = xs.to(self.device)
            self.optimizer.zero_grad()
            Z, X_Pre, E_C, E_E,Z_c,Z_e = self.model(xs)  # xrs recostruction xv

            loss_list = []
            recon_loss = criterion(xs, X_Pre)
            loss_list.append(recon_loss)


            loss = sum(loss_list)
            loss.backward()
            self.optimizer.step()


        if epoch % 10 == 0:
            print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(loss))

    def train(self, glob_iter, num_glob_iters):

        Lc_loss = 0
        epoch = 1
        print('user{}:'.format(self.id))
        for epoch in range(1, self.local_epochs + 1):
            Lc_loss = 0

            self.G_C = []
            self.G_E = []
            generator = torch.Generator().manual_seed(10)
            self.data_loader = DataLoader(self.train_data, batch_size=self.batch_size,
                                          shuffle=self.shuffle, drop_last=True,
                                          generator=generator)
            self.iter_data_loader = iter(self.data_loader)
            for epoch_n in range(len(self.data_loader)):
                self.optimizer.zero_grad()
                criterion = torch.nn.MSELoss()
                loss_list = []
                X, y = self.get_next_train_batch()

                Z, X_Pre, G_C, G_E,Z_c, Z_e = self.model(X)              ## NOTE: G_C / G_E are now evidence vectors, not graph rows
                local=G_C+G_E

                cluster_loss = self.cluster_loss_fn(G_C) + self.cluster_loss_fn(G_E)
                loss_list.append(self.client_cluster * cluster_loss)

                Z_c_norm = F.normalize(Z_c, dim=1)
                Z_e_norm = F.normalize(Z_e, dim=1)
                orth_loss = torch.mean(torch.sum(Z_c_norm * Z_e_norm, dim=1) ** 2)
                loss_list.append(self.client_orth * orth_loss)
                loss_list.append(1.0 * criterion(X, X_Pre))

                if glob_iter == 0:
                    G_C_Glo = G_C
                else:
                    G_C_Glo = self.G_C_Glo[epoch_n].to(self.device)     #(batchsize,nclass)

                S = G_C_Glo + G_E
                align_loss = criterion(S, local)
                loss_list.append(0.15 * align_loss)


                self.G_C.append(G_C_Glo.cpu().detach())
                self.G_E.append(G_E.cpu().detach())


                loss = sum(loss_list)

                Lc_loss = Lc_loss + loss

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

        print(Lc_loss.item())



        return Lc_loss / epoch




