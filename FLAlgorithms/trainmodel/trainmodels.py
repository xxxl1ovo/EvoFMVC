import torch
import torch.nn as nn
import torch.nn.functional as F
from FLAlgorithms.servers import config

paras = config.get_configs()
fusion_ways = paras['fusion_ways']
fused_nb_feats = paras['fused_nb_feats']
classes = paras['classes']



def sign_sqrt(x):
    return torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-10)

def l2_norm(x):
    return F.normalize(x, p=2, dim=-1)


def fusion(x1, x2, way, fc=None):
    if way == 'add':
        return x1 + x2
    elif way == 'mul':
        return x1 * x2
    elif way == 'max':
        return torch.max(x1, x2)
    elif way == 'avg':
        return (x1 + x2) / 2
    else:
        raise ValueError(f"Unknown fusion method: {way}")


class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 1000),
            nn.ReLU(),
            nn.Linear(1000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, feature_dim, output_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, output_dim)
        )

    def forward(self, x):
        return self.decoder(x)





class MyLocalModel(nn.Module):
    def __init__(self, feature_in, feature_dim_ae, data_size,n_clusters):
        super(MyLocalModel, self).__init__()
        self.encoder1 = Encoder(feature_in, feature_dim_ae)
        self.encoder2 = Encoder(feature_in, feature_dim_ae)
        self.decoder = Decoder(feature_dim_ae, feature_in)
        self.g_c = nn.Sequential(
            nn.Linear(feature_dim_ae, data_size)
        )
        self.g_e = nn.Sequential(
            nn.Linear(feature_dim_ae, data_size)
        )
        self.e_c = nn.Sequential(
            nn.Linear(feature_dim_ae, n_clusters),
            nn.Softplus()
        )
        self.e_e = nn.Sequential(
            nn.Linear(feature_dim_ae, n_clusters),
            nn.Softplus()
        )
    def forward(self, data):
        X = data
        Z_c = self.encoder1(X)
        Z_e = self.encoder2(X)
        Z_c = F.normalize(Z_c, p=2, dim=-1)
        Z_e = F.normalize(Z_e, p=2, dim=-1)
        Z = Z_c + Z_e

        X_Pre = self.decoder(Z)
        E_C = self.e_c(Z_c)+1
        E_E = self.e_e(Z_e)+1 # [N, C]

        return Z, X_Pre, E_C, E_E,Z_c,Z_e






class MyServerModel(nn.Module):
    def __init__(self, data_size, n_clusters):
        super(MyServerModel, self).__init__()

        self.n_clusters = n_clusters
        self.n_views = 2

        self.weight_vector = nn.Sequential(
            nn.Linear(n_clusters * self.n_views, 128),
            nn.ReLU(),
            nn.Linear(128, self.n_views * n_clusters)
        )

    def forward(self, evidences_c, evidences_e):
        """
        evidences_c, evidences_e: List[Tensor], each of shape [batch, n_clusters]
        """

        fused_cs = []
        fused_es = []

        for i in range(len(evidences_c)):
            ec = evidences_c[i]  # shape: [B, C]
            ee = evidences_e[i]  # shape: [B, C]

            input_pair = torch.cat([ec, ee], dim=-1)  # [B, 2C]

            raw_weights = self.weight_vector(input_pair)  # [B, 2C]
            raw_weights = raw_weights.view(-1, self.n_views, self.n_clusters)  # [B, 2, C]

            weights = F.softmax(raw_weights, dim=1)  # [B, 2, C]

            w_c = weights[:, 0, :]
            w_e = weights[:, 1, :]

            fused = w_c * ec + w_e * ee  # shape: [B, C]

            fused_cs.append(fused)

        return fused_es, fused_cs
