import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, v_measure_score, adjusted_rand_score
from sklearn.metrics import silhouette_score
from FLAlgorithms.trainmodel import trainmodels

class Server:
    def __init__(self, batch_size, device, dataset, learning_rate, feature_dim_ae, num_ae_epochs,
                                  num_glob_iters, local_epochs, cl_epochs,
                                  data, dims, view, data_size, class_num,
                                  ):

        self.batch_size = batch_size
        self.device = device
        self.dataset = dataset
        self.learning_rate = learning_rate
        self.feature_dim_ae = feature_dim_ae
        self.num_ae_epochs = num_ae_epochs
        self.num_glob_iters = num_glob_iters
        self.local_epochs = local_epochs
        self.cl_epochs = cl_epochs


        self.data = data
        self.dims = dims
        self.num_users = view
        self.data_size = data_size
        self.class_num = class_num

        self.G_Cs = []
        self.G_Es = []

        self.users = []
        self.model = trainmodels.MyServerModel(data_size,class_num).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                             lr=learning_rate * 1,
                                             weight_decay=0)

    def aggregate_parameters(self):
        assert (self.users is not None and len(self.users) > 0)

        self.G_Cs.clear()
        self.G_Es.clear()
        for user in self.users:
            self.G_Cs.append(user.G_C)
            self.G_Es.append(user.G_E)

    def cluster_acc(self, y_true, y_pred):
        y_true = y_true.astype(np.int64)
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        u = linear_sum_assignment(w.max() - w)
        ind = np.concatenate([u[0].reshape(u[0].shape[0], 1), u[1].reshape([u[0].shape[0], 1])], axis=1)
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

    def purity(self, y_true, y_pred):
        y_voted_labels = np.zeros(y_true.shape)
        labels = np.unique(y_true)
        ordered_labels = np.arange(labels.shape[0])
        for k in range(labels.shape[0]):
            y_true[y_true == labels[k]] = ordered_labels[k]
        labels = np.unique(y_true)
        bins = np.concatenate((labels, [np.max(labels) + 1]), axis=0)

        for cluster in np.unique(y_pred):
            hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
            winner = np.argmax(hist)
            y_voted_labels[y_pred == cluster] = winner

        return accuracy_score(y_true, y_voted_labels)



    def evaluate(self, label, pred, data=None):
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(label, torch.Tensor):
            label = label.cpu().numpy()
        if data is not None and isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        pred = np.nan_to_num(pred, nan=0)
        label = np.nan_to_num(label, nan=0)
        if data is not None:
            data = np.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6)

        silhouette = 0.0
        if data is not None:
            unique_labels = np.unique(pred)
            if len(unique_labels) > 1:
                try:
                    silhouette = silhouette_score(data, pred)
                except Exception as e:
                    print("[WARN] Failed to compute silhouette:", e)
                    silhouette = 0.0
            else:
                print("[WARN] Only one cluster found, silhouette set to 0.")

        nmi = v_measure_score(label, pred)
        ari = adjusted_rand_score(label, pred)
        acc = self.cluster_acc(label, pred)
        pur = self.purity(label, pred)

        return nmi, ari, acc, pur, silhouette

