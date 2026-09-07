# dataloader.py
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
import numpy as np
import scipy.io
import torch

class BDGP_Animal(Dataset):
    def __init__(self, path='./data/Animal.mat'):
        data = scipy.io.loadmat(path)
        scaler = MinMaxScaler()
        self.view1 = scaler.fit_transform(data['X'][0][0].T.astype(np.float32).T)
        self.view2 = scaler.fit_transform(data['X'][0][1].T.astype(np.float32).T)
        self.view3 = scaler.fit_transform(data['X'][0][2].T.astype(np.float32).T)
        self.view4 = scaler.fit_transform(data['X'][0][3].T.astype(np.float32).T)
        self.views = [self.view1, self.view2, self.view3, self.view4]
        self.labels = data['Y']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return [torch.from_numpy(view[idx]) for view in self.views], torch.from_numpy(self.labels[idx]), torch.tensor(idx)

    def get_view(self, idx):
        return self.labels if idx == -1 else self.views[idx]


class BDGP_Scene15(Dataset):
    def __init__(self, path='./data/Scene15.mat'):
        data = scipy.io.loadmat(path)
        self.views = [MinMaxScaler().fit_transform(data['X'][0][i].astype(np.float32)) for i in range(3)]
        self.labels = data['Y'].squeeze().astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return [torch.from_numpy(view[idx]) for view in self.views],\
               torch.tensor(self.labels[idx]), torch.tensor(idx)

    def get_view(self, idx):
        return self.labels if idx == -1 else self.views[idx]


class BDGP_ALOI(Dataset):
    def __init__(self, path='./data/ALOI.mat'):
        data = scipy.io.loadmat(path)
        self.views = [MinMaxScaler().fit_transform(data['X'][i][0].astype(np.float32)) for i in range(4)]
        self.labels = data['y'].squeeze().astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return [torch.from_numpy(view[idx]) for view in self.views],\
               torch.tensor(self.labels[idx]), torch.tensor(idx)

    def get_view(self, idx):
        return self.labels if idx == -1 else self.views[idx]


class YaleDataset(Dataset):
    def __init__(self, path='./data/Yale.mat'):
        data = scipy.io.loadmat(path)
        X = data['X']
        y = data['y']

        self.views = []
        scaler = MinMaxScaler()

        for i in range(X.shape[0]):
            view = X[i][0].astype(np.float32)
            view = scaler.fit_transform(view)
            self.views.append(view)

        self.labels = y.squeeze()  # shape: (165,)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return [torch.from_numpy(view[idx]) for view in self.views], torch.tensor(self.labels[idx]), torch.tensor(idx)

    def get_view(self, idx):
        return self.labels if idx == -1 else self.views[idx]

class BDGP_Leaves(Dataset):
    def __init__(self, path='./data/100Leaves.mat'):
        data = scipy.io.loadmat(path)
        scaler = MinMaxScaler()


        self.views = []
        for i in range(data['X'].shape[0]):
            view = data['X'][i, 0]  # shape: (1600, features)
            scaled_view = scaler.fit_transform(view.astype(np.float32))
            self.views.append(scaled_view)

        self.labels = data['y'].astype(np.int64).squeeze()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return [torch.from_numpy(view[idx]) for view in self.views], torch.tensor(self.labels[idx]), torch.tensor(idx)

    def get_view(self, idx):
        return self.labels if idx == -1 else self.views[idx]


class BBCSportDataset(Dataset):
    def __init__(self, path='./data/BBCSport.mat'):
        data = scipy.io.loadmat(path)
        scaler = MinMaxScaler()


        view1 = data['X'][0][0].astype(np.float32)
        view2 = data['X'][1][0].astype(np.float32)

        self.view1 = scaler.fit_transform(view1)
        self.view2 = scaler.fit_transform(view2)
        self.views = [self.view1, self.view2]


        labels = np.squeeze(data['y'])


        if labels.min() == 1:
            labels = labels - 1

        self.labels = labels.astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return [torch.from_numpy(view[idx]) for view in self.views], torch.tensor(self.labels[idx]), torch.tensor(idx)

    def get_view(self, idx):
        return self.labels if idx == -1 else self.views[idx]


class NUS_Dataset(Dataset):
    def __init__(self, path='./data/NUS.mat'):
        data = scipy.io.loadmat(path)


        raw_views = data['X'][0]
        self.views = [
            MinMaxScaler().fit_transform(view.astype(np.float32))
            for view in raw_views
        ]


        self.labels = data['Y'].squeeze().astype(np.int64) - 1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return [torch.from_numpy(view[idx]) for view in self.views],\
            torch.tensor(self.labels[idx]),\
            torch.tensor(idx)

    def get_view(self, idx):
        return self.labels if idx == -1 else self.views[idx]




def load_data(dataset):
    if dataset == "Animal":
        dset = BDGP_Animal()
    elif dataset in ("Scene15", "Scene"):
        dset = BDGP_Scene15()
    elif dataset in ("ALOI", "Aloi"):
        dset = BDGP_ALOI()
    elif dataset in ("BBCSport", "BBCSports"):
        dset = BBCSportDataset()
    elif dataset == "100Leaves":
        dset = BDGP_Leaves()
    elif dataset == "Yale":
        dset = YaleDataset()
    elif dataset == "NUS":
        dset = NUS_Dataset()
    else:
        raise NotImplementedError(f"Unknown dataset: {dataset}")

    dims = [view.shape[1] for view in dset.views]
    view = len(dset.views)
    data_size = len(dset)
    class_num = len(np.unique(dset.labels))
    return dset, dims, view, data_size, class_num
