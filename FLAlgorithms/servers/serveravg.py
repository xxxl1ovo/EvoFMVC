import csv
import os
from itertools import combinations

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import FLAlgorithms.trainmodel.trainmodels
from FLAlgorithms.servers import config, utils
from FLAlgorithms.servers.config import get_configs
from FLAlgorithms.servers.serverbase import Server
from FLAlgorithms.servers.tree_pop import gen_offspring_tree
from FLAlgorithms.servers.tree_pop import population_init
from FLAlgorithms.servers.tree_pop import tree_to_strlist
from FLAlgorithms.servers.tree_pop import utilss
from FLAlgorithms.servers.tree_pop import utils_tree
from FLAlgorithms.users.useravg import UserAVG


paras = config.get_configs()


def list2str_tree(list1):
    return '+'.join([str(i) for i in list1])


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        pos = torch.sum(z1 * z2, dim=1) / self.temperature
        z = torch.cat([z1, z2], dim=0)
        sim = torch.matmul(z1, z.T) / self.temperature
        batch_size = z1.size(0)
        labels = torch.arange(batch_size, device=z.device)
        labels = torch.cat([labels, labels], dim=0)
        mask = labels.unsqueeze(0) != labels.unsqueeze(1)
        neg = sim.masked_select(mask[:batch_size]).view(batch_size, -1)
        loss = -pos + torch.logsumexp(neg, dim=1)
        return loss.mean()


def gram_diversity_loss(evidence_list):
    loss = 0.0
    for i in range(len(evidence_list)):
        for j in range(i + 1, len(evidence_list)):
            gi = torch.matmul(evidence_list[i], evidence_list[i].T)
            gj = torch.matmul(evidence_list[j], evidence_list[j].T)
            loss += torch.sum(gi * gj)
    return loss


def dirichlet_kl_divergence(alpha_pred, alpha_target):
    s_pred = torch.sum(alpha_pred, dim=1, keepdim=True)
    s_target = torch.sum(alpha_target, dim=1, keepdim=True)
    return (
        torch.lgamma(s_pred) - torch.lgamma(s_target)
        - torch.sum(torch.lgamma(alpha_pred) - torch.lgamma(alpha_target), dim=1, keepdim=True)
        + torch.sum((alpha_pred - alpha_target) * (torch.digamma(alpha_pred) - torch.digamma(s_pred)), dim=1, keepdim=True)
    ).mean()


class FedAvg(Server):
    def __init__(self, batch_size, device, dataset, learning_rate, feature_dim_ae, num_ae_epochs,
                 num_glob_iters, local_epochs, cl_epochs,
                 data, dims, view, data_size, class_num,
                 client_cluster, client_orth, server_contrastive, server_gram):
        super().__init__(batch_size, device, dataset, learning_rate, feature_dim_ae, num_ae_epochs,
                         num_glob_iters, local_epochs, cl_epochs,
                         data, dims, view, data_size, class_num)

        configs = get_configs()
        self.fusion_ways = configs['fusion_ways']
        self.server_contrastive_loss = ContrastiveLoss(temperature=0.2)
        self.client_cluster = client_cluster
        self.client_orth = client_orth
        self.server_contrastive = server_contrastive
        self.server_gram = server_gram
        self.result_save_dir = utilss.get_result_save_dir(dataset)
        utilss.set_active_result_save_dir(self.result_save_dir)
        self.fitness_file = utilss.get_fitness_file(self.result_save_dir)
        self.history_file = utilss.get_history_file(self.result_save_dir)
        self.individual_result_dir = utilss.get_individual_result_dir(self.result_save_dir)
        self.linear_cat = nn.Linear(class_num * 2, class_num).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.linear_cat.parameters()),
            lr=self.learning_rate,
            weight_decay=0
        )
        self.fusion_x = None
        self.server_dirichlet = 0.0

        for i in range(self.num_users):
            train_data = [data.get_view(i), data.get_view(-1)]
            train_data = list(zip(*train_data))
            user = UserAVG(batch_size, device, i, dims[i],
                           feature_dim_ae, train_data,
                           data_size, learning_rate, local_epochs, class_num,
                           client_cluster, client_orth)
            self.users.append(user)

        print("Number of users :", self.num_users)
        print("Finished creating FedAvg server.")

    def inference_graph_train(self, evidence, y):
        evidence = torch.nan_to_num(evidence, nan=0.0)
        kmeans = KMeans(n_clusters=self.class_num, n_init=5, random_state=0)
        total_pred = kmeans.fit_predict(evidence.cpu().detach().numpy())
        labels_vector = np.array(y).reshape(len(total_pred))
        return total_pred, labels_vector

    def _reset_server_fusion_model(self):
        torch.manual_seed(10)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(10)
        self.model = FLAlgorithms.trainmodel.trainmodels.MyServerModel(self.data_size, self.class_num).to(self.device)
        self.linear_cat = nn.Linear(self.class_num * 2, self.class_num).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.linear_cat.parameters()),
            lr=self.learning_rate,
            weight_decay=0
        )

    def _clone_evidence_batches(self, evidence_batches):
        return [[batch.detach().cpu().clone() for batch in view_batches] for view_batches in evidence_batches]

    def _fusion_fx(self, x1, x2, way):
        if way == 'add':
            return x1 + x2
        if way == 'mul':
            return x1 * x2
        if way == 'cat':
            return self.linear_cat(torch.cat((x1.to(self.device), x2.to(self.device)), dim=-1))
        if way == 'max':
            return torch.max(x1, x2)
        if way == 'avg':
            return (x1 + x2) / 2.0
        raise ValueError(f"Unknown fusion operator: {way}")

    def _fuse_views(self, individual_code, view_tensors):
        individual_code_tree, nb_view = tree_to_strlist.viewfusion(individual_code)
        view_train_xx = [view_tensors[int(view_id)] for view_id in individual_code_tree[:nb_view]]
        _, nb_views = utils_tree.viewfusion(individual_code)
        if nb_views == 1:
            return view_train_xx[0]

        cnt = 0
        vsize = nb_views
        listview = []
        fusion_x = None
        for item in individual_code:
            if item[0] != '-':
                listview.append(cnt)
                cnt += 1
            else:
                e1 = listview.pop()
                e2 = listview.pop()
                operator_id = int(item[1:])
                fusion_x = self._fusion_fx(view_train_xx[e1], view_train_xx[e2], self.fusion_ways[operator_id])
                view_train_xx.append(fusion_x)
                listview.append(vsize)
                vsize += 1

        if fusion_x is None:
            raise ValueError(f"Invalid fusion code: {individual_code}")
        return fusion_x

    def _get_used_view_ids(self, individual_code):
        individual_code_tree, nb_view = tree_to_strlist.viewfusion(individual_code)
        return [int(view_id) for view_id in individual_code_tree[:nb_view]]

    def _get_ordered_labels(self):
        labels = []
        generator = torch.Generator().manual_seed(10)
        data_loader = DataLoader(
            self.users[0].train_data,
            batch_size=self.batch_size,
            shuffle=self.users[0].shuffle,
            drop_last=True,
            generator=generator,
        )
        for _, ys in data_loader:
            labels.append(ys)
        return np.concatenate(labels)

    def _ensure_individual_log(self, file_path):
        if not os.path.exists(file_path):
            with open(file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['round', 'fusion_code', 'train_loss', 'ACC', 'NMI', 'ARI', 'PUR', 'silhouette'])

    def _evaluate_individual(self, individual_code, glob_iter, base_G_Cs, base_G_Es):
        self._reset_server_fusion_model()
        candidate_G_Cs = self._clone_evidence_batches(base_G_Cs)
        candidate_G_Es = self._clone_evidence_batches(base_G_Es)
        code_str = list2str_tree(individual_code)
        file_path = utilss.get_individual_result_file(individual_code, self.result_save_dir)
        self._ensure_individual_log(file_path)
        used_view_ids = self._get_used_view_ids(individual_code)

        for _ in range(self.cl_epochs):
            for batch_idx in range(len(candidate_G_Cs[0])):
                cs_gpu = torch.stack([candidate_G_Cs[k][batch_idx].to(self.device) for k in range(len(candidate_G_Cs))])
                es_gpu = torch.stack([candidate_G_Es[k][batch_idx].to(self.device) for k in range(len(candidate_G_Es))])
                self.optimizer.zero_grad()
                _, cs_t = self.model(cs_gpu, es_gpu)

                loss_list = []
                for v, w in combinations(used_view_ids, 2):
                    contrastive_loss = self.server_contrastive_loss(cs_t[v], cs_t[w])
                    loss_list.append(self.server_contrastive * contrastive_loss)

                if self.server_dirichlet > 0:
                    for v in used_view_ids:
                        if v >= len(cs_t):
                            print(f"[skip dirichlet loss] view {v} out of range for fused_cs")
                            continue
                        dirichlet_loss = dirichlet_kl_divergence(cs_t[v], cs_gpu[v])
                        loss_list.append(self.server_dirichlet * dirichlet_loss)

                if self.server_gram > 0:
                    gram_loss = gram_diversity_loss([es_gpu[v] for v in used_view_ids])
                    loss_list.append(self.server_gram * gram_loss)

                if loss_list:
                    loss_avg = sum(loss_list) / len(loss_list)
                    loss_avg.backward()
                    self.optimizer.step()

                fusion_x = self._fuse_views(individual_code, cs_t)
                for k in range(len(candidate_G_Cs)):
                    candidate_G_Cs[k][batch_idx] = fusion_x.cpu().detach()

        final_temp = []
        with torch.no_grad():
            for batch_idx in range(len(candidate_G_Cs[0])):
                cs_gpu = torch.stack([candidate_G_Cs[k][batch_idx].to(self.device) for k in range(len(candidate_G_Cs))])
                es_gpu = torch.stack([candidate_G_Es[k][batch_idx].to(self.device) for k in range(len(candidate_G_Es))])
                _, cs_t = self.model(cs_gpu, es_gpu)
                fusion_x = self._fuse_views(individual_code, cs_t)
                final_temp.append(fusion_x.cpu().detach())

        final_temp = torch.cat(final_temp, dim=0)
        y1 = self._get_ordered_labels()
        total_pred, labels_vector = self.inference_graph_train(final_temp, y1)
        nmi, ari, acc, pur, silhouette = self.evaluate(labels_vector, total_pred, final_temp)

        print("=== Individual Evaluation [Round {}] ===".format(glob_iter))
        print("ACC = {:.4f} NMI = {:.4f} ARI = {:.4f} PUR = {:.4f}, Silhouette = {:.4f}".format(
            acc, nmi, ari, pur, silhouette
        ))

        with open(file_path, mode='a+', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([glob_iter, code_str, 0, acc, nmi, ari, pur, silhouette])
        utils.write_result_file(','.join([code_str, str(silhouette)]), fn=self.fitness_file)
        return silhouette, candidate_G_Cs

    def _set_global_evidence(self, global_G_Cs):
        self.G_Cs = self._clone_evidence_batches(global_G_Cs)
        for i, user in enumerate(self.users):
            user.G_C_Glo = [batch.detach().cpu().clone() for batch in self.G_Cs[i]]

    def train(self):
        for user in self.users:
            epoch = 1
            print("-------------user", user.id, ": start autoencoder pretraining-------------")
            while epoch <= self.num_ae_epochs:
                user.train_ae(epoch)
                epoch += 1

        os.makedirs(self.result_save_dir, exist_ok=True)
        os.makedirs(self.individual_result_dir, exist_ok=True)
        for result_file in (self.fitness_file, self.history_file):
            open(result_file, 'w').close()

        P_t = population_init.generate_population_tree(views=len(self.users), pop_size=10, verbose=0, flat=10)
        for code in P_t:
            utils.write_result_file(','.join([str(0), list2str_tree(code)]), fn=self.history_file)

        for glob_iter in range(self.num_glob_iters):
            print("-------------Communication round: ", glob_iter, " -------------")
            for user in self.users:
                user.train(glob_iter, self.num_glob_iters)
            self.aggregate_parameters()

            base_G_Cs = self._clone_evidence_batches(self.G_Cs)
            base_G_Es = self._clone_evidence_batches(self.G_Es)
            open(self.fitness_file, 'w').close()
            evaluated_scores = {}
            evaluated_global_evidence = {}

            def evaluate_population(population):
                for individual_code in population:
                    code_str = list2str_tree(individual_code)
                    if code_str in evaluated_scores:
                        continue
                    silhouette, global_evidence = self._evaluate_individual(
                        individual_code,
                        glob_iter,
                        base_G_Cs,
                        base_G_Es
                    )
                    evaluated_scores[code_str] = silhouette
                    evaluated_global_evidence[code_str] = global_evidence

            evaluate_population(P_t)
            for evo_iter in tqdm(range(paras['nb_iters'])):
                print(f'===================EA {evo_iter + 1}/', paras['nb_iters'])
                Q_t = gen_offspring_tree.gen_offspring(P_t, evo_iter, views=len(self.users))
                evaluate_population(Q_t)
                P_t = gen_offspring_tree.selection(P_t, Q_t)
                history_step = glob_iter * (paras['nb_iters'] + 1) + evo_iter + 1
                for code in P_t:
                    utils.write_result_file(','.join([str(history_step), list2str_tree(code)]), fn=self.history_file)

            best_code_str = max(evaluated_scores, key=evaluated_scores.get)
            self._set_global_evidence(evaluated_global_evidence[best_code_str])
            print("Round {} best fusion: {}, silhouette={:.4f}".format(
                glob_iter,
                best_code_str,
                evaluated_scores[best_code_str]
            ))
