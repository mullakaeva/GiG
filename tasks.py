import pytorch_lightning as pl
import torchmetrics
from models import GiG
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_adj
from common_utils import SoftHistogram, compute_clustering_coeff,\
    MultiTaskBCELoss, mask_nans
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
import seaborn as sns
import os.path
import shutil
import sklearn
from contextlib import suppress

losses = nn.ModuleDict({
                'BCEWithLogitsLoss': nn.BCEWithLogitsLoss(),
                'CrossEntropyLoss': nn.CrossEntropyLoss(),
                'MultiTaskBCE': MultiTaskBCELoss()
        })

class LglKlTask(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.automatic_optimization = False

        self.config = config
        self.save_hyperparameters(self.config)
        self.model = GiG(self.config)
        self.distribution_type = config["target_distribution"]
        self.initial_loss = losses[config["loss"]]
        self.saving_path = config["saving_path"]+'plots/'
        if os.path.exists(self.saving_path):
            shutil.rmtree(self.saving_path)
        access = 0o777
        os.makedirs(self.saving_path, access)

        self.alpha = config["alpha"]

    def training_step(self, batch, batch_idx):
        optimizers = self.optimizers()
        for opt in optimizers:
            opt.zero_grad()
        metrics, _  = self._shared_step(batch, addition="train")
        self.manual_backward(metrics['loss'])
        for opt in optimizers:
            opt.step()
        self.log_dict(metrics, prog_bar=True, batch_size=self.config["bs_train"])
        return metrics

    def validation_step(self, batch, batch_idx):
        metrics, _ = self._shared_step(batch, addition="val")
        self.log_dict(metrics, prog_bar=True, batch_size=self.config["bs_val"])
        return metrics

    def test_step(self, batch, batch_idx):
        metrics, _ = self._shared_step(batch, addition="test")
        self.log_dict(metrics, batch_size=self.config["bs_test"])
        return metrics

    def _shared_step(self, data, addition):
        y_hat, feature_matrix, edge_index, edge_weight, adj_matrix = self.model(data)
        if self.config["loss"] == "CrossEntropyLoss":
            loss = self.initial_loss(y_hat, data.y.long())
            y_hat = torch.softmax(y_hat,dim=1)
            labels = data.y

        elif self.config["dataset"] == 'Tox21':
            loss = self.initial_loss(y_hat, data.y)
            y_hat = torch.sigmoid(y_hat)


        else:
            loss = self.initial_loss(y_hat, data.y.float())
            y_hat = torch.sigmoid(y_hat)
            labels = data.y


        if self.config["dataset"]=='Tox21':
            y_hat, labels = mask_nans(y_hat,data.y)
        labels = labels.int()

        n_nodes = adj_matrix.shape[0]

        softhist = SoftHistogram(bins=n_nodes, min=0.5, max=n_nodes + 0.5, sigma=0.6)
        kl_loss = self.alpha * self._compute_kl_loss(adj_matrix, n_nodes, softhist)
        loss += kl_loss
        avg_coeff, eigenDecomposition_coeff = compute_clustering_coeff(np.array(adj_matrix.clone().detach().cpu()))


        acc = torchmetrics.functional.accuracy(y_hat, labels, task='multiclass' if self.config["dataset"]=='ENZYMES' else 'binary')
        f1 = torchmetrics.functional.f1_score(y_hat, labels, task='multiclass' if self.config["dataset"]=='ENZYMES' else 'binary')
        precision = torchmetrics.functional.precision(y_hat, labels, task='multiclass' if self.config["dataset"]=='ENZYMES' else 'binary')
        recall = torchmetrics.functional.recall(y_hat, labels, task='multiclass' if self.config["dataset"]=='ENZYMES' else 'binary')
        metrics = {addition + "_acc": acc, addition + "_f1": f1,
                   addition + "_precision": precision,
                   addition + "_recall": recall, addition + "_avg_coeff": avg_coeff,
                   addition + "_eigenDecomposition_coeff": eigenDecomposition_coeff,
                   "loss": loss,
                   addition + "_loss": loss
                   }

        return metrics, adj_matrix

    def _kl_div(self, p, q):
        return torch.sum(p * torch.log(p / (q + 1e-8) + 1e-8))

    def _compute_distr(self, adj, softhist):
        deg = adj.sum(-1)
        distr = softhist(deg)
        return distr / torch.sum(distr), deg

    def _compute_kl_loss(self, adj, batch_size, softhist):
        binarized_adj = torch.zeros(adj.shape).to(adj.device)
        binarized_adj[adj > 0.5] = 1
        dist, deg = self._compute_distr(adj * binarized_adj, softhist)
        target_dist = self._compute_target_distribution(batch_size)
        kl_loss = self._kl_div(dist, target_dist)
        return kl_loss

    def _compute_target_distribution(self, batch_size):
        # define target distribution
        target_distribution = torch.zeros(batch_size).to(self.model.population_level_module.sigma.device)
        if batch_size > 4:
            tab = 4
        else:
            tab = 0
        if self.distribution_type == 'power_law':

            target_distribution[tab:] = self.model.population_level_module.sigma * (
                    1.0 + torch.arange(batch_size - tab).to(self.model.population_level_module.sigma.device)).pow(self.model.population_level_module.mu)
        else:
            target_distribution[tab:] = torch.exp(
                -(self.model.population_level_module.mu - torch.arange(batch_size - tab).to(self.model.population_level_module.sigma.device) + 1.0) ** 2 / (
                        self.model.population_level_module.sigma ** 2))

        return target_distribution / target_distribution.sum()


    def configure_optimizers(self):
        if self.config["optimizer_type"] == 'adam':
            # optimizer
            population_level_module_par = [param for name_, param in
                                           self.model.population_level_module.named_parameters()
                                           if name_ not in ['temp', 'theta', 'mu', 'sigma']]
            population_level_module_par.extend(self.model.node_level_module.parameters())
            population_level_module_par.extend(self.model.gnn.parameters())
            population_level_module_par.extend(self.model.classifier.parameters())

            optimizer = torch.optim.Adam(population_level_module_par, lr=self.config["optimizer_lr"])
            kl_loss_optimizer = torch.optim.Adam([self.model.population_level_module.mu,
                                                  self.model.population_level_module.sigma,
                                                  ], lr=self.config["lr_mu_sigma"]
                                                 )
            lgl_optimizer = torch.optim.Adam([self.model.population_level_module.theta,
                                              self.model.population_level_module.temp,
                                              ], lr=self.config["lr_theta_temp"]
                                             )
        else:
            print("Not implemented")

        if self.config["scheduler"] == 'ReduceLROnPlateau':

            scheduler = {"scheduler": ReduceLROnPlateau(
                    optimizer, patience=10,
                    threshold=0.0001,
                    mode='min', verbose=True, threshold_mode='abs'),
                "interval": "epoch",
                "monitor": "loss"}

        elif self.config["scheduler"] == 'CosineAnnealingLR':
            scheduler = {"scheduler": CosineAnnealingLR(optimizer, T_max=10),
                "interval": "epoch",
                "monitor": "loss"}

        else:
            print("This scheduler is not implemented.")

        return [optimizer, kl_loss_optimizer, lgl_optimizer], scheduler

class LglTask(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.automatic_optimization = False
        self.config = config
        self.save_hyperparameters(self.config)
        self.model = GiG(self.config)
        self.initial_loss = losses[config["loss"]]
        self.saving_path = config["saving_path"]

    def training_step(self, batch, batch_idx):
        optimizers = self.optimizers()
        for opt in optimizers:
            opt.zero_grad()
        metrics, _  = self._shared_step(batch, addition="train")
        self.manual_backward(metrics['loss'])
        for opt in optimizers: 
            opt.step()
        self.log_dict(metrics, prog_bar=True, batch_size=self.config["bs_train"])
        return metrics

    def validation_step(self, batch, batch_idx):
        metrics, _ = self._shared_step(batch, addition="val")
        self.log_dict(metrics, batch_size=self.config["bs_val"])
        return metrics

    def test_step(self, batch, batch_idx):
        metrics, _ = self._shared_step(batch, addition="test")
        self.log_dict(metrics, batch_size=self.config["bs_test"])
        return metrics

    def _shared_step(self, data, addition):
        y_hat, feature_matrix, edge_index, edge_weight, adj_matrix = self.model(data)
        try:
            adj_matrix = np.array(adj_matrix.detach().cpu())
            avg_coeff, eigenDecomposition_coeff = compute_clustering_coeff(adj_matrix)
        except:
            avg_coeff = 0.0
            eigenDecomposition_coeff = 0.0
        if self.config["loss"] == "CrossEntropyLoss":
            loss = self.initial_loss(y_hat, data.y.long())
            y_hat = torch.softmax(y_hat,dim=1)
            labels = data.y


        elif self.config["dataset"] == 'Tox21':
            loss = self.initial_loss(y_hat, data.y)
            y_hat = torch.sigmoid(y_hat)


        else:
            loss = self.initial_loss(y_hat, data.y.float())
            y_hat = torch.sigmoid(y_hat)
            labels = data.y


        if self.config["dataset"]=='Tox21':
            y_hat, labels = mask_nans(y_hat,data.y)
        labels = labels.int()

        acc = torchmetrics.functional.accuracy(y_hat, labels, task='multiclass' if self.config["dataset"]=='ENZYMES' else 'binary')
        f1 = torchmetrics.functional.f1_score(y_hat, labels, task='multiclass' if self.config["dataset"]=='ENZYMES' else 'binary')
        precision = torchmetrics.functional.precision(y_hat, labels, task='multiclass' if self.config["dataset"]=='ENZYMES' else 'binary')
        recall = torchmetrics.functional.recall(y_hat, labels, task='multiclass' if self.config["dataset"]=='ENZYMES' else 'binary')
        metrics = {addition+"_acc": acc, addition+"_f1": f1,
                   addition+"_precision": precision,
                   addition+"_recall": recall, addition+"_avg_coeff": avg_coeff,
                   addition+"_eigenDecomposition_coeff": eigenDecomposition_coeff,
                   addition+"_loss": loss.clone().detach(), "loss": loss}
        return metrics, adj_matrix


    def configure_optimizers(self):
        optimizers =[]
        if self.config["optimizer_type"] == 'adam':
            # optimizer
            population_level_module_par = [param for name_, param in
                                           self.model.population_level_module.named_parameters()
                                           if name_ not in ['temp', 'theta', 'mu', 'sigma']]
            population_level_module_par.extend(self.model.node_level_module.parameters())
            population_level_module_par.extend(self.model.gnn.parameters())
            population_level_module_par.extend(self.model.classifier.parameters())

            optimizer = torch.optim.Adam(population_level_module_par, lr=self.config["optimizer_lr"])
            optimizers.append(optimizer)
            if self.config["population_level_module"] != "DGCNN":
                lgl_optimizer = torch.optim.Adam([self.model.population_level_module.theta,
                                                  self.model.population_level_module.temp,
                                                  ], lr=self.config["lr_theta_temp"]
                                                 )
                optimizers.append(lgl_optimizer)

        else:
            print("Not implemented")

        if self.config["dataset"] in ['Tox21', 'HCP']:
            scheduler = {"scheduler": ReduceLROnPlateau(
                optimizer, patience=10,
                threshold=0.0001,
                mode='min', verbose=True, threshold_mode='abs'),
                "interval": "epoch",
                "monitor": "val_loss"}

        else:
            scheduler = {"scheduler": CosineAnnealingLR(optimizer, T_max=10),
                         "interval": "epoch",
                         "monitor": "val_loss"}

        return optimizers, [scheduler]


