import numpy as np
import torch.nn as nn
import torch
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from grakel import Graph
from torch_geometric.utils import subgraph
from torch import Tensor
from torch_geometric.utils import erdos_renyi_graph
import seaborn as sns
from scipy.sparse import csgraph
from numpy import linalg as LA
import matplotlib.pyplot as plt
import copy
import  networkx as nx
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
from datasets.sampler import RandomSampler
from torch_geometric.datasets import TUDataset, MoleculeNet
import os

class SoftHistogram(nn.Module):
    def __init__(self, bins, min, max, sigma):
        super(SoftHistogram, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.centers = float(min) + self.delta * (torch.arange(bins).float() + 0.5)
        # self.centers = self.centers

    def forward(self, x):
        d = torch.cdist(self.centers[:, None].to(x.device), x[:, None])
        x = torch.softmax(-d ** 2 / self.sigma ** 2, dim=0)  # - torch.sigmoid(self.sigma * (x - self.delta/2))
        x = x.sum(dim=1)
        return x

#https://towardsdatascience.com/spectral-graph-clustering-and-optimal-number-of-clusters-estimation-32704189afbe
def eigenDecomposition(A, plot=False, topK=5):
    """
    :param A: Affinity matrix
    :param plot: plots the sorted eigen values for visual inspection
    :return A tuple containing:
    - the optimal number of clusters by eigengap heuristic
    - all eigen values
    - all eigen vectors

    This method performs the eigen decomposition on a given affinity matrix,
    following the steps recommended in the paper:
    1. Construct the normalized affinity matrix: L = D−1/2ADˆ −1/2.
    2. Find the eigenvalues and their associated eigen vectors
    3. Identify the maximum gap which corresponds to the number of clusters
    by eigengap heuristic

    References:
    https://papers.nips.cc/paper/2619-self-tuning-spectral-clustering.pdf
    http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/attachments/Luxburg07_tutorial_4488%5b0%5d.pdf
    """
    L = csgraph.laplacian(A, normed=True)
    n_components = A.shape[0]

    # LM parameter : Eigenvalues with largest magnitude (eigs, eigsh), that is, largest eigenvalues in
    # the euclidean norm of complex numbers.
    #     eigenvalues, eigenvectors = eigsh(L, k=n_components, which="LM", sigma=1.0, maxiter=5000)
    eigenvalues, eigenvectors = LA.eig(L)

    if plot:
        plt.title('Largest eigen values of input matrix')
        plt.scatter(np.arange(len(eigenvalues)), eigenvalues)
        plt.grid()

    # Identify the optimal number of clusters as the index corresponding
    # to the larger gap between eigen values
    index_largest_gap = np.argsort(np.diff(eigenvalues))[::-1][:topK]
    nb_clusters = index_largest_gap + 1

    return nb_clusters[0]

def compute_clustering_coeff(A):
    """
    A: adjacency matrix
    """
    try:
        G = nx.from_numpy_matrix(A)
        avg_coeff = nx.average_clustering(G)
        eigenDecomposition_coeff = eigenDecomposition(A)
        return avg_coeff, eigenDecomposition_coeff
    except:
        return 1.0, 1.0


class MultiTaskBCELoss(nn.Module):
    def __init__(self, ):
        super(MultiTaskBCELoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss(pos_weight =torch.Tensor([13]))
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        nan_mask = ~torch.isnan(target)
        labels_el = target[nan_mask]
        pred_el = input[nan_mask]
        ce_loss = self.loss(pred_el, labels_el)
        return ce_loss



def mask_nans(input: Tensor, target: Tensor):
    nan_mask = ~torch.isnan(target)
    labels_el = target[nan_mask]
    pred_el = input[nan_mask]
    return pred_el, labels_el

def compute_target_distribution(batch_size, model):
    # define target distribution
    target_distribution = torch.zeros(batch_size).to(model.population_level_module.sigma.device)
    if batch_size > 4:
        tab = 4
    else:
        tab = 0

    target_distribution[tab:] = torch.exp(
        -(model.population_level_module.mu - torch.arange(batch_size - tab).to(model.population_level_module.sigma.device) + 1.0) ** 2 / (
               model.population_level_module.sigma ** 2))

    return target_distribution / target_distribution.sum()


def create_random_split(dataset, config, seed=42):
    proteins_dataset = TUDataset(os.path.join('data', "PROTEINS_full"),
                         name='PROTEINS_full', use_node_attr=False)
    dataset_list = []
    for i, data in enumerate(proteins_dataset):
        data.x=dataset[i].x
        dataset_list.append(data)


    num_training = int(len(dataset_list) * config["train_split"])
    num_test = len(dataset_list) - (num_training)
    training_set, test_set = random_split(dataset_list, [num_training, num_test],
                                          generator=torch.Generator().manual_seed(seed))
    num_training = int(len(training_set) * config["train_split"])
    num_val = len(training_set) - (num_training)

    training_set, validation_set = random_split(training_set, [num_training, num_val],
                                                generator=torch.Generator().manual_seed(seed))

    loader_train = DataLoader(list(training_set), batch_size=config["bs_train"],shuffle=True)
    loader_val = DataLoader(list(validation_set), batch_size=config["bs_val"], shuffle=False)
    loader_test = DataLoader(list(test_set), batch_size=config["bs_test"], shuffle=False)
    return loader_train, loader_val, loader_test
