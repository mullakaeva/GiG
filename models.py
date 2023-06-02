import torch
import numpy as np
from torch.nn import Module, Linear, ModuleList, Sequential, LeakyReLU, ReLU
from pg_models.DynamicEdgeConv import DynamicEdgeConv
from torch_geometric.nn.pool.glob import global_add_pool, global_mean_pool
from torch_geometric.nn import GraphConv, EdgeConv, GCNConv, GATConv, SAGEConv
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn.models.basic_gnn import GIN

try:
    from torch_cluster import knn
except ImportError:
    knn = None

def DGCNN_layer(in_size, out_size, k=10):
    DGCNN_conv = Sequential(Linear(2 * in_size, out_size), LeakyReLU())

    return DynamicEdgeConv(DGCNN_conv, k=k)  # 10 #change to fix graph!!!!
#NODE-LEVEL MODULES
class NodeConvolution(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.conv_list = ModuleList()
        if self.config["node_level_module"] == "GIN":
            self.conv_list.append(GIN(in_channels=config["num_node_features"],
                                      hidden_channels=config["node_layers"][0],
                                      num_layers=self.config["node_level_hidden_layers_number"],
                                      out_channels=config["node_layers"][-1]))
        elif self.config["node_level_module"] == "GraphConv":
            self.conv_list.append(GraphConv(config["num_node_features"], config["node_layers"][0]))
            for i in np.arange(1, len(config["node_layers"])):
                self.conv_list.append(GraphConv(config["node_layers"][i - 1], config["node_layers"][i]))
            self.activation = ReLU()

        else:
            print("Not implemented node level layer")

        if config["pooling"] == 'add':
            self.pooling = global_add_pool
        elif config["pooling"] == 'mean':
            self.pooling = global_mean_pool
        else:
            print("This version is not implemented.")

    def forward(self, data):
        edge_index = data.edge_index
        x = data.x.float()
        if 'edge_weights' in data and self.config["node_level_module"]!="GIN":
            x = self.conv_list[0](x, edge_index, data.edge_weights)
        elif 'edge_attr' in data and self.config["node_level_module"]!="GIN":
            x = self.conv_list[0](x, edge_index, torch.argmax(data.edge_attr, dim=1))
        else:
            x = self.conv_list[0](x, edge_index)
        if self.config["node_level_module"]!="GIN":
            x = self.activation(x)

        for conv in self.conv_list[1:]:
            x = conv(x, edge_index)
            if self.config["node_level_module"] != "GIN":
                x = self.activation(x)
        x = self.pooling(x, data.batch)
        return x
#POPULATION-LEVEL MODULES
#cdgm
class LGL(Module):
    def __init__(self, config, input_size):
        super().__init__()
        self.conv_list = ModuleList()
        self.conv_list.append(torch.nn.Linear(input_size, config["population_layers"][0]))

        for i in np.arange(1, len(config["population_layers"])):
            layer = torch.nn.Linear(config["population_layers"][i - 1], config["population_layers"][i])
            self.conv_list.append(layer)

        self.weight_layer = torch.nn.Linear(input_size, config["population_layers"][-1])

        self.activation = ReLU()

        self.temp = torch.nn.Parameter(torch.tensor(config["temp"], requires_grad=True))
        self.theta = torch.nn.Parameter(torch.tensor(config["theta"], requires_grad=True))
        if config["population_level_module"] == "LGLKL":
            self.mu = torch.nn.Parameter(torch.tensor(config["mu"], requires_grad=True))
            self.sigma = torch.nn.Parameter(torch.tensor(config["sigma"], requires_grad=True))


    def forward(self, x):

        for k, layer in enumerate(self.conv_list):
            if k == 0:
                out_x = layer(x)
                out_x = self.activation(out_x)
            else:
                out_x = layer(out_x)
                out_x = self.activation(out_x)

        # compute pairwise distance
        diff = out_x.unsqueeze(1) - out_x.unsqueeze(0)
        # compute the norm
        diff = torch.pow(diff, 2).sum(2)
        mask_diff = diff != 0.0
        dist = - torch.sqrt(diff + torch.finfo(torch.float32).eps)
        dist = dist * mask_diff
        prob_matrix = self.temp * dist + self.theta

        adj = prob_matrix + torch.eye(prob_matrix.shape[0]).to(prob_matrix.device)
        adj = torch.sigmoid(adj)
        edge_index, edge_weight = dense_to_sparse(adj)
        return x, edge_index, edge_weight, adj
#GNN
class GNN(Module):
    def __init__(self, config, input_size):
        super().__init__()
        self.config = config
        self.graph_conv = ModuleList()
        self.activation = ReLU()

        if config["gnn_type"] == "GraphConv":
            self.graph_conv.append(GraphConv(input_size, config["gnn_layers"][0], aggr=config["gnn_aggr"]))
            for i in np.arange(1, len(config["gnn_layers"])):
                layer = GraphConv(config["gnn_layers"][i - 1], config["gnn_layers"][i], aggr=config["gnn_aggr"])
                self.graph_conv.append(layer)
        elif config["gnn_type"] == "GAT":
            self.graph_conv.append(GATConv(input_size, config["gnn_layers"][0], aggr=config["gnn_aggr"]))
            for i in np.arange(1, len(config["gnn_layers"])):
                layer = GATConv(config["gnn_layers"][i - 1], config["gnn_layers"][i], aggr=config["gnn_aggr"])
                self.graph_conv.append(layer)
        elif config["gnn_type"] == "SAGEConv":
            self.graph_conv.append(SAGEConv(input_size, config["gnn_layers"][0], aggr=config["gnn_aggr"]))
            for i in np.arange(1, len(config["gnn_layers"])):
                layer = SAGEConv(config["gnn_layers"][i - 1], config["gnn_layers"][i], aggr=config["gnn_aggr"])
                self.graph_conv.append(layer)
        elif config["gnn_type"] == "EdgeConv":
            self.graph_conv.append(EdgeConv(Sequential(Linear(2 *input_size, config["gnn_layers"][0])),
                                            aggr=config["gnn_aggr"]))
            for i in np.arange(1, len(config["gnn_layers"])):
                layer = EdgeConv(Sequential(Linear(2 *config["gnn_layers"][i - 1], config["gnn_layers"][i])), aggr=config["gnn_aggr"])
                self.graph_conv.append(layer)
        elif config["gnn_type"] == "GCN_kipf":
            self.graph_conv.append(GCNConv(input_size, config["gnn_layers"][0]))
            for i in np.arange(1, len(config["gnn_layers"])):
                layer = GCNConv(config["gnn_layers"][i - 1], config["gnn_layers"][i])
                self.graph_conv.append(layer)
        else:
            print("Not implemented!")

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is not None and self.config["gnn_type"] not in ["EdgeConv", "SAGEConv", "GAT"]:
            x = self.graph_conv[0](x, edge_index, edge_weight)
        else:
            x = self.graph_conv[0](x, edge_index)
        x = self.activation(x)
        for conv in self.graph_conv[1:]:
            x = conv(x, edge_index)
            x = self.activation(x)
        return x
#CLASSIFIER
class Classifier(Module):
    def __init__(self, config, input_size, output_dim):
        super().__init__()
        self.config = config
        if len(config["classifier_layers"]) > 0:
            fc_list = [Linear(input_size, config["classifier_layers"][0])]
            for i in np.arange(1, len(config["classifier_layers"])):
                fc_list.append(ReLU())
                fc_list.append(Linear(config["classifier_layers"][i - 1], config["classifier_layers"][i]))
            fc_list.append(ReLU())
            fc_list.append(Linear(config["classifier_layers"][- 1], output_dim))
        else:
            fc_list = [Linear(input_size, output_dim)]
        self.fc = Sequential(*fc_list)
    def forward(self, x):
        x = self.fc(x)
        return x

class GiG(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        #node-level module
        self.node_level_module = NodeConvolution(config)
        input_size = config["node_layers"][-1]
        #population-level module
        if config["population_level_module"] in ['LGL', 'LGLKL']:
            self.population_level_module = LGL(config, input_size)
        else:
            print("Please define the population-level module type: DGCNN, LGL, LGLKL")
        #GNN
        if len(config["gnn_layers"]) > 0:
            self.gnn = GNN(config, input_size)
            input_size = config["gnn_layers"][-1]
        #classifier
        if config["output_dim"] == 2:
            output_dim = config["output_dim"] - 1
        else:
            output_dim = config["output_dim"]
        self.classifier = Classifier(config, input_size, output_dim)



    def forward(self, data):
        feature_matrix = self.node_level_module(data)
        x, edge_index, edge_weight, adj_matrix = self.population_level_module(feature_matrix)
        x = self.gnn(x, edge_index, edge_weight)
        x = self.classifier(x)
        if self.config["output_dim"] == 2:
            x = x.view(-1)
        return x, feature_matrix, edge_index, edge_weight, adj_matrix


