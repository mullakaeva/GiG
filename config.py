import yaml
import os
from dataclasses import dataclass
import torch

def create_config(config_name, model_type=None, population_level_module_type=None,
                  alpha=None, pooling=None, gnn_type=None):
    # Function to load yaml configuration file
    def load_config(config_name):
        with open(os.path.join("configs/", config_name)) as file:
            config = yaml.safe_load(file)

        return config

    yaml_config = load_config(config_name)
    model =yaml_config["model"]
    node_level_module = model["node_level_module"]["type"]
    if not model_type:
        model_type = yaml_config["model"]["type"]
    if not population_level_module_type:
        population_level_module_type = model["population_level_module"]["type"]
    if model_type == 'GCN':
        population_level_module_type = 'none'
    if not pooling:
        pooling = yaml_config["model_parameters"]["pooling"]
    if not gnn_type:
        gnn_type = model["GNN"]["type"]

    d = {"model_type": model_type,
        "dataset": yaml_config["dataset_name"],
        "population_level_module": population_level_module_type,
        "node_level_module" : node_level_module,
        "gnn_type": gnn_type,
        "node_layers" : model["node_level_module"][node_level_module]["layers_sizes"],
        "gnn_layers" : model["GNN"][gnn_type]["layers_sizes"],
        "classifier_layers" : model["classifier"]["layers_sizes"],
        "pooling": pooling,
        "epochs": yaml_config["training_parameters"]["epochs"],
        "patience": yaml_config["training_parameters"]["patience"],
        "output_dim": yaml_config["data_parameters"]["output_dim"],
        "input_dim": yaml_config["data_parameters"]["input_dim"],
        "num_node_features": yaml_config["data_parameters"]["num_node_features"],
        "bs_train": yaml_config["data_parameters"]["batch_size"]["train"],
        "bs_val": yaml_config["data_parameters"]["batch_size"]["val"],
        "bs_test": yaml_config["data_parameters"]["batch_size"]["test"],
        "split_type": yaml_config["data_parameters"]["splits"]["type"],
        "train_split": yaml_config["data_parameters"]["splits"]["train"],
        "val_split": yaml_config["data_parameters"]["splits"]["val"],
        "test_split": yaml_config["data_parameters"]["splits"]["test"],
        "optimizer_type": yaml_config["training_parameters"]["optimizer"]["type"],
        "optimizer_lr": float(yaml_config["training_parameters"]["optimizer"]["lr"]),
        "loss": yaml_config["training_parameters"]["loss"]["type"],
        "loss_weights": yaml_config["training_parameters"]["loss"]["weights"],
        "scheduler": yaml_config["training_parameters"]["scheduler"]
    }

    if model_type != 'GCN':
        if population_level_module_type not in ["random", "knn"]:
             d = {**d, **{"population_layers": model["population_level_module"][population_level_module_type][
            "layers_sizes"]}}
        else:
            d = {**d, **{"k": model["population_level_module"][population_level_module_type]["k"]}}

    if gnn_type != 'GCN_kipf':
        d = {**d, **{"gnn_aggr": model["GNN"][gnn_type]["agg"]}}

    if population_level_module_type == "LGL":
        d = {**d, **{"lr_theta_temp": float(yaml_config["LGL"]["lr_theta_temp"])},
             **{"temp": float(yaml_config["LGL"]["temp"]),
                                                 "theta": float(yaml_config["LGL"]["theta"])}}
    if population_level_module_type == "LGLKL":
        d = {**d, **{"lr_theta_temp": float(yaml_config["LGLKL"]["lr_theta_temp"]),
                                               "lr_mu_sigma": float(yaml_config["LGLKL"]["lr_mu_sigma"])
                                               }}

        target_distribution = yaml_config["LGLKL"]["target_distribution"]
        if not alpha:
            alpha = float(yaml_config["LGLKL"][target_distribution]["alpha"])
        d = {**d, **{"target_distribution": target_distribution,
                            "mu": float(yaml_config["LGLKL"][target_distribution]["mu"]),
                            "sigma": float(yaml_config["LGLKL"][target_distribution]["sigma"]),
                            "alpha": alpha,
                             "temp": float(yaml_config["LGLKL"]["temp"]),
                             "theta": float(yaml_config["LGLKL"]["theta"])
                            }}
    if node_level_module == 'GIN':
        d["node_level_hidden_layers_number"] = model["node_level_module"][node_level_module]["hidden_layers_num"]


    if population_level_module_type == "DGCNN":
        d = {**d, **{"k": model["population_level_module"]["DGCNN"]["k"]}}

    # Saving paths
    d["dataset_path"] = yaml_config["paths"]["dataset_path"]
    d["saving_path"] = yaml_config["paths"]["model_path"] + '/'+ d["dataset"] + \
                       "/" + model_type +'_'+ population_level_module_type + "/"
    return d



