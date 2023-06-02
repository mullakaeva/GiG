# Graph-in-Graph (GiG)
Learning interpretable latent graphs in the non-Euclidean domain for molecule prediction and healthcare applications. The original code can be found in https://github.com/mullakaeva/GiG_original. Current code is upgraded and maintained for the latest package versions. 

## Installation

Requirements file contains necessary packages. Please first install torch and connected packages following the instructions from [pytorch](https://pytorch.org/get-started/locally/) and [pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html). 

This code is adapted for torch==1.13.1 and cuda 11.6
Installation might be done the following way:
```
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install torch_geometric
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
```


Then the rest packages can be installed by 
```bash
pip install -r requirements.txt
```
Make sure to install the latest or development version of Optuna (pip install git+https://github.com/optuna/optuna.git
).

## Instructions
### Datasets
For bench-marking datasets please follow instructions from [A Fair Comparison of Graph Neural Networks for Graph Classification](
https://github.com/diningphil/gnn-comparison#instructions) and locate it in **data/CHEMICAL**.
Code for preprocessing datasets was taken from paper [A Fair Comparison of Graph Neural Networks for Graph Classification](
https://github.com/diningphil/gnn-comparison#instructions) [1]
Do not forget to change the splits.

### Execution 
To run the experiments with LGL model
```bash
python main_grid_optuna.py --dataset <dataset_name> --population_level_module_type <model>
```
Where ```<dataset_name>``` is one of the datasets ```DD, ENZYMES,"NCI1", "PROTEINS_full" ```
and  ```<model>``` in ``` LGL, LGLKL```, where ``` LGL, LGLKL``` are corresponded for GiG LGL and GiG LGL+NDD

P.S. You can play with the parameters, make the hyperparameters search, or check them from the paper https://authors.elsevier.com/a/1hABr4rfPmE0QX

[1] Federico Errica, Marco Podda, Davide Bacciu, Alessio Micheli: A Fair Comparison of Graph Neural Networks for Graph Classification. Proceedings of the 8th International Conference on Learning Representations (ICLR 2020)
