from tasks import LglKlTask, LglTask
from datasets import *
def define_dataset(dataset_name, data_dir):

    def preprocess_dataset(name, args_dict):
        dataset_class = DATASETS[name]
        if name == 'ENZYMES':
            args_dict.update(use_node_attrs=True)
        return dataset_class(**args_dict)

    args_dict = {'DATA_DIR': data_dir,
                 'outer_k': 10,
                 'inner_k': None,
                 'use_one': False,
                 'use_node_degree': False,
                 'precompute_kron_indices': True}


    dataset = preprocess_dataset(dataset_name, args_dict)
    return dataset
DATASETS2YAML = {
                 "ENZYMES": "enzymes.yaml",
                 "DD":"dd.yaml",
                 "NCI1":"nci1.yaml",
                 "PROTEINS_full": "proteins.yaml",
                               }

DATASETS = {
    'REDDIT-BINARY': RedditBinary,
    'REDDIT-MULTI-5K': Reddit5K,
    'COLLAB': Collab,
    'IMDB-BINARY': IMDBBinary,
    'IMDB-MULTI': IMDBMulti,
    'NCI1': NCI1,
    'ENZYMES': Enzymes,
    'PROTEINS_full': Proteins,
    'DD': DD
}

def define_task(task_name, config):
    if task_name=='LGLKL':
        task = LglKlTask(config)
    elif task_name=='LGL':
        task = LglTask(config)
    else:
        print("Tasks is not implemented.")
    return task

