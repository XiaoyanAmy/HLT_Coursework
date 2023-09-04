import os
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


import torch
from  torch.utils.data import TensorDataset 
import torch.nn as nn
import torch.nn.functional as F
import os
import warnings
import yaml
from pathlib import Path
task_path = './tasks/'
train_path = './SST2_train.tsv'
val_path = './SST2_dev.tsv'
test_path = './SST2_test.tsv'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config():
    with open(Path('/root/configs/config.yaml'), "r") as config_file:
        try:
            run_config = yaml.safe_load(config_file)
        except yaml.YAMLError as exc:
            print(exc)
            return {}

    return run_config
    
    
def transform_dataset(transformer, dataset, dataloader_flag = False):
    if transformer is None:
        dataset_new = dataset
        if dataloader_flag:
            feature_loader =torch.utils.data.DataLoader(dataset_new,
                                    batch_size = 128, shuffle = True)
            return feature_loader
        return dataset_new
    
    X = dataset.features
    X = X.reshape(X.shape[0], -1)
    feature = torch.tensor(transformer.fit_transform(X)).float()
    targets = dataset.targets.clone().detach()
    dataset_new = TensorDataset(feature, targets)
    dataset_new.features = feature
    dataset_new.targets = targets
    
    if dataloader_flag:
        feature_loader =torch.utils.data.DataLoader(dataset_new,
                                    batch_size = 512, shuffle = False)
        return feature_loader
    return dataset_new


def create_feature_loader(model, data, file_path, batch_size = 512, shuffle= False):
    if os.path.isfile(file_path):
        print("loading from dataset")
        feature_dataset = torch.load(file_path)
    else:
        print("reconstruct")
        feature_dataset = model.get_feature_dataset(data, file_path)
        
    feature_loader =torch.utils.data.DataLoader(feature_dataset,
                                    batch_size=batch_size, shuffle = shuffle)
    return feature_loader
        
