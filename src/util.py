import torch
import math
import torch.nn as nn
import os
from  torch.utils.data import TensorDataset 
import torch.nn.functional as F
torch.manual_seed(42)

class MLP(nn.Module):
    # layer_sizes[0] is the dimension of the input
    # layer_sizes[-1] is the dimension of the output
    def __init__(self, layer_sizes, final_relu=False):
        super().__init__()
        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=False))
            layer_list.append(nn.Linear(input_size, curr_size))
        self.net = nn.Sequential(*layer_list)
        self.last_linear = self.net[-1]

    def forward(self, x):
        return self.net(x)
    
    
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
        
class DisMaxLossFirstPart(nn.Module):
    """This part replaces the model classifier output layer nn.Linear()."""
    def __init__(self, num_features, num_classes, temperature=1.0, flag = 1, alpha = 0):
        super(DisMaxLossFirstPart, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.distance_scale = nn.Parameter(torch.Tensor(1)) 
        nn.init.constant_(self.distance_scale, 1.0)
        self.prototypes = nn.Parameter(torch.Tensor(num_classes, num_features), requires_grad= True)
        nn.init.normal_(self.prototypes, mean=0.0, std=1.0)
        self.temperature = nn.Parameter(torch.tensor([temperature]), requires_grad=False)        
        self.validationset_available = nn.Parameter(torch.tensor([False]), requires_grad=False)
        self.precomputed_thresholds = nn.Parameter(torch.Tensor(2, 25), requires_grad=False)

    def forward(self, features):
        distances_from_normalized_vectors = torch.cdist(
            F.normalize(features), F.normalize(self.prototypes), p=2.0, compute_mode="donot_use_mm_for_euclid_dist") / math.sqrt(2.0)    
        
        isometric_distances = torch.abs(self.distance_scale) * distances_from_normalized_vectors
        logits = -(isometric_distances + isometric_distances.mean(dim=1, keepdim=True))
        # logits = - isometric_distances
        # The temperature may be calibrated after training to improve uncertainty estimation.
        return logits / self.temperature

    def extra_repr(self):
        return 'num_features={}, num_classes={}'.format(self.num_features, self.num_classes)
