import torch
import torch.nn as nn
import torch.nn.functional as F
from util import device
import math

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
'''
Class Dismax is adpated from the class DisMaxLossFirstPart from the following link:
https://github.com/dlmacedo/robust-deep-learning/

'''   
class DisMax(nn.Module):
    def __init__(self, num_features, num_classes, temperature=1.0):
        super(DisMax, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.distance_scale = nn.Parameter(torch.Tensor(1)) 
        nn.init.constant_(self.distance_scale, 1.0)
        self.prototypes = nn.Parameter(torch.Tensor(num_classes, num_features), requires_grad= True)
        nn.init.normal_(self.prototypes, mean=0.0, std=1.0)
        self.temperature = nn.Parameter(torch.tensor([temperature]), requires_grad=False)        

    def forward(self, features):
        distances_from_normalized_vectors = torch.cdist(
            F.normalize(features), F.normalize(self.prototypes), p=2.0, compute_mode="donot_use_mm_for_euclid_dist") / math.sqrt(2.0)    
        isometric_distances = torch.abs(self.distance_scale) * distances_from_normalized_vectors
        logits = -(isometric_distances + isometric_distances.mean(dim=1, keepdim=True))
        return logits / self.temperature
    
class SA_classifier(nn.Module):
    def __init__(self, extractor, layer_sizes, classifier):
        super(SA_classifier, self).__init__()
        self.extractor = extractor
        self.dropout = nn.Dropout()
        assert classifier in ['li', 'dml']
        if classifier == 'li':
            self.classifier = MLP(layer_sizes)
        else:
            self.classifier = DisMax(layer_sizes[0], layer_sizes[1])
    
    def forward(self, x, Feature_return = False):
        input_ids = torch.tensor(x['input_ids']).to(device)
        attention_mask = torch.tensor(x['attention_mask']).to(device)
        hidden_states = self.extractor(input_ids, attention_mask = attention_mask)
        x_feat = hidden_states[0][:,0,:]
        if Feature_return:
            return x_feat     
        x_feat = self.dropout(x_feat)
        output = self.classifier(x_feat)
        return output
