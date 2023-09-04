# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn import random_projection
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import cross_val_score
# import torch
# import torch.nn as nn
# import transformers as ppb
from transformers import BertModel, AutoModel, BertForSequenceClassification, BertConfig, AutoModelForSequenceClassification, AutoTokenizer
# import os
# import warnings
# import torch.nn.functional as F
# from util import MLP, create_feature_loader, transform_dataset, DisMaxLossFirstPart
from run_task import run_task, set_model, test
from data_prepare import SA_processing
from util import *

# from pathlib import Path
# run_config_file = Path('/root/configs/default.yaml')
run_config = load_config()
print("=======")
print(run_config)
print("==========")
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    LoraConfig,
    TaskType,
    IA3Config
)

warnings.filterwarnings('ignore')


seed = run_config["seed"]
torch.manual_seed(seed)
BATCH_SIZE = run_config["batch_size"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_attention_mask(padded):
    """
    If we directly send padded to BERT, that would slightly confuse it. We need to create another variable to tell 
    it to ignore (mask)the padding we've added when it's processing its input.
    That's what attention_mask is.
    """
    attention_mask = np.where(padded != 0, 1, 0)
    return attention_mask
    

    


class SA_exactor(nn.Module):
    def __init__(self, exactor):
        super(SA_exactor, self).__init__()
        self.extractor = exactor
        self.extractor = torch.nn.DataParallel(self.extractor).to(device)
 
    def get_all_embeddings(self, dataloader):
        embeddings = []
        targets = []
        self.eval()
        with torch.no_grad():
            for data, labels in dataloader:
                input_ids = torch.tensor(data['input_ids']).to(device)
                # labels = torch.tensor(data['labels']).to(device)
                labels = torch.tensor(labels).to(device)
                attention_mask = torch.tensor(data['attention_mask']).to(device)
                last_hidden_states = self.extractor(input_ids, attention_mask = attention_mask)
                embs = torch.tensor(last_hidden_states[0][:,0,:])
                embeddings.append(embs)
                targets.append(labels)
        return  torch.cat(embeddings), torch.cat(targets) 
         
           
    def get_feature_dataset(self, dataloader, save_path = None):
        features, targets = self.get_all_embeddings(dataloader)
        feature_exactor_dataset = torch.utils.data.TensorDataset(features,targets)
        feature_exactor_dataset.features = features
        feature_exactor_dataset.targets = targets
        if save_path:
            torch.save(feature_exactor_dataset, save_path)
        return feature_exactor_dataset
    
    def forward(self, x):
        input_ids = torch.tensor(x['input_ids']).to(device)
        attention_mask = torch.tensor(x['attention_mask']).to(device)
        
        hidden_states = self.extractor(input_ids, attention_mask = attention_mask)
        # last_hidden_states = hidden_states[1][0][:,0,:] 
        # x_feat = hidden_states[1][0][:,0,:]  
        x_feat = hidden_states[0][:,0,:]  
        # output = self.classifier(x_feat)
        return x_feat
   
  

def custom_classifier(model,layer_sizes):
       model.classifier = DisMaxLossFirstPart(layer_sizes[0], layer_sizes[1])
       return model
       
class SA_classifier(nn.Module):
    def __init__(self, extractor, layer_sizes, eta = [1,1]):
        super(SA_classifier, self).__init__()
        self.extractor = extractor
        self.dropout = nn.Dropout()
        # self.classifier = MLP(layer_sizes)
        self.classifier = DisMaxLossFirstPart(layer_sizes[0], layer_sizes[1])
        self.eta = nn.Parameter(torch.Tensor(eta)) 
        # self.adapter = MLP([layer_sizes[0], layer_sizes[0]], final_relu= True)
        # self.freeze_bert()
    
    def freeze_bert(self):
        """Freezes the parameters of BERT so when BertWithCustomNNClassifier is trained
        only the wieghts of the custom classifier are modified.
        """
        for param in self.extractor.named_parameters():
            param[1].requires_grad=False 
   
    
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
  
  
def prompt(model):
    
    # peft_type = PeftType.PROMPT_TUNING
    peft_config = PromptTuningConfig(task_type="FEATURE_EXTRACTION", 
                                    #  token_dim=768, 
                                    #  num_attention_heads = 2, num_layers=2, 
                                     num_virtual_tokens=7,
                                    #  prompt_tuning_init="TEXT",
                                    #  prompt_tuning_init_text="Predict if sentiment of this review is positive, negative or neutral"
    )
#     peft_config = LoraConfig(
#     task_type= "FEATURE_EXTRACTION", r=8, lora_alpha=32, lora_dropout=0.1,
#     target_modules=['query', 'value'],
#     modules_to_save=["classifier"],
    
# )
    # prefix tuning
    # peft_type = PeftType.PREFIX_TUNING
    # peft_config = PrefixTuningConfig(task_type="FEATURE_EXTRACTION", 
    #                                 #  num_layers = 12,
    #                                 #  token_dim = 768,
    #                                 #  num_attention_heads = 12,
    #                                  num_virtual_tokens=10)
   
   
    # peft_type = PeftType.IA3
    # peft_config = IA3Config(task_type="FEATURE_EXTRACTION", inference_mode=False)
    
    model_peft = get_peft_model(model, peft_config)
    model_peft.print_trainable_parameters()
    return model_peft

if __name__ == "__main__":
    task_path = './tasks/'
    emb_path = './emb/'
    train_path = './SST2_train.tsv'
    val_path = './SST2_dev.tsv'
    test_path = './SST2_test.tsv'
    os.makedirs(task_path, exist_ok=True)
    os.makedirs(emb_path, exist_ok=True)
    task_path = os.path.join(task_path, 'promp_dml_t5.pth')
    emb_train_path = os.path.join(emb_path, 'emb_train')
    emb_test_path = os.path.join(emb_path, 'emb_test')
    train_dataset, val_dataset = SA_processing(train_path, val_path)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size= BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # full bert
    # model_name = 'bert-base-uncased'
    # model_name = "roberta-base"
    # model_name = 'microsoft/deberta-base'
    model_name = run_config['model_name']
    extractor = AutoModel.from_pretrained(model_name)
    
    # distill bert 
    # model_name = 'distilbert-base-uncased'
    # extractor = ppb.DistilBertModel.from_pretrained(model_name)
    # sac = prompt(extractor)
    # hhh
    # model = SA_classifier(extractor, [768,2])
    # sac = prompt(model)
    sac = prompt(extractor)
    # sac = extractor 
   
    # bert params trainable
    n_components = 768
    layer_sizes = [n_components, 2]
    
    task = SA_classifier(sac, layer_sizes).to(device)
    lr = run_config["learning_rate"]
    epoch = 100
    step_size = 10
    run_task(task_path, task, train_loader, val_loader, lr, epoch)
    set_model(task, task_path) 
    test(task, test_path, BATCH_SIZE)
    
    hhh
    
    features = get_embedding(padded, model)
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels)
    
    lr_clf = LogisticRegression()
    lr_clf.fit(train_features, train_labels)
    print(lr_clf.score(test_features, test_labels))
    
    
