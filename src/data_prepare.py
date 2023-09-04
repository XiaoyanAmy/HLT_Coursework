from transformers import AutoTokenizer
from util import *

run_config = load_config()

def data_preprocessing(data_url = 'https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/train.tsv', n_samples = None):
    
    df = pd.read_csv(data_url, delimiter='\t', header=None)
    # is none, means return all the samples, no need sampling
    if n_samples is None:
        df_sample = df
    else:
        ## to ensure class balanced sampling
        p0, p1 = 0.5, 0.5
        df0 = df.loc[df[1] == 0]
        df1 = df.loc[df[1] == 1]
        random_state = run_config['sampling_random_state']
        df0_sample = df0.sample(n = int(n_samples*p0), random_state= random_state)
        df1_sample = df1.sample(n = int(n_samples*p1), random_state= random_state)
        df_sample = pd.concat([df0_sample, df1_sample], ignore_index=True, axis=0)    
    text = df_sample[0].values.tolist()
    labels = df_sample[1].values.tolist()
    

    
    return text, labels

def testlabel(labels):
    # count the number of different labels
    count = 0
    for i in range(len(labels)):
        if labels[i]:
            count += 1
    print(count, len(labels) - count)
  
    
def token_preprocessing(text):
    model_name = run_config['model_name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoded = tokenizer(text, truncation=True , padding=True)
    return encoded


def SA_processing(train_path, dev_path):
    n_samplings = run_config['num_samplings']
    train_text, train_label = data_preprocessing(train_path, n_samples = n_samplings)
    val_text, val_label = data_preprocessing(dev_path, n_samples= n_samplings)
    train_encod = token_preprocessing(train_text)
    val_encod = token_preprocessing(val_text)
    train_dataset = CustomData(train_encod, train_label)
    val_dataset = CustomData(val_encod, val_label)
    return train_dataset, val_dataset
    
class CustomData(torch.utils.data.Dataset):
    '''Class to store the output of the Tokenizer as pytorch dataset'''
    
    def __init__(self , encodings , labels):
        self.encodings = encodings
        self.targets = labels
        
    def __getitem__(self, idx):
        item = {key : torch.tensor(val[idx]) for key , val in self.encodings.items() }
        target = self.targets[idx]
        return (item, target)
    
    def __len__(self):
        return len(self.targets)
    