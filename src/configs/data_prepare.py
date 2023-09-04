from transformers import AutoTokenizer
from util import *
# from pathlib import Path
# run_config_file = Path('/root/configs/default.yaml')
run_config = load_config()

def data_preprocessing(data_url = 'https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/train.tsv', n_samples = None):
    
    df = pd.read_csv(data_url, delimiter='\t', header=None)
    if n_samples is None:
        df_sample = df
    else:
        ##this sample cause classed not balanced
        # p0, p1 = 0.2, 0.8
        # df_sample = df.sample(n_samples, random_state= 42)
        ## class balanced sampling
        p0, p1 = 0.5, 0.5
        df0 = df.loc[df[1] == 0]
        df1 = df.loc[df[1] == 1]
        random_state = run_config['sampling_random_state']
        df0_sample = df0.sample(n = int(n_samples*p0), random_state= random_state)
        df1_sample = df1.sample(n = int(n_samples*p1), random_state= random_state)
        df_sample = pd.concat([df0_sample, df1_sample], ignore_index=True, axis=0)    
    text = df_sample[0].values.tolist()
    labels = df_sample[1].values.tolist()
    
    count = 0
    for i in range(len(labels)):
        if labels[i]:
            count += 1
    print(count, len(labels) - count)
    
    return text, labels

def testlabel(labels):
    count = 0
    for i in range(len(labels)):
        if labels[i]:
            count += 1
    print(count, len(labels) - count)
    
def data_split(all_texts, all_labels):
    train_text , val_text , train_label , val_label = train_test_split(all_texts, all_labels, random_state = 42)
    testlabel(train_label, len(train_label))
    testlabel(val_label, len(val_label))
    return train_text , val_text , train_label , val_label
    
def token_preprocessing(text):
    #full bert
    # model_name = 'bert-base-uncased'
    # model_name = "roberta-base"
    # model_name = 'microsoft/deberta-base'
    model_name = run_config['model_name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    #distilled bert
    # model_name = 'distilbert-base-uncased'
    # tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    encoded = tokenizer(text, truncation=True , padding=True)
    # # autotokenizer
    # model_name_or_path = "roberta-large"
    # tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side='right')
    
    # if getattr(tokenizer, "pad_token_id") is None:
    #     tokenizer.pad_token_id = tokenizer.eos_token_id
        
    # encoded = tokenizer(text, truncation=True , padding=True)
    return encoded


def SA_processing(train_path, dev_path):
    # texts, labels = data_preprocessing(data_url)
    # train_text , val_text , train_label , val_label = data_split(texts, labels)
    # dev_url = './SST2_dev.tsv'
    # train_url = './SST2_train.tsv'
    train_text, train_label = data_preprocessing(train_path, n_samples = 200)
    # dev_url = 'https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/dev.tsv'
    val_text, val_label = data_preprocessing(dev_path, n_samples= 200)
    train_encod = token_preprocessing(train_text)
    val_encod = token_preprocessing(val_text)
    train_dataset = CustomData(train_encod, train_label)
    val_dataset = CustomData(val_encod, val_label)
    return train_dataset, val_dataset
    
class CustomData(torch.utils.data.Dataset):
    '''Class to store the tweet data as pytorch dataset'''
    
    def __init__(self , encodings , labels):
        self.encodings = encodings
        self.targets = labels
        
    def __getitem__(self, idx):
        item = {key : torch.tensor(val[idx]) for key , val in self.encodings.items() }
        # item['labels'] = torch.tensor(self.labels[idx])
        target = self.targets[idx]
        return (item, target)
    
    def __len__(self):
        return len(self.targets)
    
    
    
if __name__ == "__main__":
    train_dataset, val_dataset = SA_processing()
    print(train_dataset[0])