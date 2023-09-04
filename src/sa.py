from transformers import BertModel, AutoModel, BertForSequenceClassification, BertConfig, AutoModelForSequenceClassification, AutoTokenizer
from run_task import run_task, set_model, test_model
from data_prepare import SA_processing
from util import *
from tuning import model_tuning
from model import SA_classifier
run_config = load_config()
print("=======")
print(run_config)
print("==========")


warnings.filterwarnings('ignore')
seed = run_config["seed"]
torch.manual_seed(seed)
BATCH_SIZE = run_config["batch_size"]

if __name__ == "__main__":
    
    os.makedirs(task_path, exist_ok=True)
    model_save_name = run_config['model_save_name']
    task_path = os.path.join(task_path, model_save_name)
    train_dataset, val_dataset = SA_processing(train_path, val_path)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size= BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    
    model_name = run_config['model_name']
    extractor = AutoModel.from_pretrained(model_name)
    
  
    # sac = prompt(model)
    tuning_method = run_config['tuning_method']
    sac = model_tuning(extractor, tuning_method)
   
    # bert params trainable
    n_components = run_config['num_hidden_states']
    layer_sizes = [n_components, 2]
    
    classifier = run_config['classifier']
    
    task = SA_classifier(sac, layer_sizes, classifier).to(device)
    lr = run_config["learning_rate"]
    epoch = run_config['epoch']
    
    run_task(task_path, task, train_loader, val_loader, lr, epoch, classifier)
    set_model(task, task_path) 
    test_model(task, test_path, BATCH_SIZE)
    
