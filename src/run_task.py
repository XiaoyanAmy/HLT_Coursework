        
from sklearn import random_projection
from datetime import datetime
from tqdm import tqdm
from util import *
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from data_prepare import data_preprocessing, token_preprocessing, CustomData

from pytorch_metric_learning import losses
from loss_funcs import FocalLossAdaptive




os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_model(model, model_path):
    model.load_state_dict(torch.load(model_path)['model'])
   
def save_model(model, optimizer, epoch, lr_sched, eta=[], save_file = None):
    print('==> Saving...')
    state = {
        # 'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'eta': eta,
        'lr_sched': lr_sched.state_dict() 
    }
    torch.save(state, save_file)
    del state
    
def test(model, test_data_path, test_batch_size):
  test_text, test_label = data_preprocessing(test_data_path, n_samples = None)
  test_encod = token_preprocessing(test_text)
  test_dataset = CustomData(test_encod, test_label)
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)
  test_acc = validate(model, test_loader)
  print(">>>>>>>>>>>>accuracy<<<<<<<<<<", test_acc)
  return test_acc
  
def validate(model, vali_loader):
    model.eval()
    correct = 0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(vali_loader):
            # data, target = data.to(device), target.to(device)
            # bsz = data.shape[0]  
            target = target.to(device) 
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1].to(device)
            target = target.view(-1,1).to(device)
            # target_new = target.to(device).data.max(1, keepdim=True)[1]
            correct += pred.eq(target).sum()
        
    return correct / len(vali_loader.dataset)

def train(model, optimizer, scheduler, train_loader, vali_loader, epoch, model_path):
      # model.to(device)
  model.train()
  now = datetime.now()
  dt = now.strftime("%d-%m-%y_%H-%M-%S")
  logger = SummaryWriter(log_dir=f"./tf-logs/{dt}")
#   weights = torch.FloatTensor([7, 13])
  criterion = nn.CrossEntropyLoss()
#   criterion = FocalLossAdaptive()
#   criterion = FocalLoss(gamma=0.7, weights=weights)
  loss_feat = losses.TripletMarginLoss()
  # loss_feat = losses.TupletMarginLoss()
  best_vali_er = np.inf
  best_vali_acc = 0
  m = torch.nn.Sigmoid()
  for ep in tqdm(range(epoch)):
    model.train()
    train_loss = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
      optimizer.zero_grad()
      bsz = target.shape[0]
      target = target.to(device)
      # data,  target = data.to(device), target.to(device)
      output = model(data).to(device)
      loss = criterion(output, target)
      #**
      feature = model(data, Feature_return = True)
      loss_emb = loss_feat(feature, target)
    #   ##** auto eta
      # loss_all = [loss, loss_emb]
      # loss = (torch.stack(loss_all) * torch.exp(-model.eta) + 0.5*model.eta).sum()
      ##**
      loss += 1*loss_emb
      #**
    
      train_loss += bsz*loss
      loss.backward()
      optimizer.step()
    train_loss /= len(train_loader.dataset)
    
    vali_acc = validate(model, vali_loader)
    logger.add_scalar("vali_acc", vali_acc.item(), global_step=ep, walltime=None)
    # logger.add_scalar("train_total_loss", train_total_loss.item(), global_step=ep, walltime=None)
    logger.add_scalar("train_loss", train_loss.item(), global_step=ep, walltime=None)
    # print(train_main_loss.item(), train_cl_loss.item())
    print('Train Epoch: {}\ttr_loss:{:.6f}\tva_acc: {:.6f}'.format(ep, train_loss.item(), vali_acc.item()))
    
    # save the best model
    if vali_acc > best_vali_acc:
        best_vali_acc = vali_acc
        # print(">>>>>>>>>>best<<<<<<<<<<")
        save_model(model, optimizer, ep, scheduler, model.eta, model_path) 
         
    scheduler.step()     

def resume_model(model_path, model, optim, scheduler):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])
    optim.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['lr_sched'])
    
def run_task(model_path, model, train_loader, test_loader, lr = 2e-5, epoch = 30):  
   
    opt = AdamW(model.parameters(), lr=lr, correct_bias=False)
    total_steps = len(train_loader) * epoch
    scheduler = get_linear_schedule_with_warmup(
                opt,
                num_warmup_steps= 0.0*total_steps,
                num_training_steps=total_steps
                )
    if os.path.isfile(model_path):
        resume_model(model_path, model, opt, scheduler)
        # test(model, test_loader)
    
    train(model, opt, scheduler, train_loader, test_loader, epoch, model_path)
