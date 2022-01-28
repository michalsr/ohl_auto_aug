from argparse import ArgumentParser
from socketserver import ThreadingUDPServer
import torch
import torch.utils.data
import torch.nn as nn
from data_loading import CIFAR10
import logging 
import torch.optim as optim
from os import listdir, remove
from shutil import rmtree
from os.path import exists, isdir, join
from tqdm import tqdm 
import numpy as np 
import sys
def get_yes_no(msg):
  while True:
    txt = input(msg).strip().lower()
    if txt in {"y", "yes"}:
      return True
    if txt in {"n", "no"}:
      return False

def clear_if_nonempty(output_dir, override=False):
  if output_dir:
    if exists(output_dir) and listdir(output_dir):
      if override or get_yes_no("%s is non-empty, override (y/n)?" % output_dir):
        for x in listdir(output_dir):
          if isdir(join(output_dir, x)):
            rmtree(join(output_dir, x))
          else:
            remove(join(output_dir, x))
      else:
        raise ValueError(f"Output directory ({output_dir}) already exists and is not empty.")

formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)
    handler_2 = logging.StreamHandler(sys.stdout)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(handler_2)

    return logger
class Weights(nn.Module):
  def __init__(self,operations,batch_size) -> None:
      super().__init__()
      self.weights = nn.Parameter(torch.ones(operations)*(1/operations))
      #self.weights.fill_(1/num_lessons)
      self.params = [self.weights] 
      self.optimizer = torch.optim.Adam(self.params,lr=0.05,betas=(0.5,0.999))
      self.sigmoid = nn.Sigmoid()
      self.batch_size =256
      self.log_prob = []
  def sample(self):
    lessons = []
    sample_log_probs = torch.zeros(1)

    p = self.sigmoid(self.weights)
    dist = torch.distributions.categorical.Categorical(probs=p)
    for i in range(self.batch_size):
      lesson_1 = dist.sample()
      lesson_2 = dist.sample()
      lessons.append((lesson_1.item(),lesson_2.item()))
      sample_log_probs = torch.add(sample_log_probs,dist.log_prob(lesson_1))
      sample_log_probs = torch.add(sample_log_probs,dist.log_prob(lesson_2))
    self.log_prob.append(sample_log_probs)
    #print(self.log_prob,'after sampl')
    #print(trajec,len(self.log_prob))
    #assert len(self.log_prob) == trajec + 1
    #print(self.log_prob[f'trajec_{j}'].requires_grad)
    #print(self.weights.grad[0]!= None,'weight grad')
    #print(self.log_prob,'log prob')
    return lessons  
  def forward(self):
    return self.softmax(self.weights)
  def update(self,normalized_reward):
    tensor_reward = []
    loss_values = []
    #pdb.set_trace()
    for r in normalized_reward:
      # value = r.split('_')
      #print(value[1])
      #type(int(float(value[1])))
      tensor_reward.insert(int(r.item()),normalized_reward[int(r.item())])
    tensor_reward = torch.tensor(tensor_reward)
    #print(self.log_prob,'log prob')
    for log_prob, reward  in zip(self.log_prob,tensor_reward):
      #print('hello I am here')
     
      #print(reward.requires_grad)
      #print(log_prob.requires_grad,'do i require grad here')
      #print(-log_prob,'negative log prob')
      r = -log_prob*reward

      loss_values.append(r)
    # print(loss_values,'loss values')
    # for v in loss_values:
    #   print(type(v),v.requires_grad)
    self.optimizer.zero_grad()
    print(loss_values,'loss values')
    loss_values_f = torch.stack(loss_values)
    loss = loss_values_f.mean()
    print(f'Outer loss is {loss} ')
    #print(loss,'loss')
    #print(self.weights,'weights before')
    loss.backward()
    #self.weights.grad = torch.autograd.grad(loss,self.parameters())[0]
    #print(self.weights.grad,'grad')
    self.optimizer.step()
    #print(self.weights,'weights after')
    self.log_prob = []
    self.tensor_reward = []
    return loss.item()
def test(dataset,best_model_path,logger):
  correct = 0
  total = 0
  dataset.train=False 
  model,optim,sched = initialize_model(best_model_path,300)
  test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=256, shuffle=True,
        num_workers=2
    )
  model.cuda()
  model.eval()
  with torch.no_grad():
    for data in tqdm(test_loader):
        inputs, labels = data
        # calculate outputs by running images through the network
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = model(inputs)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
  logging.info(f'Acc: {correct/total}')
  print(f'Acc:{correct/total}')
  return correct/total 
def val_trajec(dataset,val_sampler,model):
  correct = 0
  total = 0
  dataset.train=False 
  val_loader = torch.utils.data.DataLoader(
        dataset, batch_size=256, sampler=val_sampler,
        num_workers=2
    )
  model.cuda()
  model.eval()
  with torch.no_grad():
    for data in val_loader:
        inputs, labels = data
        # calculate outputs by running images through the network
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = model(inputs)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
  return correct/total 
def train_trajec(train_dataset,train_sampler,model,num_train,logger,optimizer,scheduler):
    print(scheduler.get_last_lr())
    train_dataset.train=True 
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=256, sampler=train_sampler,
        num_workers=4
    )
    model = model.cuda()
    model.train()
    pbar = tqdm(train_loader, ncols=100, desc="loss=", total=len(train_loader))
    criterion = nn.CrossEntropyLoss()
    
    #print(sched.state_dict())   
    running_loss = 0
    for i, data in enumerate(pbar):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
       

        running_loss += loss.item()
        if i%100 == 0 and i>0:
          logger.info(f'Loss is {running_loss/i}')

        # print statistics
        
        # if i % 2000 == 1999:    # print every 2000 mini-batches
        #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
        #     running_loss = 0.0
    return model,optimizer,scheduler
def initialize_model(best_model_path,epoch_num):
  
  model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
  model.to("cuda:0")
  optimizer = optim.SGD(model.parameters(), lr=0.2, momentum=0.9,weight_decay=.0005)
  sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(300), eta_min=0, last_epoch=- 1, verbose=False)

  if epoch_num>0:
    state_dict = torch.load('/shared/rsaas/michal5/ohl_auto_aug/outputs/best_model.pt',map_location="cuda:0") 
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    sched.load_state_dict(state_dict['scheduler'])
  return model,optimizer,sched 
def save_best_model(model,optimizer,scheduler,epoch,end_of_epoch=False):
  
  if end_of_epoch:
    state_dict = torch.load('/shared/rsaas/michal5/ohl_auto_aug/outputs/temp_model.pt',map_location="cuda:0")
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    scheduler.load_state_dict(state_dict['scheduler'])
    scheduler.step()
    state_dict = {'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch}
    torch.save(state_dict,'/shared/rsaas/michal5/ohl_auto_aug/outputs/best_model.pt')
  else:
    state_dict = {'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch}
    torch.save(state_dict,'/shared/rsaas/michal5/ohl_auto_aug/outputs/temp_model.pt')
    

def train_epoch(weights,best_model_path,highest_reward,epoch_num,logger,inner_dict=None,resume=False):
    best_model = None 
    cifar_10 =CIFAR10(train=True)
    if resume == False:

      rewards = []

      train_sampler,val_sampler,num_train = cifar_10.get_samplers()
      new_best_model_path = None
      highest_reward = -1
      start_trajec = 0
    else:

      rewards = inner_dict['reward_list']
      train_sampler = inner_dict['train_sampler']

      val_sampler = inner_dict['val_sampler']
      num_train = 45000
      new_best_model_path = inner_dict['new_best_model_path']
      highest_reward = inner_dict['highest_reward']
      start_trajec = inner_dict['trajec']


    




  
    for trajec in range(start_trajec,8):
        logger.info(f'Trajec {trajec} for epoch {epoch_num}')
        all_augmentations = {}
        aug_list = []
        for v in train_sampler.indices:
          all_augmentations[v] = None

        model,optimizer,scheduler= initialize_model(best_model_path,epoch_num)
        

        num_samples = 45000/256
        for i in range(int(num_samples)+1):
          batch_augs = weights.sample() 
          for b in batch_augs:
            aug_list.append(b)
        for i,v in enumerate(all_augmentations.keys()):
          all_augmentations[v] = aug_list[i]

        # print(len(all_augmentations),num_train)
        # assert len(all_augmentations) >= num_train
        cifar_10.augmentations = all_augmentations
        trained_model,optimizer,scheduler = train_trajec(cifar_10,train_sampler,model,num_train,logger,optimizer,scheduler)
        reward = val_trajec(cifar_10,val_sampler,trained_model)
        logger.info(f'Reward for trajec {trajec} is {reward}')
        rewards.append(reward)
        if reward > highest_reward:
          save_best_model(model,optimizer,scheduler,epoch_num)
          #scheduler.step()
          # state_dict = {'model':model.state_dict(),'optim':optimizer.state_dict(),'sched':scheduler.state_dict(),'epoch':epoch_num,'trajec':trajec}
          # torch.save(state_dict,f'/shared/rsaas/michal5/ohl_auto_aug/outputs/temp_model.pt')
          #new_best_model_path = f'/shared/rsaas/michal5/ohl_auto_aug/outputs/best_model.pt'
          
          highest_reward =reward 
        save_inner(trajec,epoch_num,all_augmentations,train_sampler,val_sampler,best_model_path,new_best_model_path,highest_reward,rewards,optimizer,scheduler)
    
    best_model_path = new_best_model_path
    save_best_model(model,optimizer,scheduler,epoch_num,end_of_epoch=True)
    return rewards,best_model_path,highest_reward,cifar_10
def save_inner(trajec,epoch_num,all_augmentations,train_sampler,val_sampler,best_model_path,new_best_model_path,highest_reward,rewards,optim,sched):
    save_dict = {'trajec':trajec,'epoch':epoch_num,'augmentations':all_augmentations,'train_sampler':train_sampler,'val_sampler':val_sampler,'best_model_path':best_model_path,
    'new_best_model_path':new_best_model_path,'highest_reward':highest_reward,'reward_list':rewards,'optim':optim.state_dict(),'sched':sched.state_dict()}
    torch.save(save_dict,'/shared/rsaas/michal5/ohl_auto_aug/outputs/training_loop.pt')
def save_outer(weights):
    save_dict = {'weights':weights.state_dict(),'weight_optim':weights.optimizer.state_dict(),'weight_log_prob':weights.log_prob}
    torch.save(save_dict,'/shared/rsaas/michal5/ohl_auto_aug/outputs/outer_loop.pt')
def compute_normalized_validation(rewards):
    new_rewards = []
    mean = np.mean(rewards)
    for i in range(len(rewards)):
      new_rewards.append(rewards[i]-mean)
    return new_rewards
        #sample 2 times for every image (2 times 45,000)
        #apply 


def main():
    parser = ArgumentParser()
    parser.add_argument("--resume",type=bool,default=None )
    logger = setup_logger('logger','/shared/rsaas/michal5/ohl_auto_aug/outputs/log_file.log')
    args = parser.parse_args()
    highest_reward = -1 
    best_model_path = None 
    start_epoch = 0
    if args.resume == None:
      
      clear_if_nonempty('/shared/rsaas/michal5/ohl_auto_aug/outputs/')
    else:
      inner_dict = torch.load('/shared/rsaas/michal5/ohl_auto_aug/outputs/training_loop.pt')
      outer_dict = torch.load('/shared/rsaas/michal5/ohl_auto_aug/outputs/outer_loop.pt')
      start_epoch = inner_dict['epoch']
  
    

       
    w = Weights(36,256)
    if args.resume != None:

      w.load_state_dict(outer_dict['weights'])
    for i in range(start_epoch,300):
      if args.resume != None:
        rewards,best_model_path,highest_reward,dataset = train_epoch(w,best_model_path,highest_reward,i,logger,inner_dict=inner_dict,resume=True)
      else:
         rewards,best_model_path,highest_reward,dataset = train_epoch(w,best_model_path,highest_reward,i,logger)
      normalized_rewards = compute_normalized_validation(rewards)
      w.update(normalized_rewards)
      save_outer(w,)
    test(dataset,best_model_path,logger)
    
      
    
       


if __name__ == '__main__':
  main()