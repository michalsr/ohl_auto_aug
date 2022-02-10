from argparse import ArgumentParser
from socketserver import ThreadingUDPServer
import torch
import torch.utils.data
import os 
import torch.nn as nn
from data_loading import CIFAR10,CIFAR10Test
import logging 
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from os import listdir, remove
from shutil import rmtree
from os.path import exists, isdir, join
from tqdm import tqdm 
import numpy as np 
import pdb 
import sys
import math
from resnet import ResNet18
from scheduler import WarmupCosineLR
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
  def sample(self,num_samples):
    lessons = []
    sample_log_probs = torch.zeros(1)
    p = self.sigmoid(self.weights)
    p = p/p.sum(dim=-1,keepdim=True )

    dist = torch.distributions.categorical.Categorical(probs=p)
    print(dist.mean,'dist mean')
    for i in range(num_samples):
      lesson_1 = dist.sample()
      lessons.append(lesson_1.item())
      sample_log_probs = torch.add(sample_log_probs,dist.log_prob(lesson_1))
    self.log_prob.append(sample_log_probs)
    return lessons  
  def forward(self):
    return self.softmax(self.weights)
  def update(self,normalized_reward):
    tensor_reward = []
    loss_values = []
    for r in normalized_reward:
      tensor_reward.insert(int(r.item()),normalized_reward[int(r.item())])
    tensor_reward = torch.tensor(tensor_reward)
    for log_prob, reward  in zip(self.log_prob,tensor_reward):
      r = -log_prob*reward

      loss_values.append(r)
    self.optimizer.zero_grad()
    loss_values_f = torch.stack(loss_values)
    loss = loss_values_f.mean()
    print(f'Outer loss is {loss} ')
    loss.backward()
    self.optimizer.step()
    self.log_prob = []
    self.tensor_reward = []
    return loss.item()
class TrainLoop(nn.Module):
  def __init__(self,output_dir,train_dataset,test_dataset,logger,num_trajec,num_epoch,prefix):
   super().__init__()
   self.output_dir = output_dir
   self.writer = None 
   self.train_dataset = train_dataset
   self.test_dataset = test_dataset
   self.logger = logger 
   self.weights = None
   self.trajec = num_trajec 
   self.epochs = num_epoch
   self.best_model_path = None 
   self.prefix = prefix
   self.lr_scheduler = {'name':'cosine'}
   self.initial_lr = 0.2
  def test(self):
    correct = 0
    total = 0
    self.test_dataset.train=False 
    self.test_dataset.test = True 
    model,optim = self.initialize_model(self.epochs)
    test_loader = torch.utils.data.DataLoader(
          self.test_dataset, batch_size=100, shuffle=False,num_workers=4
      )
    model = model.cuda()
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
    self.logger.info(f'Acc: {correct/total}')
    print(f'Acc:{correct/total}')
    return correct/total 
  def save_outer(self):
    save_dict = {'weights':self.weights.state_dict(),'weight_optim':self.weights.optimizer.state_dict(),'weight_log_prob':self.weights.log_prob}
    torch.save(save_dict,f'{self.prefix}/ohl_auto_aug/{self.output_dir}/outer_loop.pt')
  def save_inner(self,trajec,epoch_num,all_augmentations,train_sampler,val_sampler,new_best_model_path,highest_reward,rewards,optimizer):
    save_dict = {'trajec':trajec,'epoch':epoch_num,'augmentations':all_augmentations,'train_sampler':train_sampler,'val_sampler':val_sampler,'best_model_path':self.best_model_path,
    'new_best_model_path':new_best_model_path,'highest_reward':highest_reward,'reward_list':rewards,'optim':optimizer.state_dict(),'augmentation_list':self.train_dataset.augmentation_list}
    torch.save(save_dict,f'{self.prefix}/ohl_auto_aug/{self.output_dir}/training_loop.pt')
  def compute_normalized_rewards(self,rewards):
    new_rewards = []
    mean = np.mean(rewards)
    for i in range(len(rewards)):
      new_rewards.append(rewards[i]-mean)
    return new_rewards
  def outer_train(self,resume,auto_aug):
    if auto_aug != None:
      auto_aug_train = True 
    else:
      auto_aug_train = False
    best_model_path = None 
    start_epoch = 0
    if not os.path.exists(f'{self.prefix}/ohl_auto_aug/'):
      print(f'Does not exist {self.prefix}/ohl_auto_aug/')
      raise TypeError 
    if not os.path.exists(f'{self.prefix}/ohl_auto_aug/{self.output_dir}/'):
      os.makedirs(f'{self.prefix}/ohl_auto_aug/{self.output_dir}/')
      clear_if_nonempty(f'{self.prefix}/ohl_auto_aug/{self.output_dir}/')
    else:
      inner_dict = torch.load(f'{self.prefix}/ohl_auto_aug/{self.output_dir}/training_loop.pt')
      outer_dict = torch.load(f'{self.prefix}/ohl_auto_aug/{self.output_dir}/outer_loop.pt')
      start_epoch = inner_dict['epoch']  
    if not os.path.exists(f'{self.prefix}/ohl_auto_aug/{self.output_dir}/tensorboard'):
      os.makedirs(f'{self.prefix}/ohl_auto_aug/{self.output_dir}/tensorboard')
    self.writer = SummaryWriter(f'{self.prefix}/ohl_auto_aug/{self.output_dir}/tensorboard')
    self.weights = Weights(36*36,256)
    if resume != None:
      self.weights.load_state_dict(outer_dict['weights'])
    for i in range(start_epoch,self.epochs):
      if resume != None and i<=start_epoch:
        self.train_dataset.augmentation_list = inner_dict['augmentation_list']
        rewards= self.train_epoch(i,inner_dict=inner_dict,resume=True)
      else:
        rewards = self.train_epoch(i)
      normalized_rewards = self.compute_normalized_rewards(rewards)
      self.writer.add_scalar('average weight',torch.mean(self.weights.weights),i)
      avg = torch.mean(self.weights.weights).item()
      self.logger.info(f'Average weight is {avg}')
      self.weights.update(normalized_rewards)
      self.save_outer()
   
    self.test()
  def train_trajec(self,train_sampler,model,optimizer,trajec_num,epoch_num):
    
    self.train_dataset.train=True 
    train_loader = torch.utils.data.DataLoader(
        self.train_dataset, batch_size=256, sampler=train_sampler,
        num_workers=4
    )
    optimizer = self.update_lr(epoch_num,optimizer)
    print(optimizer.param_groups[0]['lr'],' last learning rate')
    model = model.cuda()
    model.train()
    pbar_1 = tqdm(train_loader, ncols=100, desc="loss=", total=len(train_loader))
    criterion = nn.CrossEntropyLoss()  
    running_loss = 0
    for i, data in enumerate(pbar_1):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs =model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
      

        running_loss += loss.item()
        if i%100 == 0 and i>0:
          self.logger.info(f'Loss is {running_loss/i}')
          self.writer.add_scalar(f'loss_trajec_{trajec_num}_epoch_{epoch_num}',running_loss/i,i)
    self.writer.add_scalar(f'loss_trajec_{trajec_num}_epoch_{epoch_num}',running_loss/len(pbar_1),len(pbar_1))
      
     
    return model,optimizer
  def save_best_model(self,model,optimizer,epoch,end_of_epoch=False):
  
    if end_of_epoch:
      old_state_dict = torch.load(f'{self.prefix}/ohl_auto_aug/{self.output_dir}/temp_model.pt',map_location="cuda:0")
      state_dict = {'model':old_state_dict['model'],'optimizer':old_state_dict['optimizer'],'epoch':epoch}
      torch.save(state_dict,f'{self.prefix}/ohl_auto_aug/{self.output_dir}/best_model.pt')
    else:
      state_dict = {'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch}
      torch.save(state_dict,f'{self.prefix}/ohl_auto_aug/{self.output_dir}/temp_model.pt')
    
  def val_trajec(self,val_sampler,model):
    correct = 0
    total = 0
    self.train_dataset.train=False 
    val_loader = torch.utils.data.DataLoader(
          self.train_dataset, batch_size=100, sampler=val_sampler,
          num_workers=4
      )
    model = model.cuda()
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
  def update_lr(self,epoch,optimizer):
    if self.lr_scheduler['name'] == 'cosine':
      new_lr = self.initial_lr*(1 + math.cos(math.pi * epoch / self.epochs)) / 2
    for g in optimizer.param_groups:
      g['lr'] =new_lr 
    return optimizer
  def initialize_model(self,epoch_num):
  
    model = ResNet18()
    model.fc = nn.Linear(512,10)
    model.to("cuda:0")
    optimizer = optim.SGD(model.parameters(), lr=0.2, momentum=0.9, weight_decay=5e-4)
            
    if epoch_num>0:
      print('Initializing from old model')
      state_dict = torch.load(f'{self.prefix}/ohl_auto_aug/{self.output_dir}/best_model.pt',map_location="cuda:0") 
      model.load_state_dict(state_dict['model'])
      optimizer.load_state_dict(state_dict['optimizer'])

 
    return model,optimizer
  def train_epoch(self,epoch_num,inner_dict=None,resume=False):
    if resume == False:
      rewards = []
      train_sampler,val_sampler,num_train = self.train_dataset.get_samplers()
      new_best_model_path = None
      highest_reward = -1
      start_trajec = 0
      weights_step = 0
      train_step = 0 
    else:
      rewards = inner_dict['reward_list']
      train_sampler = inner_dict['train_sampler']
      val_sampler = inner_dict['val_sampler']
      num_train = 45000
      new_best_model_path = inner_dict['new_best_model_path']
      highest_reward = inner_dict['highest_reward']
      start_trajec = inner_dict['trajec']
     
    for trajec in range(start_trajec,self.trajec):
        self.logger.info(f'Trajec {trajec} for epoch {epoch_num}')
        all_augmentations = {}
        aug_list = []
        for v in train_sampler:
          all_augmentations[v] = None
        model,optimizer= self.initialize_model(epoch_num)  
        sampled_augs = self.weights.sample(45000)
        for a in sampled_augs:
          aug_list.append(a)     
        for i,v in enumerate(all_augmentations.keys()):
          all_augmentations[v] = aug_list[i]
        self.train_dataset.augmentations = all_augmentations
        print(self.train_dataset.auto_aug,'auto aug')
        trained_model,optimizer = self.train_trajec(train_sampler,model,optimizer,trajec,epoch_num)
        reward = self.val_trajec(val_sampler,trained_model)
        self.logger.info(f'Reward for trajec {trajec} is {reward}')
        rewards.append(reward)
        self.writer.add_scalar(f'rewards_epoch_{epoch_num}',reward,trajec)
        if reward > highest_reward:
          self.save_best_model(trained_model,optimizer,epoch_num)
          highest_reward =reward 
    self.save_inner(trajec,epoch_num,all_augmentations,train_sampler,val_sampler,new_best_model_path,highest_reward,rewards,optimizer)
    self.writer.add_scalar(f'highest_reward',highest_reward,epoch_num)
    self.best_model_path = new_best_model_path
    self.save_best_model(trained_model,optimizer,epoch_num,end_of_epoch=True)
    return rewards



def main():
    parser = ArgumentParser()
    parser.add_argument("--resume",type=bool,default=None )
    parser.add_argument("--auto_aug",type=bool,default=None)
    parser.add_argument("--prefix",type=str,default=None)
    parser.add_argument("--output_dir",type=str,default=None)
    parser.add_argument("--trajec",type=int,default=8)
    parser.add_argument("--epoch",type=int,default=300)
    args = parser.parse_args()
    if not os.path.exists(f'{args.prefix}/ohl_auto_aug'):
      print(f'{args.prefix}/ohl_auto_aug does not exist')
      raise TypeError
    if not os.path.exists(f'{args.prefix}/ohl_auto_aug/{args.output_dir}'):
      os.makedirs(f'{args.prefix}/ohl_auto_aug/{args.output_dir}')
    logger = setup_logger('logger',f'{args.prefix}/ohl_auto_aug/{args.output_dir}/log_file.log')
    if args.auto_aug != None:
      auto_aug = True 
    else:
      auto_aug = False
    highest_reward = -1 

    cifar_10_train = CIFAR10(train=True,auto_aug=auto_aug,prefix=args.prefix)
    cifar_10_test = CIFAR10Test(train=False,test=True,prefix=args.prefix)
    logger.info(f'Auto augmentation is {auto_aug}')
    loop = TrainLoop(args.output_dir,cifar_10_train,cifar_10_test,logger,int(args.trajec),int(args.epoch),args.prefix)
    loop.outer_train(args.resume,auto_aug)
   
    
    
       


if __name__ == '__main__':
  main()