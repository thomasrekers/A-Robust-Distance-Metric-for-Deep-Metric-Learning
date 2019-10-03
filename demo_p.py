#!/usr/bin/env python
# coding: utf-8

# In[1]:


import create_loss as crloss

import random
import numpy as np
import itertools as it
# In[1]:
import torch
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import datasets, transforms

import os
import time
from data.my_dataset import MyDataset

import fire


# In[3]:


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# In[ ]:


def train_epoch(model,loader,criterion,optimizer,epoch,n_epochs,print_freq=1):
    '''
    Args:
        model:
        loader:
        optimizer:
        criterion:
        
        
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    losses = AverageMeter()
    
    model.train()
    end = time.time()
    batch_idx = 0
    for inputs,labels in loader:
        
        #print(batch_idx)
        # Create vaiables
        
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # feed the network
        batch_size = labels.size(0)
        embeddings = model(inputs) # BS* m
        loss      = criterion(embeddings, labels)
        #record loss
        losses.update(loss.item(), batch_size)
        
            
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        # print stats
        if batch_idx % print_freq == 0:
            res = '\t'.join([
                'Epoch: [%d/%d]' % (epoch + 1, n_epochs),
                'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
                'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                'Loss %.4f (%.4f)' % (losses.val, losses.avg),
            ])
            print(res)
        batch_idx +=1
        
    return losses.avg
 
    


# In[ ]:


def train(model,train_set,  # model and data
          batch_size ,n_epochs,T,  # others
          embed_size,loss,metric,sampler,num_cls,lambda_,   # train
          save,result_f,model_f,
          wd=0,momentum=0.9,lr=0.001):      # optim
    
    '''
    Args:
        model:
        result_f:
        data_sets:
        
        opti_para:
        scheduler_para:
            
    ''' 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    freq = 7
    model = model.to(device)
    
    # Wrap model for multi-GPUs, if necessary
    model_wrapper = model
    
    # Optimizer  : optim strategy to be completed...
    
    to_optim   = [{'params':model_wrapper.parameters(),'lr':lr,'momentum': momentum ,'weight_decay':wd}]
    criterion,to_optim = crloss.select_loss(loss=loss,e_size = embed_size, metric=metric, sampling_method=sampler,num_cls=num_cls,lambda_=lambda_,lr=lr,momentum=momentum,wd=wd,to_optim=to_optim) # check if the new para are in the cuda or not??
    
    criterion = criterion.to(device)
    
    
    optimizer = torch.optim.SGD(to_optim)#model_wrapper.parameters(), lr=lr, momentum=0.9, dampening=0, weight_decay=0, nesterov=False)
    
    # scheduler
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[5,9,12,14])
    # ...
    
    # Start log
    with open(os.path.join(save, result_f), 'w') as f:
        f.write('epoch,train_loss,l_r,dif\n')
    l_rate = lr
    best_loss = 1000
    pre_loss = 10.
    for epoch in range(n_epochs):
       # scheduler.step()
    
        train_loss = train_epoch(
            model=model_wrapper,
            loader=train_set,
            optimizer=optimizer,
            criterion = criterion,
            epoch=epoch,
            n_epochs=n_epochs,
        )
        dif = abs(pre_loss-train_loss)
        if epoch % freq == 0:
            dif = pre_loss-train_loss
            pre_loss = train_loss
            
            if dif< T:
                
                l_rate *= 0.5
        #       for p in optimizer.param_groups:
         #          p['lr']*=0.5
                print('lr has been update \n')
            
            
        # Determine if model is the best and save the best model
        if train_loss < best_loss:
            # save the model
            torch.save(model.state_dict(), os.path.join(save, model_f))
            if metric == 'maha' or metric == 'rM':
                cri_f =  'L.dat'
                torch.save(criterion.state_dict(),os.path.join(save,cri_f))
            # update the best_loss
            best_loss = train_loss
            
        # Log results
        with open(os.path.join(save, result_f), 'a') as f:
            f.write('%03d,%0.6f,%0.9f,%05f\n' % (  #**
                (epoch + 1),
                train_loss,
                l_rate,
                dif
                ))
        
            
        
        


# In[4]:



def demo_p(data,save,fine_tune,
         metric,loss,sampler='rand',embed_size=16,num_cls = 196,  lambda_=0, lr=0.001,T = 0.005,
         model_saved=None,result='r_.csv',model_name='m_.dat',list_file = './cars_train.txt',
         batch_size=100,n_epochs = 10,seed=1):
    '''
    Arg:
        # data
        data: where the data are stored
        dataset: the name of the dataset: CIFAR196,etc.
        train_size:
        test_size:
        
        # log
        save: tr   where the logs are stored
        model_saved: trained_model
        result:  loss
        model_new : 
        
        # trainning para
        embed_size: m : 16,32,64
        
        metric:  'E', 'rE','maha','snr','rM'
        
        sampler:'dist','npair'
        
        loss; 'tripl','contras','npair','lifted',
        
        #margin
            
        # others
        n_epochs :10 tr
        batch_size   tr
        seed :1
        
    
    
    '''
    # seed
    torch.backends.cudnn.deterministic=True
    if seed is not None:
       
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    # dataLoader
    train_set = MyDataset(dataroot=data,phase='train',image_list_file =list_file )
    #test_set  = MyDataset(dataroot=data,phase='test',image_lise_file = list_file)
   
    #test_loader = DataLoader(dataset=test_set, batch_size=batch_size,shuffle=False, num_workers=4,pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size,shuffle=True, num_workers=4,pin_memory=torch.cuda.is_available()) 
    # model_setup
    if model_saved:
        model = models.resnet18(pretrained=False)
        for p in model.parameters():
            p.requires_grad=False
        
        inp_fts =  model.fc.in_features
        #classifier[6].in_features
        #inp_fts =  model.fc.in_features#model.classifier[6] = nn.Linear(inp_fts, embed_size)
        
        model.fc = nn.Linear(inp_fts,embed_size)
        model.load_state_dict(torch.load(os.path.join(save, model_saved),map_location = 'cpu'))
        
        if fine_tune:
            for p in model.parameters():
                p.requires_grad = not p.requires_grad
    else:
        model = models.resnet18(pretrained=True)
        for p in model.parameters():
            p.requires_grad=False
        
        inp_fts =  model.fc.in_features#
        #inp_fts=model.classifier[6].in_features
        model.fc = nn.Linear(inp_fts,embed_size)
        #model.classifier[6] = nn.Linear(inp_fts, embed_size)
        if fine_tune:
            for p in model.parameters():
                p.requires_grad = not p.requires_grad
      
        
    print(model)
    # create folder for logging file
    # Make save directory
    if not os.path.exists(save):
        os.makedirs(save)
    if not os.path.isdir(save):
        raise Exception('%s is not a dir' % save)
    
    train(model=model,train_set=train_loader,  # model and data
          batch_size = batch_size,n_epochs=n_epochs, lambda_=lambda_,lr=lr,T=T, # others
          embed_size=embed_size,loss = loss,metric = metric,sampler =sampler,num_cls = num_cls,   # train
          save=save,result_f=result,model_f=model_name  # log
          )
    print('Done')
    
    
if __name__ == '__main__':
    fire.Fire(demo_p)
  


# In[ ]:




