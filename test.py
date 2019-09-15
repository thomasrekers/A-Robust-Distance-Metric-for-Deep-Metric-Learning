#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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






def evl_epoch(model,loader):
    '''
    Args:
        model:
        loader:
        
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
    
    # initialize the label and featrs
    cles = torch.FloatTensor()
    cles = cles.to(device)

    featrs = torch.FloatTensor()
    featrs = featrs.to(device)
    
    model.eval()
    with torch.no_grad():
        for batch_idx (inputs,labels) in enumerate(loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            cles = torch.cat((cles, labels), 0)
            
            ouputs = model(inputs)
            featrs = torch.cat((featrs, outputs.data), 0)
        
        #AUROCs = compute_AUCs(gt, pred)
        
    return featrs,cles
 
    


# In[ ]:


def evl(model,train_set,  # model and data
          batch_size ,n_epochs,  # others
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
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Wrap model for multi-GPUs, if necessary
    model_wrapper = model
 
    # Start log
    with open(os.path.join(save, result_f), 'w') as f:
        f.write('epoch,grad,\n')
  
    
    q_featrs,q_labels = evl_epoch(
            model=model_wrapper,
            loader=q_set)
    
    featrs,labels = evl_epoch(
        model = model_wrapper,
        loader = c_set)
    
    # find out the nearest :
    for i in q_featrs:
        #compute the dist between i and featrs
        dis_mat = 
        #  idx of the nearest 
        idx =
        # labels of the k nearest sample
        labels
        # precision and so no
      


def test(data,save,
         metric,loss,sampler='rand',embed_size=16,num_cls = 196,  lambda_=0, lr=0.001,
         model_saved=None,result='r_.csv',
         list_file_1 = './cars_test_q.txt',list_file_2 = './cars_test_c.txt',
         batch_size=500 ):
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

    # dataLoader
    q_set = MyDataset(dataroot=data,phase='test',image_list_file =list_file_1 )
    c_set = MyDataset(dataroot = data,phase='test',image_list_file=list_file_2)
    
    #test_loader = DataLoader(dataset=test_set, batch_size=batch_size,shuffle=False, num_workers=4,pin_memory=torch.cuda.is_available())
    q_loader = DataLoader(dataset=q_set, batch_size=batch_size,shuffle=False, num_workers=4,pin_memory=torch.cuda.is_available())
    c_loader = DataLoader(dataset=c_set, batch_size=batch_size,shuffle=False, num_workers=4,pin_memory=torch.cuda.is_available())
    # model_setup
    if metric=='maha' or metric == 'rM':
        model = models.alexnet(pretrained=False)
        inp_fts =  model.classifier[6].in_features
        model.classifier[6] = nn.Linear(inp_fts, embed_size)
        model.load_state_dict(torch.load(os.path.join(save, model_saved),map_location = 'cpu'))
        
    else:
        model = models.alexnet(pretrained=False)
        inp_fts =  model.classifier[6].in_features
        model.classifier[6] = nn.Linear(inp_fts, embed_size)
        model.load_state_dict(torch.load(os.path.join(save, model_saved),map_location = 'cpu'))
        
        
    print(model)
    # create folder for logging file
    # Make save directory
    if not os.path.exists(save):
        os.makedirs(save)
    if not os.path.isdir(save):
        raise Exception('%s is not a dir' % save)
    
    evl(model=model,q_set=q_loader,c_set=c_loader,  # model and data
          bs = batch_size
        ,
          embed_size=embed_size,loss = loss,metric = metric,sampler =sampler,num_cls = num_cls,   # train
          save=save,result_f=result,model_f=model_name  # log
          )
    print('Done')
    
    
if __name__ == '__main__':
    fire.Fire(test)
  






