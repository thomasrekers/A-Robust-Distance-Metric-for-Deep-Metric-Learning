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

def K_nearest_neighbors(anchor, comparisonLabels, comparisonSamples, K):
    # useful matrices for filtering anchor out of comparisonSamples
    oneMatrix = torch.ones(comparisonSamples.shape)
    zeroMatrix = torch.zeros(comparisonSamples.shape)
    oneVector = torch.ones(comparisonLabels.shape)
    zeroVector = torch.zeros(comparisonLabels.shape)
    
    # filter anchor out of the comparisonSamples
    # count number of equal elements per row
    rowSum = torch.where(comparisonSamples == anchor, oneMatrix, zeroMatrix).sum(1)
    duplicates = torch.where(rowSum != comparisonSamples.shape[1], oneVector, zeroVector)
    # remove entry from labels or row from samples if anchor equals sample
    filteredLabels = comparisonLabels[duplicates.nonzero().squeeze().detach()]
    filteredSamples = comparisonSamples[duplicates.nonzero().squeeze().detach()]
    
    # calculate K nearest neighbors
    # sort samples by distance from anchor
    dist = crloss.Metric('snr')
    indexRanking = torch.argsort((filteredSamples - anchor).pow(2).sum(1))
    # select the K nearest samples
    K_nearest_labels = filteredLabels[indexRanking[0:K]]
    K_nearest_samples = filteredSamples[indexRanking[0:K]]
    return (K_nearest_labels, K_nearest_samples)
  
def recall_evaluation(batchLabels, batch, comparisonLabels, comparisonSamples, K):
    N = batchLabels.shape[0]
    absoluteScore = 0
    for i in range(0, N):
        anchor = batch[i]
        K_nearest_labels, K_nearest_samples = K_nearest_neighbors(anchor, comparisonLabels, comparisonSamples, K)
        for j in range(0, K):
            if batchLabels[i] == K_nearest_labels[j]:
                absoluteScore = absoluteScore + 1
                break
    return absoluteScore/N

def calculate_precision(anchorLabel, anchor, comparisonLabels, comparisonSamples, K):
    absoluteScore = 0
    K_nearest_labels, K_nearest_samples = K_nearest_neighbors(anchor, comparisonLabels, comparisonSamples, K)
    for i in range(0, K):
        if anchorLabel == K_nearest_labels[i]:
            absoluteScore = absoluteScore + 1
    return absoluteScore/(K+1)

def compareLabels(label1, label2):
    return 1 if label1 == label2 else 0

def aP_K(anchorLabel, anchor, comparisonLabels, comparisonSamples, K):
    numeratorSum = 0
    denominatorSum = 0
    K_nearest_labels, _ = K_nearest_neighbors(anchor, comparisonLabels, comparisonSamples, K)
    for i in range(0, K):
        precision = calculate_precision(anchorLabel, anchor, comparisonLabels, comparisonSamples, i)
        delta = compareLabels(anchorLabel, K_nearest_labels[i])
        numeratorSum = numeratorSum + precision*delta
        denominatorSum = denominatorSum + delta
    return 0 if denominatorSum == 0 else numeratorSum/denominatorSum

def mAP_evaluation(batchLabels, batch, comparisonLabels, comparisonSamples, K):
    N = batchLabels.shape[0]
    aPSum = 0
    for i in range(0, N):
        aPSum = aPSum + aP_K(batchLabels[i], batch[i], comparisonLabels, comparisonSamples, K)
    return aPSum/N

def f1_evaluation(batchLabels, batch, comparisonLabels, comparisonSamples, K):
    N = batchLabels.shape[0]
    precisionSum = 0
    for i in range(0, N):
        precisionSum = precisionSum + calculate_precision(batchLabels[i], batch[i], comparisonLabels, comparisonSamples, K)
    precision = precisionSum/N
    recall = recall_evaluation(batchLabels, batch, comparisonLabels, comparisonSamples, K)
    return 2*precision*recall/(precision + recall)

def start_evaluation(data, labels_test, model_path, K, batchSize, embSize, loaderSize):
    # load model
    model = models.alexnet(pretrained=False)
    inp_fts =  model.classifier[6].in_features
    model.classifier[6] = nn.Linear(inp_fts, embSize)
    model.load_state_dict(torch.load(model_path,map_location = 'cpu'))
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    test_set  = MyDataset(dataroot=data,phase='test',image_list_file = 'labels_test.txt')
    test_loader = DataLoader(dataset=test_set, batch_size=loaderSize,shuffle=True, num_workers=4,pin_memory=torch.cuda.is_available())
    
    model.eval()
    counter = 0
    labelData = torch.zeros(0)
    sampleData = torch.zeros(0, embSize)
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Create vaiables
            if torch.cuda.is_available():
                inputs = inputs.cuda()

            # feed the network
            embeddings = model(inputs) # BS* m
            counter = counter + 1
            print(counter)
            if counter == 9:
              break
            embeddings = embeddings.to(torch.device("cpu"))
            labels = torch.reshape(labels, (-1,))
            labelData = torch.cat((labelData, labels))
            sampleData = torch.cat((sampleData, embeddings))

    batchLabels = labelData[0:batchSize]
    batch = sampleData[0:batchSize]

    print(recall_evaluation(batchLabels, batch, labelData, sampleData, K))

start_evaluation('./../cars', './labels_test', './../m_3.dat', 2, 500, 16, 100)
