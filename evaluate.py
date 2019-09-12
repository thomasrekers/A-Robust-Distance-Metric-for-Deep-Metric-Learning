import torch

def K_nearest_neighbors(anchor, comparisonLabels, comparisonSamples, K):
    oneMatrix = torch.ones(comparisonSamples.shape)
    zeroMatrix = torch.zeros(comparisonSamples.shape)
    oneVector = torch.ones(comparisonLabels.shape)
    zeroVector = torch.zeros(comparisonLabels.shape)
    
    rowSum = torch.where(comparisonLabels == anchor, oneMatrix, zeroMatrix).sum(1)
    duplicates = torch.where(rowSum != comparisonSamples.shape[1], oneVector, zeroVector)
    filteredLabels = comparisonLabels[duplicates.nonzero().squeeze().detach()]
    filteredSamples = comparisonSamples[duplicates.nonzero().squeeze().detach()]
    
    indexRanking = torch.argsort((filteredSamples - anchor).pow(2).sum(1))
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
    return absoluteScore/K

def compareLabels(label1, label2):
    return 1 if label1 == label2 else 0

def aP_K(anchorLabel, anchor, comparisonLabels, comparisonSamples, K):
    numeratorSum = 0
    denominatorSum = 0
    K_nearest_labels, _ = K_nearest_neighbors(anchor, comparisonLabels, comparisonSamples, K)
    for i in range(0, K):
        precision = calculate_precision(anchorLabel, anchor, comparisonLabels, comparisonSamples, i)
        delta = comparisonLabels(anchor, K_nearest_labels[i])
        numeratorSum = numeratorSum + precision*delta
        denominatorSum = denominatorSum + delta
    return numeratorSum/denominatorSum

def mAP_evaluation(batchLabels, batch, comparisonLabels, comparisonSamples, K):
    N = batchLabels.shape[0]
    aPSum = 0
    for i in range(0, N):
        aPSum = aPSum + aP_K(batchLabels[i], batch[i], comparisonLabels, comparisonSamples, K)
    return aPSum/N

def f1_evaluation(batchLabels, batch, comparisonLabels, comparisonSamples, K):
    N = batchLabels.shape[0]
    precisionSum = 0
    for i in range[0, N]:
        precisionSum = precisionSum + calculate_precision(batchLabels[i], batch[i], comparisonLabels, comparisonSamples, K)
    precision = precisionSum/N
    recall = recall_evaluation(batchLabels, batch, comparisonLabels, comparisonSamples, K)
    return 2*precision*recall/(precision + recall)
