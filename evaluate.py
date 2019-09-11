def K_nearest_neighbors(anchor, comparisonLabels, comparisonSamples, K):
    indexRanking = torch.argsort((testSamples - anchor).pow(2).sum(1))
    K_nearest_labels = testLabels[indexRanking[0:K]]
    K_nearest_samples = testSamples[indexRanking[0:K]]
    return (K_nearest_labels, K_nearest_samples)

def outputs_to_prediction(model, outputs):
    return predictions

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

def f1_evaluation(model, labels, outputs, metric, K):
    predictions = outputs_to_predictions(outputs)
    
    # Calculate true positives, false positives and false negatives
    truePositives = ...
    falsePositives = ...
    falseNegatives = ...
    
    precision = truePositives/(truePositives + falsePositives)
    recall = truePositives/(truePositives + falseNegatives)
    
    return 2*precision*recall/(precision + recall)
