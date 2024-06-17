import torch
import numpy as np
from sklearn.metrics import f1_score, \
        precision_score,recall_score,\
              accuracy_score,\
                roc_auc_score

# Each class needs label, prediction probabilities
# Label: (N, 1)
# Prediction: (N, C)
# Each class, calculate TP, TN, FP, FN based on the probability

def classification_metrics_1_class(class_index, scores, labels, score_is_prob = False):
    '''
    class_index: int
    labels: tensor(N)
    scores: tensor(N, C)
    ---
    Return a tuple of (f1, precision, recall, accuracy, precision_recall_curve, roc_auc_score)
    '''
    probs = scores if score_is_prob else torch.nn.functional.softmax(scores, dim = 1)
    label_binary = labels == class_index
    predictions = torch.argmax(probs, dim = 1) == class_index
    label_binary = label_binary.to("cpu")
    predictions = predictions.to("cpu")
    probs = probs.to("cpu")
    f1 = f1_score(label_binary, predictions, zero_division=np.nan)
    precision = precision_score(label_binary, predictions, zero_division=np.nan)
    recall = recall_score(label_binary, predictions, zero_division=np.nan)
    accuracy = accuracy_score(label_binary, predictions)
    auc = roc_auc_score(label_binary, probs[:, class_index]) if class_index in labels else np.nan
    return f1, precision, recall, accuracy, auc 

def classification_metrics_n_class(categories, scores, labels, prefix, score_is_prob = False):
   '''
   labels: tensor(N)
   scores: tensor(N, C)
   ---
   Return lists of f1, precision, recall, accuracy, roc_auc_score 
   '''    
   f1_scores = {}
   precisions = {}
   recalls = {}
   accuracies = {}
   roc_auc_scores = {}
   probs = scores if score_is_prob else torch.nn.functional.softmax(scores, dim = 1)
   for i in range(len(categories)):
       f1, precision, recall, accuracy, auc = classification_metrics_1_class(i, probs, labels, score_is_prob= True)
       category = categories[i]
       f1_scores[f'{prefix}-f1-{category}'] = f1
       precisions[f'{prefix}-precision-{category}'] = precision
       recalls[f'{prefix}-recall-{category}'] = recall
       accuracies[f'{prefix}-accuracy-{category}'] = accuracy
       roc_auc_scores[f'{prefix}-auc-{category}'] = auc
   return f1_scores, precisions, recalls, accuracies, roc_auc_scores

# Error analysis (pie chart, which class it often mistakenly for)

