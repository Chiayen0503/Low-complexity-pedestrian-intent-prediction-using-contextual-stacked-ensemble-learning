import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score

"""
Parameters:
    probs: predicted confidence
    preds: predicted labels
    targets: ground truth labels
    device: gpu or cpu 
"""

def calculate_acc(preds, targets): 
    train_corrects = 0
    train_corrects += (preds == targets).sum().item()
    acc = train_corrects/len(targets)
    return acc


def get_precision_recall_F1(preds, targets, probs): 
    """
    1. This function uses sklearn funcs: precision_score, recall_score, f1_score and roc_auc_score. 
    2. The func does not support confidence threshold feature, i.e, we can't calculate scores by adjusting a confidence level threshold.
    3. This function treats each class the same, i.e, it is a balanced score function for producing average performance of all classes => specifically deal with imbalanced data
    
    
    """
    p = precision_score(targets, preds, average='macro')
    r = recall_score(targets, preds, average='macro')
    f1 = f1_score(targets, preds, average='macro')
    targets= targets.squeeze()
    try:
        probs = probs.squeeze()
        auc = roc_auc_score(targets,probs)
        return "{:.2f}".format(p), "{:.2f}".format(r), "{:.2f}".format(f1),  "{:.2f}".format(auc)
    except:
        return "{:.2f}".format(p), "{:.2f}".format(r), "{:.2f}".format(f1),  0

#     print('precision: {:.2f}'.format(p))
#     print('recall: {:.2f}'.format(r))
#     print('F1: {:.2f}'.format(f1))
#     print('AUC: {:.2f}'.format(auc))
#     return "{:.2f}".format(p), "{:.2f}".format(r), "{:.2f}".format(f1),  "{:.2f}".format(auc)