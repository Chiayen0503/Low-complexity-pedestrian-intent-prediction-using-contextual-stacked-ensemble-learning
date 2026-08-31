from torchmetrics.functional import precision_recall, accuracy
from torchmetrics import F1Score, AUROC
from torchmetrics.classification import BinaryAUROC
from sklearn.metrics import accuracy_score
import torch

def threshold_based_acc(y_probs, targets, threshold = 0.5):
    y_pred = (y_probs >= threshold).int()
    acc = accuracy(
        y_pred, 
        targets, 
        task='binary'
    ).item()
    return acc

def get_precision_recall_F1_v2(targets, probs, thres, device): #probs: predicted confidence level, thres: threshold of confidence. 
    """
    This function uses torchmetrics funcs: precision_recall, F1Score, AUROC.
    The func support confidence threshold feature.
    """
    probs, targets = probs.to(device), targets.to(device)
    
    p = precision_recall(probs, targets, average='macro', threshold=thres, num_classes=1)[0].to(device).item()
    r = precision_recall(probs, targets, average='macro', threshold=thres, num_classes=1)[1].to(device).item()
    f1_func = F1Score(task="binary", threshold=thres, average='macro').to(device)
    f1 = f1_func(probs, targets).item()
    auroc = AUROC(task="binary", threshold=thres)
    auc = auroc(probs, targets)
    #threshold based acc
    acc=threshold_based_acc(probs, targets, threshold=thres)
    print('acc: {:.2f}'.format(acc))
    print('AUC: {:.2f}'.format(auc))
    print('F1: {:.2f}'.format(f1))
    print('precision: {:.2f}'.format(p))
    print('recall: {:.2f}'.format(r))
    
    return "{:.2f}".format(acc), "{:.2f}".format(auc), "{:.2f}".format(f1),  "{:.2f}".format(p), "{:.2f}".format(r)



def f1_greedy_searching(probs, targets):
    device = torch.device('cuda')
    best_f1 = 0
    best_thres = 0
    
    thres_list = [i/100 for i in range(0, 100, 1)]
    for thres in thres_list:
        f1_func = F1Score(task="binary", threshold=thres, average='macro').to(device)
        f1 = f1_func(probs, targets).item()
        if f1>best_f1:
            best_f1 = f1
            best_thres = thres
        print(thres, f1)
    return best_f1, best_thres