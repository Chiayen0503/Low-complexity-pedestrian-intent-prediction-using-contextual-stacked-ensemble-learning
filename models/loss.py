import torch
import torch.nn as nn
import torch.nn.functional as F
class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda() #alpha is big if first class is minority class, wheras is small when first class is majority class
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, 
reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()
    
    
def calculate_class1_inverse_class_freq(y_data): #y_data is (y_train + y_val)
    """
    suppose class1 is neg class
    """
    neg = 0 #class 1
    pos = 0 #class 2 
    for label in y_data:
        if label[0] == 0:
            neg+=1
        else:
            pos+=1
    return pos/(neg+pos)
