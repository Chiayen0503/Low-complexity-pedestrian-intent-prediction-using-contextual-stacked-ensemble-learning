import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import numpy as np
import utils.model_saver as MS
from models.loss import WeightedFocalLoss
import random
from model_evaluation.performance_scores import get_precision_recall_F1
from models.gru_attention_model import GRU

def get_norm_adj():
    adj = torch.tensor([[0,1,0,0,0,0,0,0,0,0,0,0,0,0],#1
                        [1,0,1,1,0,0,0,0,1,1,0,0,0,0],#0,2,3,8,9
                        [0,1,0,0,1,0,0,0,0,0,0,0,0,0],#1,4
                        [0,1,0,0,0,1,0,0,0,0,0,0,0,0],#1,5
                        [0,0,1,0,0,0,1,0,0,0,0,0,0,0],#2,6
                        [0,0,0,1,0,0,0,1,0,0,0,0,0,0],#3,7
                        [0,0,0,0,1,0,0,0,0,0,0,0,0,0],#4
                        [0,0,0,0,0,1,0,0,0,0,0,0,0,0],#5
                        [0,1,0,0,0,0,0,0,0,1,1,0,0,0],#1,9,10
                        [0,1,0,0,0,0,0,0,1,0,0,1,0,0],#1,8,11
                        [0,0,0,0,0,0,0,0,1,0,0,0,1,0],#8,12
                        [0,0,0,0,0,0,0,0,0,1,0,0,0,1],#9,13
                        [0,0,0,0,0,0,0,0,0,0,1,0,0,0],#10
                        [0,0,0,0,0,0,0,0,0,0,0,1,0,0]])#11
    I = torch.eye(14)#identity matrix
    adj = adj+I

    D_hat = I
    for i in range(14):
        rowsum = 0
        for j in range(14):
            rowsum += adj[i,j]
        D_hat[i,i] = rowsum**(-0.5)


    adj = np.dot(D_hat, adj)
    norm_adj = np.dot(adj, D_hat)
    norm_adj = torch.FloatTensor(norm_adj)
    return norm_adj

    
class PedestrianGraph(nn.Module):
    def __init__(self, input_size=224, hidden_size=16, num_layers=1, num_classes=1, device=torch.device('cuda')):
        super(PedestrianGraph, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, 5, padding='same') #input channel, output channel, kernel size
        self.adjust1 = nn.Conv2d(2, 16, 1, padding='same')#for identity map to match post-conv shape
        
        self.fc = nn.Linear(2, 1)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.gru  = GRU(input_size, hidden_size, num_layers, num_classes, device)
        self.fc = nn.Linear(hidden_size , num_classes)
        
    def forward(self, x, norm_adj):
        #first block: GCN layer
        x_identity_1 = x #(batch_size, frame=32, keypoints=14, 2)
        x = torch.permute(x, (2, 0, 1, 3)) #(batch_size, 32, 14, 2) -> (14, batch_size, 32, 2) 
        x = torch.einsum('ip,pjkl->ijkl',norm_adj,x) #(14x14) * (14, batch_size, 32, 2) -> (14, batch_size, 32, 2) 
        x = torch.permute(x, (1, 3, 0, 2)) # (14, batch_size, 32, 2) -> (batch_size, 2, 14, 32). 
        x = F.relu(self.conv1(x)) #(batch_size, 2, 14, 32) -> (batch_size, 16, 14, 32)
        x = self.dropout1(x)#(batch_size, 16, 14, 32)
        
        #skip connection
        x_identity_1 = torch.permute(x_identity_1, (0, 3, 2, 1))#(batch_size, 32, 14, 2) -> (batch_size, 2, 14, 32)
        x_identity_1 = self.adjust1(x_identity_1)# (batch_size, 64, 14, 32)
        x = x_identity_1+x
        
        x = torch.permute(x, (0, 3, 1, 2))#(batch_size, 64, 14, 32) -> #(batch_size, 32, 14, 64)
        x = torch.flatten(x, start_dim=2, end_dim=-1)
        
        #GRU
        _, hid = self.gru(x)#reduce feature dimension to #(batch_size, 32, flatten features)
        hid = torch.squeeze(hid)
        out = self.fc(hid)
        
        return out
    
def train(model, train_loader, optimizer, criterion, norm_adj, device, seed):
    model.train()
    train_losses = 0
    train_corrects = 0
    counter = 0
    sample_counter = 0
    torch.manual_seed(seed)
    train_probs = []
    train_preds = []
    train_gts = []
    
    for batch, (data, target) in enumerate(train_loader, 1):
        counter+=1
        sample_counter+=len(data)
        data, target, norm_adj = data.to(device), target.to(device), norm_adj.to(device)
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data, norm_adj)
        # calculate the loss
        loss = criterion(output, target)
        train_losses+=loss.item()
        # calculate the acc
        probs = torch.sigmoid(output.data)
        pred = torch.round(probs)
        train_corrects += (pred == target).sum().item()
        train_probs.append(probs.cpu().detach().numpy())
        train_preds.append(pred.cpu().detach().numpy())
        train_gts.append(target.cpu().detach().numpy())
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
      
    train_preds, train_gts, train_probs = np.concatenate(train_preds, axis=0), np.concatenate(train_gts, axis=0), np.concatenate(train_probs, axis=0)
    _,re,_,_ = get_precision_recall_F1(preds=train_preds, targets=train_gts, probs=train_probs)    
    # loss and accuracy for the complete epoch
    epoch_loss = train_losses/ counter
    epoch_acc = train_corrects/ sample_counter

    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, norm_adj, device, seed):
    model.eval()
    val_losses = 0
    val_corrects = 0
    counter = 0
    sample_counter =0 
    torch.manual_seed(seed)
    
    val_probs = []
    val_preds = []
    val_gts = []
    with torch.no_grad():
        for data, target in val_loader:
            counter += 1
            sample_counter += len(data)
            data, target, norm_adj = data.to(device), target.to(device), norm_adj.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data, norm_adj)
            # calculate the loss
            loss = criterion(output, target)
            val_losses+=loss.item()  
            # calculate the accuracy
            probs = torch.sigmoid(output.data)
            pred = torch.round(probs)
            val_corrects += (pred == target).sum().item()
            val_probs.append(probs.cpu().detach().numpy())
            val_preds.append(pred.cpu().detach().numpy())
            val_gts.append(target.cpu().detach().numpy())
    
    
    val_preds, val_gts, val_probs = np.concatenate(val_preds, axis=0), np.concatenate(val_gts, axis=0), np.concatenate(val_probs, axis=0)
    _,re,f1,_ = get_precision_recall_F1(preds=val_preds, targets=val_gts, probs=val_probs)

    # loss and accuracy for the complete epoch
    epoch_loss = val_losses/ counter
    epoch_acc = val_corrects/ sample_counter
#     epoch_acc = 100. * (val_corrects / len(val_loader.dataset))
    

    return epoch_loss, epoch_acc, float(f1)


def train_pedgraph(train_loader, val_loader, save_model_folder_acc, save_model_folder_f1, loss_name, alpha=None):
    epochs = 200
    norm_adj = get_norm_adj()
    model = PedestrianGraph()
    model_name = 'graph'

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model.to(device) 
    
    if not os.path.exists(save_model_folder_acc):
        os.mkdir(save_model_folder_acc)
        
    if not os.path.exists(save_model_folder_f1):
        os.mkdir(save_model_folder_f1)
        
    save_best_model = MS.SaveBestModel()
    save_best_model_acc = MS.SaveBestModel_by_accuracy()
    save_best_model_f1 = MS.SaveBestModel_by_f1()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-8)
    
    if loss_name == "BCE":
        if alpha!=None:
            pos_weight = torch.tensor([1-alpha]).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()
    elif loss_name == "focal":
        if alpha!=None:
            criterion = WeightedFocalLoss(alpha)
        else:
            criterion = WeightedFocalLoss()
    else:
        print("Not support such loss!")
        raise SystemExit()

    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []

    seed = 2676
    print('Dataloader seed: {0}'.format(seed))
    
    for epoch in range(1, epochs + 1):
        scheduler_counter = epoch%5000  
        if scheduler_counter==0:
            for i in range(len(optimizer.param_groups)):  
                optimizer.param_groups[i]['weight_decay'] *= 0.98  

        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_acc = train(model, train_loader, 
                                                optimizer, criterion, norm_adj, device, seed)
        valid_epoch_loss, valid_epoch_acc, f1 = validate(model, val_loader,  
                                                    criterion,norm_adj, device, seed)
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        # save the best model till now if we have the least loss in the current epoch
#         save_best_model(
#             valid_epoch_loss, epoch, model, model_name, optimizer, criterion, save_model_folder
#         )
#         print('-'*50)
        
        save_best_model_acc(valid_epoch_acc, epoch, model, model_name, optimizer, criterion, save_model_folder_acc)
        save_best_model_f1(f1, epoch, model, model_name, optimizer, criterion, save_model_folder_f1)

    # save the trained model weights for a final time
#     MS.save_model(epochs, model, model_name, optimizer, criterion, save_model_folder)
    # save the loss and accuracy plots
    MS.save_plots(model_name, train_acc, valid_acc, train_loss, valid_loss, save_model_folder_acc)
    MS.save_plots(model_name, train_acc, valid_acc, train_loss, valid_loss, save_model_folder_f1)
    print('TRAINING COMPLETE')     