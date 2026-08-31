import os 
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.model_saver as MS
from models.loss import WeightedFocalLoss
import numpy as np
from model_evaluation.performance_scores import get_precision_recall_F1

class Perceptron_ablation(torch.nn.Module):
    def __init__(self):
        super(Perceptron_ablation, self).__init__()
        self.fc = nn.Linear(2, 1)
        
    def forward(self, x):
        output = self.fc(x)
        return output
    


def train(model, train_loader, optimizer, criterion, device, p1, p2):
    model.train()
    train_losses = 0
    train_corrects = 0
    counter = 0
    train_probs = []
    train_preds = []
    train_gts = []
    
    for batch, (x1, x2, x3, target) in enumerate(train_loader, 1):
        counter+=1
        x1, x2, x3, target = x1.to(device), x2.to(device), x3.to(device), target.to(device)
        if p1=='x1':
            if p2=='x2':
                x = torch.cat((x1, x2), 1)
            else:#p2=='x3'
                x = torch.cat((x1, x3), 1)
        elif p1=='x2' and p2=='x3':
            x = torch.cat((x2, x3), 1)
        else:
            sys.exit('only x1_x2, x1_x3 or x2_x3 is acceptable')
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        x = torch.cat((x1, x2), 1)
        output = model(x)
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
    _,re,f1,_ = get_precision_recall_F1(preds=train_preds, targets=train_gts, probs=train_probs)    
    # loss and accuracy for the complete epoch
    epoch_loss = train_losses/ counter
    epoch_acc = float(re)*100
#     epoch_acc = 100. * (train_corrects / len(train_loader.dataset))

    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device, p1, p2):
    model.eval()
    val_losses = 0
    val_corrects = 0
    counter = 0
    val_probs = []
    val_preds = []
    val_gts = []    
    with torch.no_grad():
        for x1, x2, x3, target in val_loader:
            counter += 1
            x1, x2, x3, target = x1.to(device), x2.to(device), x3.to(device), target.to(device)
            if p1=='x1':
                if p2=='x2':
                    x = torch.cat((x1, x2), 1)
                else:#p2=='x3'
                    x = torch.cat((x1, x3), 1)
            elif p1=='x2' and p2=='x3':
                x = torch.cat((x2, x3), 1)
            else:
                sys.exit('only x1_x2, x1_x3 or x2_x3 is acceptable')

            output = model(x) # 50 * 1
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
    epoch_acc = float(re)*100
#     epoch_acc = 100. * (val_corrects / len(val_loader.dataset))

    return epoch_loss, epoch_acc , float(f1)


def train_perceptron_ablations(train_loader, val_loader, save_model_folder_acc, save_model_folder_f1, loss_name, p1, p2):
    epochs = 100
    device = torch.device('cuda')
    model = Perceptron_ablation()
    model_name = 'ablation'
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model.to(device) 

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-8)
    
    if loss_name == "BCE":
        criterion = nn.BCEWithLogitsLoss()
    elif loss_name == "focal":
        criterion = WeightedFocalLoss()
    else:
        print("Not support such loss!")
        raise SystemExit()

    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
       
    if not os.path.exists(save_model_folder_acc):
        os.mkdir(save_model_folder_acc)
    if not os.path.exists(save_model_folder_f1):
        os.mkdir(save_model_folder_f1)
        
    save_best_model_acc = MS.SaveBestModel_by_accuracy()
    save_best_model_f1 = MS.SaveBestModel_by_f1()


    for epoch in range(1, epochs + 1):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_acc = train(model, train_loader, 
                                                optimizer, criterion, device, p1, p2)
        valid_epoch_loss, valid_epoch_acc, f1 = validate(model, val_loader,  
                                                    criterion, device, p1, p2)
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        save_best_model_acc(valid_epoch_acc, epoch, model, model_name, optimizer, criterion, save_model_folder_acc)
        save_best_model_f1(f1, epoch, model, model_name, optimizer, criterion, save_model_folder_f1)

    # save the trained model weights for a final time
#     MS.save_model(epochs, model, model_name, optimizer, criterion, save_model_folder)
    # save the loss and accuracy plots
    MS.save_plots( model_name, train_acc, valid_acc, train_loss, valid_loss, save_model_folder_acc)
    MS.save_plots( model_name, train_acc, valid_acc, train_loss, valid_loss, save_model_folder_f1)
    print('TRAINING COMPLETE')