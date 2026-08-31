import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import utils.model_saver as MS
from models.loss import WeightedFocalLoss
import random
from model_evaluation.performance_scores import get_precision_recall_F1
import sys
import os 


class SFGRU(nn.Module):
    """
    An encoder-decoder model for pedestrian trajectory prediction.

    Attributes:
        _num_hidden_units: Number of GRU hidden units.
        _regularizer_value: The value of L2 regularizer for training.
        _regularizer: Training regularizer set as L2.

    Methods:
        stacked_rnn: Generates the network model.
        _gru: A helper function for creating a GRU unit.
    """

    def __init__(self, x1_data_sizes, x2_data_size, num_hidden_units=4):
        super(SFGRU, self).__init__()
        # Network parameters
        self._num_hidden_units = num_hidden_units
        self.grus = nn.ModuleList()  # To store the GRU layers
        self.dense = nn.Linear(num_hidden_units+x2_data_size[-1], 1)
        # self.dense = nn.Linear(num_hidden_units+x2_data_size, 1)
        self.num_layers = len(x1_data_sizes)
        
        for i in range(self.num_layers):
            if i == 0:
                first_data = x1_data_sizes[i][1]
                self.grus.append(nn.GRU(input_size=first_data, hidden_size=num_hidden_units, batch_first=True))
            else:
                self.grus.append(nn.GRU(input_size=num_hidden_units + x1_data_sizes[i][1], hidden_size=num_hidden_units, batch_first=True))
        
        

    def forward(self, x1, x2):
        """
        x should be a list of data
        """
        num_layers = self.num_layers
        for i in range(self.num_layers):            
            if i == 0:
                out, _ = self.grus[i](input=x1[i])
            elif i<num_layers-1:
                cat = torch.cat((x1[i], out), dim=2)
                out, _ = self.grus[i](input=cat)
            else:#last layer
                cat = torch.cat((x1[i], out), dim=2)
                _, hn = self.grus[i](input=cat)                
        
        hn = torch.squeeze(hn)
        cat = torch.cat([hn,x2], dim=1)        
        model_output = self.dense(cat)  # Dense layer applied to the last output in the sequence
        return model_output


def split_categorical_data(x, data_sizes):
    """
    suppose x.shape = (frames, features)
    suppose data_sizes = [(frames, features1), (frames, features2), (.., feature3)....]
    """
    data_li = []
    feature_index = 0
    for s in data_sizes:
        old_index=feature_index
        feature_index = feature_index+s[-1]
        subdata = x[:,:,old_index:feature_index]#batch,frame,feature
        data_li.append(subdata)

    for i, d in enumerate(data_li):
        if d.shape[1:]!=data_sizes[i]:
            print(d.shape, data_sizes[i])
            sys.exit('shape of all data should match data_sizes')
    return data_li

def train(model, x1_data_sizes, train_loader, optimizer, criterion, device, seed):
    model.train()
    train_losses = 0
    train_corrects = 0
    counter = 0
    model.to(device)
    torch.manual_seed(seed)
    train_probs = []
    train_preds = []
    train_gts = []
    for batch, (x1, x2, target) in enumerate(train_loader, 1):
        counter+=1
        
        x1, x2, target= x1.to(device), x2.to(device), target.to(device)
#         print(x1.shape, x2.shape, target.shape)
#         print(x1_data_sizes)
        x1 = split_categorical_data(x1, x1_data_sizes)
        
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        output = model(x1, x2)
        
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
#         torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
      
    train_preds, train_gts, train_probs = np.concatenate(train_preds, axis=0), np.concatenate(train_gts, axis=0), np.concatenate(train_probs, axis=0)
    _,re,_,_ = get_precision_recall_F1(preds=train_preds, targets=train_gts, probs=train_probs)    
    # loss and accuracy for the complete epoch
    epoch_loss = train_losses/ counter
    epoch_acc = float(re)*100

    return epoch_loss, epoch_acc


def validate(model, x1_data_sizes, val_loader, criterion, device, seed):
    model.eval()
    val_losses = 0
    val_corrects = 0
    counter = 0
    torch.manual_seed(seed)
    val_probs = []
    val_preds = []
    val_gts = []
    
    with torch.no_grad():
        for x1, x2, target in val_loader:
            counter += 1
            x1, x2, target= x1.to(device), x2.to(device), target.to(device)
            x1 = split_categorical_data(x1, x1_data_sizes)#split x1 to list based on categories 
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(x1, x2) # 50 * 1
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

    return epoch_loss, epoch_acc, float(f1)

def train_stacked_gru(x1_data_sizes, x2_data_size, train_loader, val_loader, save_model_folder_acc, save_model_folder_f1, loss_name, alpha):
    epochs = 100
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    model = SFGRU(x1_data_sizes, x2_data_size)
    model_name = 'gru'
    model.to(device) 

    if not os.path.exists(save_model_folder_acc):
        os.mkdir(save_model_folder_acc)
        
    if not os.path.exists(save_model_folder_f1):
        os.mkdir(save_model_folder_f1)
    
    save_best_model = MS.SaveBestModel()
    save_best_model_acc = MS.SaveBestModel_by_accuracy()
    save_best_model_f1 = MS.SaveBestModel_by_f1()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0001)
    
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
    seed = 2726
    print('Dataloader seed: {0}'.format(seed))

    for epoch in range(1, epochs + 1):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_acc = train(model, x1_data_sizes, train_loader, 
                                                optimizer, criterion, device, seed)
        valid_epoch_loss, valid_epoch_acc, f1 = validate(model, x1_data_sizes, val_loader,  
                                                    criterion, device, seed)
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
#         save the best model till now if we have the least loss in the current epoch
#         save_best_model(
#             valid_epoch_loss, epoch, model, model_name, optimizer, criterion, save_model_folder
#         )
        print('-'*50)
        save_best_model_acc(valid_epoch_acc, epoch, model, model_name, optimizer, criterion, save_model_folder_acc)
        save_best_model_f1(f1, epoch, model, model_name, optimizer, criterion, save_model_folder_f1)

    # save the trained model weights for a final time
#     MS.save_model(epochs, model, model_name, optimizer, criterion, save_model_folder)
    # save the loss and accuracy plots
    MS.save_plots( model_name, train_acc, valid_acc, train_loss, valid_loss, save_model_folder_acc)
    MS.save_plots( model_name, train_acc, valid_acc, train_loss, valid_loss, save_model_folder_f1)
    print('TRAINING COMPLETE')  
   
