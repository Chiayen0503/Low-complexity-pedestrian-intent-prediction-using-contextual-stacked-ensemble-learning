import torch
import matplotlib.pyplot as plt


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(self, best_valid_loss=float('inf')):
        self.best_valid_loss = best_valid_loss
        
    def __call__(self, current_valid_loss, epoch, model, model_name, optimizer, criterion, save_folder=None):
        if save_folder:
            f_name = save_folder+'best_{0}_model.pth'.format(model_name)
        else:
            f_name = 'best_{0}_model.pth'.format(model_name)
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, f_name)

            
            
class SaveBestModel_by_accuracy:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(self, best_valid_acc=-float('inf')):
        self.best_valid_acc = best_valid_acc
        
    def __call__(self, current_valid_acc, epoch, model, model_name, optimizer, criterion, save_folder=None):
        if save_folder:
            f_name = save_folder+'best_{0}_model_byacc.pth'.format(model_name)
        else:
            f_name = 'best_{0}_model_byacc.pth'.format(model_name)
        if current_valid_acc > self.best_valid_acc:
            self.best_valid_acc = current_valid_acc
            print(f"\nBest validation acc: {self.best_valid_acc}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, f_name)   
            
class SaveBestModel_by_f1:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(self, best_valid_f1=-float('inf')):
        self.best_valid_f1 = best_valid_f1
        
    def __call__(self, current_valid_f1, epoch, model, model_name, optimizer, criterion, save_folder=None):
        if save_folder:
            f_name = save_folder+'best_{0}_model_byf1.pth'.format(model_name)
        else:
            f_name = 'best_{0}_model_byf1.pth'.format(model_name)
        if current_valid_f1 > self.best_valid_f1:
            self.best_valid_f1 = current_valid_f1
            print(f"\nBest validation f1: {self.best_valid_f1}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, f_name)  
            
def save_model(epochs, model, model_name, optimizer, criterion, save_folder=None):
    """
    Function to save the trained model to disk.
    """
    print(f"Saving final model...")
    if save_folder:
        f_name = save_folder+'final_{0}_model.pth'.format(model_name)
    else:
        f_name = 'final_{0}_model.pth'.format(model_name)
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, f_name)
    
def save_plots(model_name, train_acc, valid_acc, train_loss, valid_loss, save_folder=None):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    if save_folder:
        f1 = save_folder+'{0}_accuracy.png'.format(model_name)
        f2 = save_folder+'{0}_loss.png'.format(model_name)
    else:
        f1 = '{0}_accuracy.png'.format(model_name)
        f2 = '{0}_loss.png'.format(model_name)
     
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f1)
    
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f2)