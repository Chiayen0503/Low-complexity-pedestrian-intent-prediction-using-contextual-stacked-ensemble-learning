import torch
from models.pedgraph_model_3 import *
from models.onedcnn_model_16frames import *
from models.stacked_gru_16frames import *
from models.ensemble_ablation import *
from models.loss import *


def predict_graph(models_folder, x, device, best_by):
    models_root = 'training_results/'+models_folder
    if not best_by:
        model_path = models_root+'best_graph_model.pth'
    else:
        model_path = models_root+'best_graph_model_{0}.pth'.format('by'+best_by)
    x = torch.FloatTensor(x).to(device)
    norm_adj = get_norm_adj().to(device)
    model = PedestrianGraph().to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    y_hat = model(x, norm_adj)
    return y_hat


def predict_1dcnn_addfeatures(model_folder, x, device,  best_by):
    
    models_root = 'training_results/'+model_folder
    if not best_by:
        model_path = models_root+'best_1dcnn_model.pth'
    else:
        model_path = models_root+'best_1dcnn_model_{0}.pth'.format('by'+best_by)
    x = torch.FloatTensor(x).to(device)
    model = CNN_ForecastNet().to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    y_hat = model(x)
    return y_hat


def predict_gru(x1_data_sizes, x2_data_size, models_folder, x1, x2, device, best_by):
    models_root = 'training_results/'+models_folder
    if not best_by:
        model_path = models_root+'best_gru_model.pth'
    else:
        model_path = models_root+'best_gru_model_{0}.pth'.format('by'+best_by)
    
    x1 = torch.FloatTensor(x1).to(device)
    x2 = torch.FloatTensor(x2).to(device)
    print(x1.size(), x2.size())
    x1 = split_categorical_data(x1, x1_data_sizes)
    a = torch.cat(x1)
    b = torch.tensor(x2)
    print(a.size(), b.size())
    model = SFGRU(x1_data_sizes, x2_data_size).to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    y_hat = model(x1, x2)
    return y_hat


def predict_ensemble_ablation(models_folder, x, device, best_by):
    model_root = 'training_results/'+models_folder
    if not best_by:
        model_path = model_root+'best_ablation_model.pth'
    else:
        model_path = model_root+'best_ablation_model_{0}.pth'.format('by'+best_by)
        
    model = Perceptron_ablation().to(device)
    x = torch.FloatTensor(x).to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    y_hat = model(x)
    return y_hat 


