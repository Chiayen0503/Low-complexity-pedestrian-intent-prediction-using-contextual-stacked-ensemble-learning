import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def load_dict(f):
    f = open(f)
    data = json.load(f)
    return data

def read_data(datafolder):
    fnames = []
    data_list = []
    for f in os.listdir(datafolder):
        f_path = os.path.join(datafolder, f)
        if f.split('.')[-1] == 'npy':
            fnames.append(f)
            data_list.append(np.load(f_path))
        elif f.split('.')[-1] == 'json':
            fnames.append(f)
            f_path = open(f_path)
            data_list.append(json.load(f_path))
        else:
            print('Not load: {0}'.format(f)+'. '+'Currently only support: .npy, .json')
            
    return fnames, data_list


def data_to_variables(datafolder):
    fnames, data_list = read_data(datafolder)
    print(fnames)
    for i in range(len(fnames)):
        f = fnames[i]
        if f == 'x_k.npy':
            x_k=data_list[i]
        elif f == 'x_t.npy':
            x_t=data_list[i]
        elif f == 'x_s.npy':
            x_s=data_list[i]
        elif f == 'y_k.npy':
            y_k=data_list[i]
        elif f == 'y_t.npy':
            y_t=data_list[i]
        elif f == 'y_s.npy':
            y_s=data_list[i]
        elif f.endswith('.json'):
            samplelen_info = data_list[i]
        else:
            print('Not support given fnames: {0}!'.format(f))
    print('Loaded data: {0}'.format(fnames))
    try:
        return x_k, x_t, x_s, y_k, y_t, y_s, samplelen_info
    except:
        return x_k, x_t, y_k, y_t, samplelen_info




class TrajectoryData(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)
    

class SkeletonData(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)

    
class CategoricalData(Dataset):
    
    def __init__(self, x1_data, x2_data, y_data):
        self.x1_data = x1_data
        self.x2_data = x2_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.x1_data[index], self.x2_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.y_data)
    

class leveloneData(Dataset):
    
    def __init__(self, X1_data, X2_data, y_data):
        self.X1_data = X1_data
        self.X2_data = X2_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X1_data[index], self.X2_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.y_data)
    


class leveloneData(Dataset):
    
    def __init__(self, X1_data, X2_data, X3_data, y_data):
        self.X1_data = X1_data
        self.X2_data = X2_data
        self.X3_data = X3_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X1_data[index], self.X2_data[index], self.X3_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.y_data)
    