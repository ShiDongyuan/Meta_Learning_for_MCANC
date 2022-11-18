import torch
import os
import numpy as np 
import torch.nn as nn 
import torch.optim as optim
import scipy.signal as signal 
import matplotlib.pyplot as plt

from Buiding_custom_dataset             import Noise_Dataset
from torch.utils.data                   import DataLoader
from copy                               import deepcopy
from scipy.io import loadmat, savemat

from adpative_algorithm import Adaptive_n_iteration_algorithm, train_adaptive_algorithm

if __name__=="__main__":
    
    # Setting up the noise dataset  
    noies_dataset = Noise_Dataset(csv_file='index.csv', root_dir='noise_dataset')
    data_loader   = DataLoader(dataset=noies_dataset, batch_size=1, shuffle=True)
    
    # Determining the processors 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'inf 1: Using the {device} for the model trainning!')
    
    # Building the adaptive algorithm 
    Wc_ini             = torch.zeros((4, 4, 512), dtype=torch.float32)
    
    lr = 0.000001 
    
    # Filtering 
    Erro_record = []
    t = 0 
    for Fxs, Diss in data_loader:
        
        # Building adatpive algorithm 
        Adatpvie_algorithm = Adaptive_n_iteration_algorithm(Wc_initialization=Wc_ini, e_num=4, device=device)
        
        # Loading the data from the data loder. 
        Fxs  = Fxs.to(device)
        Diss = Diss.to(device)
        
        # Trainning the adaptive algorithm 
        Erro_signal = train_adaptive_algorithm(Model=Adatpvie_algorithm, Filtered_ref=Fxs[0], Disturbance=Diss[0], device=device, Stepsize=0.000001)
        Erro_sum    = np.array(Erro_signal).sum()
        print(f' The {t}-iteration loss error is : {Erro_sum}')
        Erro_record.append(Erro_sum)
        
        # Updating the intial control filter 
        Wo     = Adatpvie_algorithm._get_coeff_()
        # Wc_ini = Wc_ini + lr*(Wo-Wc_ini)
        Wc_ini = Wo
        t      += 1
    
    plt.plot(np.array(Erro_record))
    plt.grid()
    plt.show()
    
    data_dic = {'Control filter': Wo.detach().numpy()}
    mat_file = os.path.join('pth_modle','reptile_control.mat')
    savemat(mat_file, data_dic) 
        