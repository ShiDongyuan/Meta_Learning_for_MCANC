#      __  __  ____ __  __  ____    ____                _ _             _     __  __       _
#     |  \/  |/ ___|  \/  |/ ___|  / ___|_ __  ___   __| (_) __ _ _ __ | |_  |  \/  | ___ | |_  __ _
#     | |\/| | |   | |\/| | |     | |  _| '__|/ _ \ / _` | |/ _` | '_ \| __| | |\/| |/ _ \| __|/ _` |
#     | |  | | |___| |  | | |___  | |_| | |  |  __/| (_| | | (_| | | | | |_  | |  | |  __/| |_| (_| |
#     |_|  |_|\____|_|  |_|\____|  \____|_|   \___| \__,_|_|\__,_|_| |_|\__| |_|  |_|\___| \__|\__,_|

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

from adaptive_algorithm_v1 import Adaptive_n_iteration_algorithm, frequency_response_depict

#----------------------------------------------------------------
# Function: Reptile algorithm is running with n iterations.
#----------------------------------------------------------------

def Reptile_train_progress(Model, Filtered_ref, Disturbance, n_iteration, Len_c, device, Stepsize):
    
    # Building the optimizer 
    optimizer    = optim.SGD([Model.Wc], lr=Stepsize)
    
    Erro_signal  = []                      # the recorded error signal 
    len_data     = Len_c                   # the length of each data batch
    Filtered_ref = Filtered_ref.to(device) # the filtered reference signals 
    Disturbance  = Disturbance.to(device)  # the disturbance signals 
    
    for itera in range(n_iteration):
        # Feedforward progress 
        xin        = Filtered_ref[:,:,:,itera:itera+len_data]
        dis        = Disturbance[:,itera+len_data-1]

        anti_noise = Model.forward(xin)
        loss,err   = Model.Loss_function(anti_noise,dis)
        
        # Recording the squared error data 
        Erro_signal.append(loss.cpu().detach().numpy())
        
        # Backward progress 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return Erro_signal
#----------------------------------------------------------------
# Function : main 
#----------------------------------------------------------------
if __name__=="__main__":
    
    # Setting up the noise dataset  
    noies_dataset = Noise_Dataset(csv_file='index.csv', root_dir='noise_dataset_1024')
    data_loader   = DataLoader(dataset=noies_dataset, batch_size=1, shuffle=True)
    
    # Determining the processors 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'inf 1: Using the {device} for the model trainning!')
    
    # Building the adaptive algorithm 
    Len_c              = 512 
    Wc_ini             = torch.zeros((4, 4, Len_c), dtype=torch.float32)
    
    niteration         = 25 
    alpha              = 0.0000003
    
    # Filtering 
    Erro_vector = []
    t           = 0
    for Fxs, Diss in data_loader:
        
        # Loading the data from the data loder. 
        Fxs  = Fxs[0]
        Diss = Diss[0]
        
        Adatpvie_algorithm = Adaptive_n_iteration_algorithm(Wc_initialization=Wc_ini, e_num=4, device=device)
        
        Erro_signal = Reptile_train_progress(Model=Adatpvie_algorithm, Filtered_ref=Fxs, Disturbance=Diss, n_iteration= niteration, Len_c= Len_c, device=device, Stepsize=0.0000001)
        
        err_sum = np.array(Erro_signal).sum()
        print(f' The {t}-th interation loss is : {err_sum}')
        
        Erro_vector.append(err_sum)
        Wo     = Adatpvie_algorithm._get_coeff_()
        # Wc_ini = Wc_ini + alpha*(Wo-Wc_ini)
        Wc_ini = Wo 
        t += 1
        
    plt.plot(Erro_vector, label ='Loss values')
    plt.grid()
    plt.legend()
    plt.show()
    
    frequency_response_depict(Wc_ini[0,0,:].detach().numpy())
    
    data_dic = {'Control filter': Wc_ini.detach().numpy()}
    mat_file = os.path.join('pth_modle','reptile_control_v3.mat')
    savemat(mat_file, data_dic) 
        
