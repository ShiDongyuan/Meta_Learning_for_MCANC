import torch
import os
import numpy as np 
import matplotlib.pyplot as plt

from Buiding_custom_dataset          import Noise_Dataset
from torch.utils.data                import DataLoader
from Reptile_Meta_learning_for_MCANC import Reptile_Meta, bulidng_data_by_DelayLine, LossFunction_reptile
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy

def save_training_mode_to_workspace(path_root, pth_file, state_dict):
    current_dictory = os.getcwd()
    pth_file_folder = os.path.join(current_dictory, path_root)

    if os.path.exists(pth_file_folder) == False:
        os.mkdir(pth_file_folder)
    
    model_file = os.path.join(pth_file_folder,pth_file)
    
    torch.save(state_dict,model_file)
    print(f'Infor : the model have save to {model_file}')

def Train_Reptile_for_MCANC(Epoch, Model, criterion, optimizer, Data_loder, Wc_ini, Lr, device='cpu', writer=None):
    ERR = []
    for i in range(Epoch):
        Err_iteration = Train_Reptile_one_iteration(Model, criterion, optimizer, Data_loder, Wc_ini, Lr, device, writer=writer, epoch=i)
        ERR.append(Err_iteration)
    
    return ERR

def Train_Reptile_one_iteration(Model, criterion, optimizer, Data_loder, Wc_ini, Lr, device, writer=None, epoch=None):

    Len_c = 512 
    
    t = 0 
    loss_vector = []
    for Fxs, Diss in Data_loder:
        
        # Loading the data from the data loder. 
        Fx        = Fxs[0]
        Dis       = Diss[0]
        Fx_extend = bulidng_data_by_DelayLine(Len_c=512, Fx=Fx)
        Fx        = Fx.to(device)
        Dis       = Dis.to(device)
        Fx_extend = Fx_extend.to(device)
        
        # Loading the state dictonary 
        state_dict = Model.state_dict()
        state_dict['Control_filter'] = Wc_ini
        Model.load_state_dict(state_dict)
        
        # Walking in Len_c iterations.
        loss_sum = 0
        for i in range(Len_c):
            
            # foward progress 
            anti_noise, Gamme_vector = Model(Fx_extend[i])
            # print(f'Debug 1: input referen is {Fx_extend[i]} and size is {Fx_extend[i].shape}')
            
            # computing the loss function 
            loss       = criterion(anti_noise, Dis[:,i], Gamme_vector[i])
            # print(f'Debug 2: the disturbance is {Dis[:,i]}')
            # print(f'Debug 3: the Gamma factor is {Gamme_vector[i]}')
            loss_sum   += loss.cpu().item()
            # print(f'Debug 4: the loss function is {loss.cpu().item()}')
            
            # set the gradient to zeros 
            optimizer.zero_grad()
            
            # backward progress
            loss.backward()
            
            # updating the modle 
            optimizer.step()
            
        loss_vector.append(loss_sum)
        print(f'The loss of {t}-iteration is : {loss_sum}.')
        writer.add_scalar('training loss',loss_sum,
                            epoch * len(Data_loder) + t)
        t += 1
        
        state_dict = deepcopy(Model.state_dict())
        Wc_ini     = Wc_ini + Lr*(state_dict['Control_filter'] - Wc_ini)
        # print(f'Debug 5: Wc_ini is {Wc_ini}')
    return loss_vector

if __name__ == "__main__":
    
    root_dir = 'noise_dataset'
    csv_file = 'index.csv'
    
    nois_dataset = Noise_Dataset(csv_file, root_dir)
    data_loader  = DataLoader(dataset=nois_dataset, batch_size=1,shuffle=True)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'inf 1: Using the {device} for the model trainning!')

    Meta_model   = Reptile_Meta(num_ref=4, num_sec=4, Len_c=512, Gamma=0.9, device=device).to(device)

    criterion = LossFunction_reptile
    optimizer = torch.optim.SGD(Meta_model.parameters(), lr=1e-7)

    writer = SummaryWriter('fashion_mnist_experiment_1')
    
    Wc_ini = torch.zeros((4, 4, 512), dtype=torch.float32)

    ERR = Train_Reptile_for_MCANC(Epoch=1, Model=Meta_model, criterion=criterion, optimizer=optimizer, Data_loder= data_loader, Wc_ini = Wc_ini, Lr=1e-8, device=device, writer=writer)
    error_vector = np.array(ERR).reshape(1,-1)
    plt.plot(error_vector.squeeze())
    plt.ylabel('some numbers')
    plt.grid()
    plt.show()
    
    path_root = 'pth_modle'
    pth_file  = 'Reptile_v1.pth'
    save_training_mode_to_workspace(path_root, pth_file, Meta_model.state_dict())