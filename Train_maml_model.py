from Modified_MAMAL_for_MCANC import Modified_MAML, Loss_Function_maml, bulidng_data_by_DelayLine
from Buiding_custom_dataset   import Noise_Dataset
from torch.utils.data         import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torch
import os
import numpy as np 
import matplotlib.pyplot as plt

def Train_Modified_MAML_for_MCANC(Epoch, Model, criterion, optimizer, Data_loder, device='cpu', writer=None):
    ERR = []
    for i in range(Epoch):
        Err_iteration =Train_Modified_MAML_one_iteration(Model, criterion, optimizer, Data_loder, device, writer=writer, epoch=i)
        ERR.append(Err_iteration)
    
    return ERR
    
def Train_Modified_MAML_one_iteration(Model, criterion, optimizer, Data_loder, device, writer=None, epoch=None):
    
    t = 0 
    Err = []
    for Fxs, Diss in Data_loder:
        Fx  = Fxs[0]
        Dis = Diss[0]
        
        Fx_extend                       = bulidng_data_by_DelayLine(Len_c=512, Fx=Fx)
        
        Fx = Fx.to(device)
        Dis = Dis.to(device)
        Fx_extend = Fx_extend.to(device)
        
        anti_noise_matrix, Gamma_vector = Model(Fx, Dis, Fx_extend)
        
        loss = criterion(anti_noise_matrix, Dis, Gamma_vector)
        
        if t % 100 == 99:
            error = loss.cpu().item()
            print(f'The {t}th iteration: loss is {error}')
            Err.append(error)
            
            writer.add_scalar('training loss',error,
                            epoch * len(Data_loder) + t)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t +=1
    
    return Err

def save_training_mode_to_workspace(path_root, pth_file, state_dict):
    current_dictory = os.getcwd()
    pth_file_folder = os.path.join(current_dictory, path_root)

    if os.path.exists(pth_file_folder) == False:
        os.mkdir(pth_file_folder)
    
    model_file = os.path.join(pth_file_folder,pth_file)
    
    torch.save(state_dict,model_file)
    print(f'Infor : the model have save to {model_file}')

if __name__ == "__main__":
    
    root_dir = 'noise_dataset'
    csv_file = 'index.csv'
    
    nois_dataset = Noise_Dataset(csv_file, root_dir)
    data_loader  = DataLoader(dataset=nois_dataset, batch_size=1,shuffle=True)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'inf 1: Using the {device} for the model trainning!')
    
    
    Meta_model   = Modified_MAML(num_ref=4, num_sec=4, Len_c=512, L_r=3e-7, Gamma=0.99, device=device).to(device)
    
    criterion = Loss_Function_maml
    optimizer = torch.optim.SGD(Meta_model.parameters(), lr=3e-7)
    
    writer = SummaryWriter('fashion_mnist_experiment_1')
    
    ERR = Train_Modified_MAML_for_MCANC(Epoch=3, Model=Meta_model, criterion=criterion, optimizer=optimizer, Data_loder= data_loader, device=device, writer=writer)
    error_vector = np.array(ERR).reshape(1,-1)
    plt.plot(error_vector.squeeze())
    plt.ylabel('some numbers')
    plt.grid()
    plt.show()
    
    path_root = 'pth_modle'
    pth_file  = 'MAML_v1.pth'
    save_training_mode_to_workspace(path_root, pth_file, Meta_model.cpu().state_dict)

        
        