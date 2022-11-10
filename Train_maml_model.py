from torch.utils.data         import DataLoader
from Modified_MAMAL_for_MCANC import Modified_MAML, Loss_Function_maml, bulidng_data_by_DelayLine
from Buiding_custom_dataset   import Noise_Dataset

import torch

def Train_Modified_MAML_for_MCANC(Epoch, Model, criterion, optimizer, Data_loder):
    for i in range(Epoch):
        Train_Modified_MAML_one_iteration(Model, criterion, optimizer, Data_loder)
    
def Train_Modified_MAML_one_iteration(Model, criterion, optimizer, Data_loder):
    
    t = 0 
    for Fxs, Diss in Data_loder:
        Fx  = Fxs[0]
        Dis = Diss[0]
        
        Fx_extend                       = bulidng_data_by_DelayLine(Len_c=512, Fx=Fx)
        anti_noise_matrix, Gamma_vector = Model(Fx, Dis, Fx_extend)
        
        loss = criterion(anti_noise_matrix, Dis, Gamma_vector)
        
        if t % 100 == 99:
            print(f'The {t}th iteration: loss is {loss.item()}')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t +=1

if __name__ == "__main__":
    
    root_dir = 'noise_dataset'
    csv_file = 'index.csv'
    
    nois_dataset = Noise_Dataset(csv_file, root_dir)
    data_loader  = DataLoader(dataset=nois_dataset, batch_size=1,shuffle=True)
    
    Meta_model   = Modified_MAML(num_ref=4, num_sec=4, Len_c=512, L_r=1e-5, Gamma=0.99)
    
    criterion = Loss_Function_maml
    optimizer = torch.optim.Adam(Meta_model.parameters(), lr=1e-5)
    
    Train_Modified_MAML_for_MCANC(Epoch=10, Model=Meta_model, criterion=criterion, optimizer=optimizer, Data_loder= data_loader)
        
        