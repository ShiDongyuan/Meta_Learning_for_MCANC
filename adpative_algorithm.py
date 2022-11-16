import torch
import numpy as np 
import torch.nn as nn 
import torch.optim as optim
import scipy.signal as signal 
import matplotlib.pyplot as plt

from Buiding_custom_dataset             import Noise_Dataset
from torch.utils.data                   import DataLoader
from copy                               import deepcopy

#---------------------------------------------------------------
# Class Adaptive_n_iteration_algorithm
#---------------------------------------------------------------
class Adaptive_n_iteration_algorithm():
    
    def __init__(self, Wc_initialization, e_num, device):
        self.r_num = Wc_initialization.shape[0] # The number of the reference sensors.
        self.s_num = Wc_initialization.shape[1] # The number of the secondary sources.
        self.len   = Wc_initialization.shape[2] # The length of the control filter.
        self.e_num = e_num                      # The number of the error sensors. 
        
        self.Wc    = torch.tensor(Wc_initialization.detach(), requires_grad=True, dtype=torch.float, device=device)
        self.Xd    = torch.zeros(self.r_num, self.s_num, self.e_num, self.len, dtype=torch.float, device=device)
    
    # forward progress 
    def forward(self, Xin):
        
        # Shifting the delay line and taking the new input data 
        self.Xd             = torch.roll(self.Xd, 1, dims=3) 
        self.Xd[:,:,:,0]    = Xin
        
        # Computing the anti noise singal matrix.
        anti_noise_elements = torch.einsum('...rsen,...rsn ->...rse', self.Xd, self.Wc)
        anti_noise          = torch.einsum('...rse->...e', anti_noise_elements)
        
        return anti_noise

    # loss function 
    def Loss_function(self, anti_noise, Dis):
        """ The loss function for the adaptive alogrithm (mean square error)

        Args:
            anti_noise (_float_): The anti-noise signals [E_num]
            Dis (_float_): The disturbance signal [E_num]

        Returns:
            _type_: The value of the loss function
        """
        # noise suppression 
        Error_signal = Dis - anti_noise 
        # geting the square of the error signal
        loss         = torch.einsum('e,e->', Error_signal, Error_signal)
        
        return loss, Error_signal 
    
    # exctrat the coefficients from the model 
    def _get_coeff_(self):
        return self.Wc

#---------------------------------------------------------------
# Function: training the adaptive algorithm 
#---------------------------------------------------------------
def train_adaptive_algorithm(Model, Filtered_ref, Disturbance, device, Stepsize):
    
    # Building the optimizer 
    optimizer    = optim.SGD([Model.Wc], lr=Stepsize)
    
    Erro_signal  = []                      # the recorded error signal 
    len_data     = Disturbance.shape[1]    # the length of each data batch
    Filtered_ref = Filtered_ref.to(device) # the filtered reference signals 
    Disturbance  = Disturbance.to(device)  # the disturbance signals 
    
    for itera in range(len_data):
        # Feedforward progress 
        xin        = Filtered_ref[:,:,:,itera]
        dis        = Disturbance[:,itera]

        anti_noise = Model.forward(xin)
        loss,err   = Model.Loss_function(anti_noise,dis)
        
        # Recording the squared error data 
        Erro_signal.append(loss.cpu().detach().numpy())
        
        # Backward progress 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return Erro_signal

if __name__=="__main__":
    
    # Setting up the noise dataset  
    noies_dataset = Noise_Dataset(csv_file='index.csv', root_dir='noise_dataset_1024')
    data_loader   = DataLoader(dataset=noies_dataset, batch_size=1, shuffle=True)
    
    # Determining the processors 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'inf 1: Using the {device} for the model trainning!')
    
    # Building the adaptive algorithm 
    Wc_ini             = torch.zeros((4, 4, 512), dtype=torch.float32)
    Adatpvie_algorithm = Adaptive_n_iteration_algorithm(Wc_initialization=Wc_ini, e_num=4, device=device)
    # Adatpvie_algorithm.initialize_control_filter(torch.ones((4, 4, 512), dtype=torch.float32))
    
    # Filtering 
    for Fxs, Diss in data_loader:
        
        # Loading the data from the data loder. 
        Fxs  = Fxs.to(device)
        Diss = Diss.to(device)
        
        # Trainning the adaptive algorithm 
        Erro_signal = train_adaptive_algorithm(Model=Adatpvie_algorithm, Filtered_ref=Fxs[0], Disturbance=Diss[0], device=device, Stepsize=0.00003)
        
        plt.plot(Diss[0].cpu()[0,:], label='Disturbance')
        plt.plot(np.array(Erro_signal)[:,0], label ='Residual error')
        plt.grid()
        plt.legend()
        plt.show()
        
        Wc = Adatpvie_algorithm._get_coeff_().cpu()
        print(Wc.shape)
        plt.plot(Wc[0,0,:].detach().numpy())
        plt.grid()
        plt.show()