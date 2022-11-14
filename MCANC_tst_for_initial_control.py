import numpy             as np 
import scipy.signal      as signal 
import matplotlib.pyplot as plt 
import os
import torch

from scipy.io import loadmat
from Multichannel_FxLMS_with_initialization_algorithm import McFxLMS_algorithm, train_fxmclms_algorithm

def load_noise_data_from_workspace(file_root, file_name):
    current_dictory = os.getcwd()
    noise_file = os.path.join(current_dictory, file_root, file_name)
    noise      = loadmat(noise_file)
    
    return noise['Primary_noise'], noise['Disturbances']

def load_path_from_workspace(Folder, file_name):
    current_dictory = os.getcwd()
    mat_file_name   = os.path.join(current_dictory, Folder, file_name) 
    mat_path_files  = loadmat(mat_file_name)
    
    return mat_path_files['primary_path_matrix'], mat_path_files['secondary_path_matrix']

def load_intitial_control_filter(path_root, pth_file):
    model_file = os.path.join(path_root, pth_file)
    model_dict = torch.load(model_file)
    Wc         = model_dict['initial_weigths']
    
    return Wc 

def plot_error_disturbance(disturbacne, error, error_v1, Wc_ini, Wo):
    fig, axs = plt.subplots(3)
    axs[0].plot(range(len(disturbacne)),disturbacne,label='ANC off') 
    axs[0].plot(range(len(error_v1)),error_v1, label='ANC on')
    axs[0].grid()
    axs[0].set_title('With the zero intialization')
    axs[0].legend()
    axs[1].plot(range(len(disturbacne)),disturbacne,range(len(error)),error)
    axs[1].grid()
    axs[1].set_title('With the Meta intialization')
    
    w, h  = signal.freqz(Wc_ini, 1, fs=16000)
    _, h1 = signal.freqz(Wo, 1, fs=16000)
    axs[2].plot(w, 20*np.log10(np.abs(h)), label = 'Metal Intial control filter' )
    axs[2].plot(w, 20*np.log10(np.abs(h1)), label = 'Optimal control filter' )
    axs[2].set_title('Digital filter frequency response')
    axs[2].set_ylabel('Amplitude Response [dB]')
    axs[2].set_xlabel('Frequency (Hz)')
    axs[2].legend()
    axs[2].grid()
    plt.show()

if __name__=="__main__":
    
    # Loading the primary nosie and disturbance 
    Folder     = 'raw_noise_data_tst'
    noise_file = 'board_noise_980-2500.mat'
    Pri, Dis   = load_noise_data_from_workspace(file_root=Folder, file_name=noise_file)
    Pri_tensor = torch.tensor(Pri, dtype=torch.float)
    Dis_tensor = torch.tensor(Dis, dtype=torch.float)
    Ref        = Pri_tensor.repeat(4,1)
    
    # print(Ref.shape)
    # plt.plot(range(len(Dis[0,:])),Dis[0,:])
    
    # Loading the primary and the secondary paths
    primary_paths, secondary_paths = load_path_from_workspace(Folder='Path_data', file_name='path_matrix.mat')
    secon                          = torch.tensor(secondary_paths, dtype=torch.float)
    
    # Loading the initialization of the control filter 
    path_root = 'pth_modle'
    pth_file  = 'MAML_v1.pth'
    Wc_ini    = load_intitial_control_filter(path_root=path_root, pth_file=pth_file)
    
    # Creating the instance of McFxLMS
    McFxLMS_model = McFxLMS_algorithm(Wc_initialization=Wc_ini, Sec=secon, device='cpu')
    
    erro_vctor   = train_fxmclms_algorithm(Model=McFxLMS_model, Ref=Ref, Disturbance=Dis_tensor, device='cpu', Stepsize = 0.0000005, so = None)
    error_signal = np.array(erro_vctor)
    
    # Creating the instance of McFxLMS with zero intialization 
    McFxLMS_algorithm_v1 = McFxLMS_algorithm(Wc_initialization=torch.zeros_like(Wc_ini), Sec=secon, device='cpu')
    
    erro_vctor      = train_fxmclms_algorithm(McFxLMS_algorithm_v1, Ref=Ref, Disturbance=Dis_tensor, device='cpu', Stepsize = 0.0000005, so = None)
    error_signal_v1 = np.array(erro_vctor)
    
    Wo = McFxLMS_algorithm_v1._get_coeff_()
    
    plot_error_disturbance(Dis[0,:], error_signal[:,0], error_signal_v1[:,0], Wc_ini[0,0,:], Wo[0,0,:])


    