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

def load_intitial_control_filter_v1(path_root, pth_file):
    model_file = os.path.join(path_root, pth_file)
    model_dict = torch.load(model_file)
    Wc         = model_dict['Control_filter']
    
    return Wc 

def plot_error_disturbance(disturbacne, error, error_v1, error_v2, Wc_ini, Wo, Wr):
    fs = 16000
    fig, axs = plt.subplots(4)
    axs[0].plot(np.arange(len(disturbacne))/fs,disturbacne,label='ANC off') 
    axs[0].plot(np.arange(len(error_v1))/fs,error_v1, label='ANC on')
    axs[0].grid()
    axs[0].set_title('McFxLMS with the zero intialization')
    axs[0].set_ylabel('Error signal')
    axs[0].set_xlabel('Times (seconds)')
    axs[0].legend()
    axs[1].plot(np.arange(len(disturbacne))/fs,disturbacne,np.arange(len(error))/fs,error)
    axs[1].grid()
    axs[1].set_title('McFxLMS with the MAML-Meta intialization')
    axs[1].set_ylabel('Error signal')
    axs[1].set_xlabel('Times (seconds)')

    axs[2].plot(np.arange(len(disturbacne))/fs,disturbacne,np.arange(len(error_v2))/fs,error_v2)
    axs[2].grid()
    axs[2].set_title('McFxLMS with the Monte-Carlo Markov-chain Grediant Meta intialization')
    axs[2].set_ylabel('Error signal')
    axs[2].set_xlabel('Times (seconds)')
    
    w, h  = signal.freqz(Wc_ini, 1, fs=16000)
    _, h1 = signal.freqz(Wo, 1, fs=16000)
    _, h2 = signal.freqz(Wr, 1, fs=16000)
    axs[3].plot(w, 20*np.log10(np.abs(h)),  label = 'MAML-Meta initial control filter')
    axs[3].plot(w, 20*np.log10(np.abs(h1)), label = 'Optimal control filter'     )
    axs[3].plot(w, 20*np.log10(np.abs(h2)), label = 'MCMC-Grediant-Meta initial control filter' )
    axs[3].set_title('Filter frequency response')
    axs[3].set_ylabel('Amplitude Response [dB]')
    axs[3].set_xlabel('Frequency (Hz)')
    axs[3].legend()
    axs[3].grid()
    plt.show()

def Compare_convergences_of_different_initialization(err_v1, label_v1, err_v2, label_v2, fs):
    index = np.arange(len(err_v1))/fs
    plt.plot(index, err_v1, label = label_v1)
    plt.plot(index, err_v2, label = label_v2)
    plt.ylabel('Error signal')
    plt.xlabel('Times (seconds)')
    plt.legend()
    plt.grid()
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

    # Determining the processors 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'inf 1: Using the {device} for the model trainning!')
    
    # Loading the primary and the secondary paths
    primary_paths, secondary_paths = load_path_from_workspace(Folder='Path_data', file_name='path_matrix.mat')
    secon                          = torch.tensor(secondary_paths, dtype=torch.float)
    
    #Exp--1---
    # Loading the initialization of the control filter 
    path_root = 'pth_modle'
    pth_file  = 'MAML_v1.pth'
    Wc_ini    = load_intitial_control_filter(path_root=path_root, pth_file=pth_file)
    
    # Creating the instance of McFxLMS
    McFxLMS_model = McFxLMS_algorithm(Wc_initialization=Wc_ini, Sec=secon, device=device)
    
    erro_vctor   = train_fxmclms_algorithm(Model=McFxLMS_model, Ref=Ref, Disturbance=Dis_tensor, device=device, Stepsize = 0.0000005, so = None)
    error_signal = np.array(erro_vctor)
    
     #Exp--2---
    # Creating the instance of McFxLMS with zero intialization 
    McFxLMS_algorithm_v1 = McFxLMS_algorithm(Wc_initialization=torch.zeros_like(Wc_ini), Sec=secon, device=device)
    
    erro_vctor      = train_fxmclms_algorithm(McFxLMS_algorithm_v1, Ref=Ref, Disturbance=Dis_tensor, device=device, Stepsize = 0.0000005, so = None)
    error_signal_v1 = np.array(erro_vctor)
    
    Wo = McFxLMS_algorithm_v1._get_coeff_()
    
    # #Exp--3---
    # # Loading the intial from MCMC 
    # path_root  = 'pth_modle'
    # mat_file   = 'reptile_control.mat'
    
    # model_file = os.path.join(path_root, mat_file)
    # model_dict = loadmat(model_file)
    # Wc         = model_dict['Control filter']

    # # Creating the instance of McFxLMS with zero intialization 
    # McFxLMS_algorithm_v2 = McFxLMS_algorithm(Wc_initialization=torch.tensor(Wc), Sec=secon, device=device)
    
    # erro_vctor      = train_fxmclms_algorithm(McFxLMS_algorithm_v2, Ref=Ref, Disturbance=Dis_tensor, device=device, Stepsize = 0.0000005, so = None)
    # error_signal_v2 = np.array(erro_vctor)
    
    # Wr = McFxLMS_algorithm_v2._get_coeff_()
    
    #Exp--4---
    path_root = 'pth_modle'
    pth_file  = 'Reptile_v1.pth'
    Wc_ini_4    = load_intitial_control_filter_v1(path_root=path_root, pth_file=pth_file)
    
    # Creating the instance of McFxLMS
    McFxLMS_model_v4 = McFxLMS_algorithm(Wc_initialization=Wc_ini_4, Sec=secon, device=device)
    
    erro_vctor      = train_fxmclms_algorithm(Model=McFxLMS_model_v4, Ref=Ref, Disturbance=Dis_tensor, device=device, Stepsize = 0.0000005, so = None)
    error_signal_v4 = np.array(erro_vctor)

    plot_error_disturbance(Dis[0,:], error_signal[:,0], error_signal_v1[:,0], error_signal_v4[:,0], Wc_ini[0,0,:], Wo[0,0,:], Wc_ini_4[0,0,:])

    Compare_convergences_of_different_initialization(err_v1=error_signal_v1[:,0], label_v1='MAML-Meta initial control filter', err_v2=error_signal_v4[:,0], label_v2='MCMC-Grediant-Meta initial control filter', fs=16000)
    