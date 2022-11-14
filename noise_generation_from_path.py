import numpy             as np 
import scipy.signal      as signal 
import matplotlib.pyplot as plt 
import os

from scipy.io import loadmat, savemat

def load_path_from_workspace(Folder, file_name):
    current_dictory = os.getcwd()
    mat_file_name   = os.path.join(current_dictory, Folder, file_name)
    
    mat_path_files  = loadmat(mat_file_name)
    
    return mat_path_files['primary_path_matrix'], mat_path_files['secondary_path_matrix']

def generate_disturbance_filter_x_noises( primary_noise, T=2, fs=16000):
    """Generating the primary and filtered reference signals 

    Args:
        primary_noise : The primary noise. 
        T (float64)   : The duration of the simulation.
        fs (float64)  : The sampling rate. 
    """
    # Loading the primary and secondary paths
    primary_paths, secondary_paths = load_path_from_workspace(Folder='Path_data', file_name='path_matrix.mat')
    
    # Generating the disturbances
    # primary_noise = np.random.randn(T*fs)
    num_primary   = primary_paths.shape[0]
    disturbances  = np.zeros((num_primary,T*fs))

    for i in range(num_primary):
        disturbances[i,:] = signal.lfilter(primary_paths[i,:],1,primary_noise)
    
    # Generating the filtered references 
    num_ref = 4 
    num_sec = secondary_paths.shape[0]
    num_err = secondary_paths.shape[1]
    filterd_ref = np.zeros((num_ref,num_err,num_err,T*fs))
    
    for i in range(num_ref):
        for j in range(num_sec):
            for m in range(num_err):
                filterd_ref[i,j,m,:] = signal.lfilter(secondary_paths[j,m,:],1,primary_noise)
    
    Dis_Fx_dic={'Disturbances' : disturbances, 'Filtered_reference' : filterd_ref, 'Primary_noise': primary_noise}
    
    return Dis_Fx_dic

def generating_primary_noise(Cutoff_freq, T, fs):
    
    # the number of the noise type
    type_num       = len(Cutoff_freq)
    primary_noises = np.zeros((type_num,T*fs))
    
    for i in range(type_num):
        Length = 512 
        b1     = signal.firwin(numtaps=Length, cutoff=Cutoff_freq[i], fs=fs, pass_zero=False)
        noise  = np.random.randn(T*fs)
        primary_noises[i,:] = signal.lfilter(b1,1,noise)
    
    return primary_noises

def save_file_in_workspace(Folder, data_dic, file_name):
    current_dictory = os.getcwd()
    mat_file_folder = os.path.join(current_dictory, Folder)

    if os.path.exists(mat_file_folder) == False:
        os.mkdir(mat_file_folder)
    
    mat_file = os.path.join(mat_file_folder,file_name)

    savemat(mat_file, data_dic) 
    
if __name__ == "__main__":
    T  = 12 
    fs = 16000
    
    primary_noise_cut_fre = [[230, 1200], [1500, 3200], [980, 2500], [2300, 5300]]
    primary_noises        = generating_primary_noise(Cutoff_freq=primary_noise_cut_fre, T=T, fs=fs)
    
    for i in range(len(primary_noises)):
        d_fx_dic  = generate_disturbance_filter_x_noises(primary_noises[i,:], T, fs)
        file_name = f'board_noise_{primary_noise_cut_fre[i][0]}-{primary_noise_cut_fre[i][1]}.mat'
        save_file_in_workspace(Folder='raw_noise_data_tst',data_dic=d_fx_dic,file_name=file_name)
        print(f"{i}.===>Complished " + file_name + "<===")
    

   