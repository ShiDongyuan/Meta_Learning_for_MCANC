import numpy             as np 
import scipy.signal      as signal 
import matplotlib.pyplot as plt 
import pandas            as pd
import os

import progressbar
from scipy.io import loadmat, savemat

def load_raw_noise_from_workspace(Folder='raw_noise_data'):
    """Reading the raw data from the workspace 

    Args:
        Folder (str, optional): The address of the folder containing the raw noise data. Defaults to 'raw_noise_data'.

    Returns:
        _type_: Disturbances and filtered reference vectors. 
    """
    current_dictory = os.getcwd()
    mat_folder      = os.path.join(current_dictory, Folder)
    assert os.path.exists(mat_folder)
    g               = os.walk(mat_folder)
    
    
    Loaded_Dis = {} 
    Loaded_Fxs = {}
    for path, dir_list, file_list in g:
        for file_name in file_list:
            mat_file_name = os.path.join(mat_folder,file_name)
            assert os.path.exists(mat_file_name)
            Dis_Fx_dic    = loadmat(mat_file_name)
            Loaded_Dis[file_name] = Dis_Fx_dic['Disturbances']
            Loaded_Fxs[file_name] = Dis_Fx_dic['Filtered_reference']
    
    return Loaded_Dis, Loaded_Fxs

def save_file_in_workspace(Folder, data_dic, file_name):
    current_dictory = os.getcwd()
    mat_file_folder = os.path.join(current_dictory, Folder)

    if os.path.exists(mat_file_folder) == False:
        os.mkdir(mat_file_folder)
    
    mat_file   = os.path.join(mat_file_folder,file_name)

    savemat(mat_file, data_dic) 

def save_csv_in_workspace(Folder, name_dic, index_file_name):
    current_dictory = os.getcwd()
    mat_file_folder = os.path.join(current_dictory, Folder)

    if os.path.exists(mat_file_folder) == False:
        os.mkdir(mat_file_folder)
    
    index_file = os.path.join(mat_file_folder,index_file_name)
    df = pd.DataFrame(name_dic)
    df.to_csv(index_file)

def building_data_set(Num_data, Len_c, Dis, Fx):
    keys       = list(Dis.keys())
    Num_file   = len(keys)
    Posibility = np.zeros(Num_file)
    
    for i in range(Num_file):
        Posibility[i] = Dis[keys[i]].shape[1]
    Posibility = Posibility/np.sum(Posibility)
    
    Samples_indx = np.random.choice(Num_file, size=Num_data,replace=True,p=Posibility)
    
    file_name_list = []
    type_name_list = []

    bar = progressbar.ProgressBar(maxval=Num_data-1, \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    
    bar.start()
    
    for i in range(Num_data):
        noise_track_index = Samples_indx[i] 
        T_num             = Dis[keys[noise_track_index]].shape[1]
        index_start       = np.random.randint(low=0, high=T_num-Len_c)
        index_end         = index_start+Len_c
        noise_dict        = {'disturbance': Dis[keys[noise_track_index]][:,index_start:index_end], 'filtered_reference': Fx[keys[noise_track_index]][:,:,:,index_start:index_end]}
        file_name         = f'{i}.mat'
        file_name_list.append(file_name)
        type_name_list.append(keys[noise_track_index])
        save_file_in_workspace(Folder='noise_dataset', data_dic=noise_dict, file_name=file_name)
        bar.update(i)
    name_dict   = {'file_name':file_name_list, 'Type_noise':type_name_list}  
    save_csv_in_workspace(Folder='noise_dataset', name_dic=name_dict, index_file_name='index.csv')   
    bar.finish()

if __name__ == "__main__": 
    Dis, Fx  = load_raw_noise_from_workspace()
    Num_data = 10000 
    Len_c    = 512
    building_data_set(Num_data, Len_c, Dis, Fx)