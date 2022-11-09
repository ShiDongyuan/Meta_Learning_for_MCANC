import numpy             as np 
import scipy.signal      as signal 
import matplotlib.pyplot as plt 
import pandas            as pd
import os
import time

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
    
    Dis        = np.array([])
    Filtered_X = np.array([])
    i = True 
    for path, dir_list, file_list in g:
        for file_name in file_list:
            mat_file_name = os.path.join(mat_folder,file_name)
            assert os.path.exists(mat_file_name)
            Dis_Fx_dic    = loadmat(mat_file_name)
            
            if i == True :
               y = Dis_Fx_dic['Disturbances']
               Dis = np.zeros_like(y)
               Dis = y.copy()

               y = Dis_Fx_dic['Filtered_reference']
               Filtered_X = np.zeros_like(y)
               Filtered_X = y.copy()
               i   = False
            else:
                y = Dis_Fx_dic['Disturbances']
                Dis = np.append(Dis,y,axis=1)
                
                y = Dis_Fx_dic['Filtered_reference']
                Filtered_X = np.append(Filtered_X,y,axis=3)
    return Dis, Filtered_X

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
    Len_data = Dis.shape[1]
    print(Len_data)
    index_vector = np.random.randint(low=0, high=Len_data-Len_c-1, size=Num_data)
    
    file_name_list = []
    
    bar = progressbar.ProgressBar(maxval=Num_data-1, \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    
    bar.start()
    for i in range(Num_data):
        index_start = index_vector[i]
        index_end   = index_start+Len_c
        noise_dict  = {'disturbance': Dis[:,index_start:index_end], 'filtered_reference': Fx[:,:,:,index_start:index_end]}
        file_name   = f'{i}.mat'
        file_name_list.append(file_name)
        save_file_in_workspace(Folder='noise_dataset', data_dic=noise_dict, file_name=file_name)
        bar.update(i)
    name_dict   = {'file_name':file_name_list}  
    save_csv_in_workspace(Folder='noise_dataset', name_dic=name_dict, index_file_name='index.csv')    
    bar.finish()
                
if __name__ == "__main__": 
    Dis, Fx  = load_raw_noise_from_workspace()
    Num_data = 10000 
    Len_c    = 512
    building_data_set(Num_data, Len_c, Dis, Fx)
    