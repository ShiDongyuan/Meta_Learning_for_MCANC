import os 
import torch 
import pandas as pd 
import numpy  as np 

from scipy.io         import loadmat
from torch.utils.data import Dataset, DataLoader

class Noise_Dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.root_dir = root_dir 
        self.cvs_file = os.path.join(root_dir, csv_file)
        
        assert os.path.exists(self.cvs_file)
        self.landmark_frame = pd.read_csv(self.cvs_file)
        self.transform      = transform
    
    def __len__(self):
        return len(self.landmark_frame)

    def __getitem__(self, index):
        mat_file =os.path.join(self.root_dir,
                               self.landmark_frame['file_name'][index])
        
        assert os.path.exists(mat_file)
        Dis_Fx_dic = loadmat(mat_file)
        Dis        = torch.from_numpy(Dis_Fx_dic['disturbance']).to(torch.float32)
        Fx         = torch.from_numpy(Dis_Fx_dic['filtered_reference']).to(torch.float32)
        # sample = {'disturbance': Dis, 'filtered_reference': Fx}
        
        if self.transform:
            sample =self.transform(Fx, Dis)
        
        return Fx, Dis
        

if __name__ == "__main__":
    root_dir = 'noise_dataset'
    csv_file = 'index.csv'
    
    # pf = pd.read_csv(os.path.join(root_dir,csv_file),index_col=0)
    
    # print(pf.head())
    # print(pf['file_name'][3])
    # print(len(pf))
    
    nois_dataset = Noise_Dataset(csv_file, root_dir)
    
    Fx, Dis= nois_dataset[0]
    print(Fx.shape)
    print(Dis.shape)
    
    train_data2 = DataLoader(dataset=nois_dataset, batch_size=1,shuffle=True)
    
    Fxs, Diss = next(iter(train_data2))
    print(Diss[0].shape)
        
