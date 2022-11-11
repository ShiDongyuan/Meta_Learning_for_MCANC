import os 
import torch
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

def frequency_response_depict(Wc):
    w, h = signal.freqz(Wc, 1, fs=16000)
    plt.title('Digital filter frequency response')
    plt.plot(w, 20*np.log10(np.abs(h)))
    plt.title('Digital filter frequency response')
    plt.ylabel('Amplitude Response [dB]')
    plt.xlabel('Frequency (rad/sample)')
    plt.grid()
    plt.show()

if __name__=="__main__":
    path_root = 'pth_modle'
    pth_file  = 'MAML_v1.pth'
    
    model_file = os.path.join(path_root, pth_file)
    
    model_dict = torch.load(model_file)
    
    print(model_dict['initial_weigths'].shape)
    
    Wc = model_dict['initial_weigths'][0,0,:]
    
    # for k, v in model_dict.items(): 
    #     print(k)
    frequency_response_depict(Wc)