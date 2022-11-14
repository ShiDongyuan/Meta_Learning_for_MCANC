import os 
import torch
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

def frequency_response_depict(Wc):
    fig, axs = plt.subplots(2)
    w, h = signal.freqz(Wc, 1, fs=16000)
    axs[0].set_title('Impulse response')
    axs[0].plot(Wc)
    axs[0].set_xlabel('Taps')
    axs[0].grid()
    axs[1].set_title('Digital filter frequency response')
    axs[1].plot(w, 20*np.log10(np.abs(h)))
    axs[1].set_title('Digital filter frequency response')
    axs[1].set_ylabel('Amplitude Response [dB]')
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].grid()
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