import numpy             as np 
import scipy.signal      as signal 
import matplotlib.pyplot as plt 
import os

from scipy.io import savemat

def plot_frequency_response_of_filter(filter_impluse_response, fs):
    w1, h1 = signal.freqz(filter_impluse_response, fs=fs)
    plt.title('Digital filter frequency response')
    plt.plot(w1, 20*np.log10(np.abs(h1)), 'b')
    plt.ylabel('Amplitude Response (dB)')
    plt.xlabel('Frequency (Hz)')
    plt.grid()
    plt.show()

def Design_filter(Cutoff_freq, Length, fs):
    b1 = signal.firwin(numtaps=Length, cutoff=Cutoff_freq, fs=fs, pass_zero=False)
    return b1 

def construct_primary_secondary_paths(Primary_freq, Secondary_freq, Len_p=512, Len_s=256, fre_sample=16000):
    num_primary         = len(Primary_freq)
    primary_path_matrix = np.zeros((num_primary, Len_p))

    for i in range(num_primary):
        b                        = Design_filter(Cutoff_freq=Primary_freq[i], Length=Len_p, fs=fre_sample)
        primary_path_matrix[i,:] = b
        # plot_frequency_response_of_filter(b,fs)

    num_secondary         = len(Secondary_freq)
    num_error             = len(Secondary_freq[0])
    secondary_path_matrix = np.zeros((num_secondary,num_error,Len_s))

    for i in range(num_secondary):
        for j in range(num_error):
            b                            = Design_filter(Cutoff_freq=Secondary_freq[i][j], Length=Len_s, fs=fre_sample)
            secondary_path_matrix[i,j,:] = b
            # plot_frequency_response_of_filter(b,fs)
    
    return primary_path_matrix, secondary_path_matrix

def save_file_in_workspace(Folder, data_dic):
    current_dictory = os.getcwd()
    mat_file_folder = os.path.join(current_dictory, Folder)

    if os.path.exists(mat_file_folder) == False:
        os.mkdir(mat_file_folder)
    
    mat_file = os.path.join(mat_file_folder,"path_matrix.mat")

    savemat(mat_file, data_dic) 

if __name__ == "__main__":
    
    # the sample rate 
    fs = 16000 
    # the length of the filter 
    L  = 256 
    # the cut-off frequencies 
    cut_freq = [200, 1200] 

    b1 = Design_filter(cut_freq, L, fs)

    plot_frequency_response_of_filter(b1,fs=fs)

    # print(b1.shape)

    # L_tst = [[[3, 4], [3, 6], [7, 8]],[[3, 4], [3, 6], [7, 8]]]
    # print(len(L_tst[0]))
    # print(L_tst[1])

    # s_tst = np.zeros((3,4,5))
    # print(s_tst.shape)

    # print (os.getcwd())

    # file_addrss = os.path.join(os.getcwd(),'Path_data')
    # print(os.path.exists(file_addrss) == False)

    Pri_path_frec = [[200, 6600],[120, 6400],[150, 6700],[80, 6200]]
    Sec_path_frec = [[[200, 6600],[120, 6400],[150, 6700],[80, 6200]]
                    , [[100, 6600],[120, 6400],[150, 6700],[80, 6200]]
                    , [[70, 6600],[120, 6400],[150, 6700],[80, 6200]]
                    , [[90, 6600],[120, 6400],[150, 6700],[80, 6200]]]

    Primary_paths, Secondary_paths = construct_primary_secondary_paths(Pri_path_frec, Sec_path_frec) 
    path_dic = { 'primary_path_matrix': Primary_paths
                ,'secondary_path_matrix': Secondary_paths}

    save_file_in_workspace(Folder='Path_data',data_dic=path_dic)
    