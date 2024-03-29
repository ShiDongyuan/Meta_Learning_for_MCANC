#   __  __      _____       _     __  __  ____
#  |  \/  | ___|  ___|_  __| |   |  \/  |/ ___|
#  | |\/| |/ __| |_  \ \/ /| |   | |\/| |\___ \
#  | |  | | (__|  _|  >  < | |___| |  | | ___) |
#  |_|  |_|\___|_|   /_/\_\|_____|_|  |_||____/
#------------------------------------------------------------
import torch
import numpy as np 
import torch.nn as nn 
import torch.optim as optim
import scipy.signal as signal 
import progressbar
from   Bcolors import bcolors

from rich.console import Console
from rich.table import Column, Table
from rich.progress import track

#------------------------------------------------------------
# Class McFxLMS algorithm
#------------------------------------------------------------
class McFxLMS_algorithm():
    
    def __init__(self, Wc_initialization, Sec, device):
        """The multichannel FxLMS algorithm for the MCANC system, which has r_num references, s_num secondary source, and e_num error sensor.

        Args:
            Wc_initialization (_float32_): The initnal control filter coefficients. Its dimension is [r_num x s_num x Len]
            Sec (_float32_): The secondary path impulse response, whose dimension is [s_num x ls]
            device (_type_): 'GUP' or 'CPU'
        """
        self.r_num = Wc_initialization.shape[0] # The number of the reference sensors.
        self.s_num = Wc_initialization.shape[1] # The number of the secondary sources.
        self.len   = Wc_initialization.shape[2] # The length of the control filter.
        self.e_num = Sec.shape[1]               # The number of the error sensors. 
        self.ls    = Sec.shape[2]               # The length of the secondary estimations. 
        
        self.Wc    = torch.tensor(Wc_initialization.detach(), requires_grad=True, dtype=torch.float, device=device)
        self.Xd    = torch.zeros(self.r_num, self.len, dtype=torch.float, device=device)
        self.Xdm   = torch.zeros(self.ls , self.r_num, self.len, device=device) # [Ls x R_num x Len]
        self.Sec   = Sec.to(device)
        
        self.Yd    = torch.zeros(self.s_num, self.ls, device=device)
        #------------------------------------------------------#
        # print("<<---------------------------------------------------------------------->>")
        # print(bcolors.OKCYAN + f'Reference Num: {self.r_num}, Secondary source Num: {self.s_num}, Error sensors Num: {self.e_num}'+ bcolors.ENDC)
        # print(bcolors.OKCYAN + f'The length of control filter is {self.len}, Dimension of Wc is [{self.r_num} x {self.s_num}]'+ bcolors.ENDC)
        # print(bcolors.OKCYAN + f'The length of the secondary path is {self.ls}' + bcolors.ENDC)
        # print("<<---------------------------------------------------------------------->>")
        
        console = Console()
        table   = Table(show_header=True, header_style="bold magenta")
        table.add_column("Layers", style="dim", width=12,justify='center')
        table.add_column(f'[{self.r_num}x{self.s_num}x{self.e_num}] McFxLMS algorithm', justify="left")
        table.add_column("Dimension", justify="left")
        table.add_column("Elements", justify="left")
        table.add_column("Weights", justify="left")
        
        table.add_row(
            "-1-",
            "Control filter",
            f'[{self.r_num} x {self.s_num} x {self.len}]',
            f'{self.r_num*self.s_num}',
            f'{self.r_num*self.s_num*self.len}'
        )
        table.add_row(
            "-2-",
            "Secondary path estimate",
            f'[{self.s_num} x {self.e_num} x {self.ls}]',
            f'{self.e_num*self.s_num}',
            f'{self.s_num*self.e_num*self.ls}'
        )
        
        console.print(table)
    
    def feedforward(self, xin):
        '''
        Parameter:
        param1 - xin : the reference signal [R_num x 1]
        param2 - Dis : the disturbance signal [E_num]
        '''
        # Shifting the input delay line 
        self.Xd      = torch.roll(self.Xd,1,1)
        self.Xd[:,0] = xin
        # Shifting the input delay matrix  
        self.Xdm        = torch.roll(self.Xdm,1,0) # [Ls x R_num x Len]
        self.Xdm[0,:,:] = self.Xd 
        # ----------Computational Graph route 1 -------------------------------
        # Building the control singal >>--y_mac--<<
        Xd_e          = self.Xd.unsqueeze(1) # [R_num x 1 x Len]
        y_mac         = torch.einsum('rsl,rsl->rs',Xd_e,self.Wc) # [R_num x S_num]---> Filtering the reference signal via different control fitlers 
        y_out         = torch.einsum('rs->s',y_mac) # [S_num] ----> Sum the control sigal from different microphone.
        # Cutting the computational chain 
        y_out_NAG     = y_out.detach()
        self.Yd       = torch.roll(self.Yd,1,1)
        self.Yd[:,0]  = y_out_NAG
        # Generating the control signal 
        Yd_e          = self.Yd.unsqueeze(1) # [S_num x 1 x Ls]
        y_anti_mac    = torch.einsum('sel,sel->se', Yd_e, self.Sec) # [S_num x E_num] ---> Filtering the control signal. 
        y_anti        = torch.einsum('se->e',y_anti_mac) #[E_num] --> sun the anti-noise from different speakers.
        # e            = Dis - y_anti 
        # ----------Computational Graph route 2 -------------------------------
        # Buiding the control signal under the assumpation that all control 
        # filters are same at Ls time.
        Xdm_e         = self.Xdm.unsqueeze(2) # [Ls x R_num x 1     x Len]  
        Wc_e          = self.Wc.unsqueeze(0)  # [1  x R_num x S_num x Len]
        y_e_mac       = torch.einsum('nrsl,nrsl->nrs',Xdm_e,Wc_e) # [Ls x R_num x S_num]
        y_e_out       = torch.einsum('nrs->ns',y_e_mac) # [Ls x S_num]
        # Generating the control signal
        Yd_e_AG       = y_e_out.unsqueeze(1) # [Ls x 1 x S_num]
        Yd_e_AG       = torch.permute(Yd_e_AG,(2,1,0)) # [S_num x E_num x Ls]
        y_anti_mac_AG = torch.einsum('sel,sel->se',Yd_e_AG, self.Sec)
        y_anti_AG     = torch.einsum('se->e',y_anti_mac_AG)
        # E_estimate    = Dis - y_anti_AG 
              
        return y_anti_AG, y_anti 
    
    def LossFunction(self, y_anti_AG, y_anti, Dis): 
        '''
        Parameter:
        param1 - y_anti_AG : the anti-nose signal [E_num] with auto gradient 
        param2 - y_anti    : the anti-noise signal 
        param3 - Dis       : the disturbance signal [E_num]
        '''
        e          = Dis - y_anti 
        E_estimate = Dis - y_anti_AG
        loss       = torch.sum(2*torch.einsum('e,e->e', e, E_estimate))
        return loss, e 
    
    def _get_coeff_(self):
        return self.Wc.cpu().detach().numpy()

#------------------------------------------------------------------------------
# Function : train_fxlms_algorithm() 0.00000005
#------------------------------------------------------------------------------
def train_fxmclms_algorithm(Model, Ref, Disturbance, device, Stepsize = 0.00000005, so = None):
    '''
    Parameter:
    param1 - Model : the instance of the multichannel FxLMS algorithm.
    param2 - Ref   : the reference signal vector [R_num x T] 
    R_num-- the number of the reference microphones. T -- the time index.
    param3 - Disturbance : the distrubance vector [E_num x T].
    param4 - Stepsize : the value of the step size. 
    '''
    #--------------------------------------------
    # if torch.cuda.is_available():
    #     device = "cuda"
    # else:
    #     device = "cpu"
    print(bcolors.OKCYAN + f"Using {device} for training the McFxLMS algorithm !!!" + bcolors.ENDC)
    #--------------------------------------------
    
    print(bcolors.WARNING + "<<-------------------------------START---------------------------------->>" + bcolors.ENDC)
    print(f'The length of the data is {Disturbance.shape[1]}.')
    
    #Stepsize = 0.00000005  
    optimizer= optim.SGD([Model.Wc], lr=Stepsize)
    
    # bar.start()
    Erro_signal = []
    len_data    = Disturbance.shape[1]
    Ref         = Ref.to(device)
    Disturbance = Disturbance.to(device)
    for itera in track(range(len_data),description="Processing..."):
        # Feedfoward
        xin = Ref[:,itera]
        dis = Disturbance[:,itera]
        y_anti_AG, y_anti = Model.feedforward(xin)
        loss,e            = Model.LossFunction(y_anti_AG, y_anti, dis)
            
        # Backward 
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()
        Erro_signal.append(e.cpu().numpy())
        if so != None:
            so.progress_update.emit(itera+1)
        
    print(bcolors.WARNING + "<<-------------------------------END------------------------------------>>" + bcolors.ENDC)
    return Erro_signal
#------------------------------------------------------------
# Function : Generating the testing bordband noise 
#------------------------------------------------------------
def Generating_boardband_noise_wavefrom_tensor(Wc_F, Seconds, fs):
    filter_len = 1024 
    bandpass_filter = signal.firwin(filter_len, Wc_F, pass_zero='bandpass', window ='hamming',fs=fs) 
    N = filter_len + Seconds*fs
    xin = np.random.randn(N)
    y   = signal.lfilter(bandpass_filter,1,xin)
    yout= y[filter_len:]
    # Standarlize 
    yout = yout/np.sqrt(np.var(yout))
    # return a tensor of [1 x sample rate]
    return torch.from_numpy(yout).type(torch.float).unsqueeze(0)

#------------------------------------------------------------
# Function : Generating the testing bordband noise 
#------------------------------------------------------------
def Generating_boardband_noise_wavefrom_tensor(Wc_F, Seconds, fs):
    filter_len = 1024 
    bandpass_filter = signal.firwin(filter_len, Wc_F, pass_zero='bandpass', window ='hamming',fs=fs) 
    N = filter_len + Seconds*fs
    xin = np.random.randn(N)
    y   = signal.lfilter(bandpass_filter,1,xin)
    yout= y[filter_len:]
    # Standarlize 
    yout = yout/np.sqrt(np.var(yout))
    # return a tensor of [1 x sample rate]
    return torch.from_numpy(yout).type(torch.float).unsqueeze(0)

#-------------------------------------------------------------
# Function    : Disturbance_reference_generation_from_Fvector()
# Discription : Generating the distubrane and reference signal from the defined parameters
#-------------------------------------------------------------
def Disturbance_Noise_generation_from_Fvector(fs, T, f_vector, Pri_path, Sec_path):
    """
    Pri_path and Sec_path are  One dimension arraies 
    """
    # ANC platform configuration
    t     = np.arange(0,T,1/fs).reshape(-1,1)
    len_f = 1024
    b2    = signal.firwin(len_f, [f_vector[0],f_vector[1]], pass_zero='bandpass', window ='hamming',fs=fs)
    xin   = np.random.randn(len(t))
    Re    = signal.lfilter(b2,1,xin)
    Noise = Re[len_f-1:]
    # Noise = Noise/np.sqrt(np.var(Noise))
    
    # Construting the desired signal 
    Dir, Fx = signal.lfilter(Pri_path, 1, Noise), signal.lfilter(Sec_path, 1, Noise)
    
    return torch.from_numpy(Dir).type(torch.float), torch.from_numpy(Noise).type(torch.float)

#-------------------------------------------------------------------