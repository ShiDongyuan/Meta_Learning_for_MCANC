from torch          import nn, float32, zeros, einsum, roll, transpose
from torch.autograd import Variable

def bulidng_data_by_DelayLine(Len_c, Fx):
    """Constructing the delay-line input based on the filtered reference matrix. 

    Args:
        Len_c (_type_): The length of the control filter or the delay-line
        Fx (_type_): The filtered reference matrix 

    Returns:
        _type_: the input data matrix based on delay-line operation.
    """
    
    # deterime the dimension of the filtered reference matrix 
    num_ref = Fx.shape[0]
    num_sec = Fx.shape[1]
    num_err = Fx.shape[2]
    num_len = Fx.shape[3]
    
    # constructing the delayline data 
    Fx_extend = zeros(size=(Len_c, num_ref, num_sec, num_err, Len_c), dtype=float32)
    Fx_delay  = zeros(size=(num_ref, num_sec, num_err, Len_c), dtype=float32)
    
    for i in range(Len_c):
        Fx_delay          = roll(Fx_delay, 1, dims=3)
        Fx_delay[:,:,:,0] = Fx[:,:,:,i]
        Fx_extend[i]      = Fx_delay
    
    return Fx_extend

class Reptile_Meta(nn.Module):
    
    def __init__(self, num_ref, num_sec, Len_c, Gamma, device='cpu'):
        """ The reptile meta learning algorithm for MCANC system, which has num_ref reference sensors, num_sec secondary sources. 

        Args:
            num_ref (_int_): The number of the reference sensors. 
            num_sec (_int_): The number of the secondary sources. 
            Len_c   (_int_): The lenght of the control filter.
            Gamma   (_float32_): The forgeting factor. 
            device (str, optional): The processor used for training the Meta learning model. Defaults to 'cpu'.
        """
        super().__init__()
        self.Control_filter  = nn.Parameter(zeros(size=(num_ref, num_sec, Len_c), dtype=float32))
        self.Lenc            = Len_c
        self.device          = device
        self.gamma           = Gamma 
        self.Gam_vector      = self.Construct_Gamma_vector()

    def Construct_Gamma_vector(self):
        Gam_vector = zeros(self.Lenc , dtype=float32)
        for i in range(self.Lenc):
            Gam_vector[i] = self.gamma**(self.Lenc-1-i)
        
        return Gam_vector.to(self.device)

    def Adaptive_filtering(self, Weights, Fx):
        anti_noise_elements = einsum('...rsen,...rsn ->...rse', Fx, Weights)
        anti_noise          = einsum('...rse->...e', anti_noise_elements)
        
        return anti_noise
    
    def forward(self, Fx):
        # Adaptive filtering
        anti_noise_matrix = self.Adaptive_filtering(Weights=self.Control_filter, Fx=Fx)
        
        return anti_noise_matrix, self.Gam_vector
    
def LossFunction_reptile(anti_noise, Dis, gamma_element):
    Error_signal = Dis - anti_noise 
    loss         = gamma_element*einsum('e, e ->', Error_signal, Error_signal)
    
    return loss 