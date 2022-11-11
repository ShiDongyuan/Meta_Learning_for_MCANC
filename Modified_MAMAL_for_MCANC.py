#     __  __           _ _  __ _           _    __  __     _    __  __ _        __  __       _
#    |  \/  | ___   __| (_)/ _(_) ___   __| |  |  \/  |   / \  |  \/  | |      |  \/  | ___ | |_  __ _
#    | |\/| |/ _ \ / _` | | |_| |/ _ \ / _` |  | |\/| |  / _ \ | |\/| | |      | |\/| |/ _ \| __|/ _` |
#    | |  | | (_) | (_| | |  _| |  __/| (_| |  | |  | | / ___ \| |  | | |___   | |  | |  __/| |_| (_| |
#    |_|  |_|\___/ \__,_|_|_| |_|\___| \__,_|  |_|  |_|/_/   \_\_|  |_|_____|  |_|  |_|\___| \__|\__,_|

from torch import nn, float32, zeros, einsum, roll, transpose
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

class Modified_MAML(nn.Module):
    
    def __init__(self, num_ref, num_sec, Len_c, L_r, Gamma, device='cpu'):
        """ The modified MAML meta learning algorithm for MCANC system (R x S x E). The MCANC system has R references, S secondary sources, and E error sensors. 

        Args:
            num_ref (_int_): The number of the reference sensors.
            num_sec (_int_): The number of the secondary sources. 
            Len_c (_int_): The length of the control filter.
            L_r (_float_): The learning rate for the first gradient descent algorithm.
            Gamma (_float_): The forgeting facotr that is used avoid the zeor-padding effect in the delay-line.
            device (_str__): Using 'GPU' or 'cpu' for trainning. 
        """
        super().__init__()
        self.initial_weigths = nn.Parameter(zeros(size=(num_ref,num_sec, Len_c), dtype=float32))
        self.Lenc            = Len_c
        self.lr              = L_r
        self.device          = device
        self.gamma           = Gamma 
        self.Gam_vector      = self.Construct_Gamma_vector()
    
    def Construct_Gamma_vector(self):
        Gam_vector = zeros(self.Lenc , dtype=float32)
        for i in range(self.Lenc):
            Gam_vector[i] = self.gamma**(self.Lenc-1-i)
        
        return Gam_vector.to(self.device)
    
    def First_grad(self, initial_weigths, Fx, Dis):
        weights_A      = Variable(initial_weigths.detach(), requires_grad=True).to(self.device)
        anti_noise_ele = einsum('rsen, rsn->rse', Fx, weights_A)
        anti_noise     = einsum('rse->e', anti_noise_ele)
        error          = Dis[:,-1]-anti_noise
        Loss_1         = einsum('i,i->', error, error)
        Loss_1.backward()
        
        return weights_A.grad.detach()
    
    def Adaptive_filtering(self, Weights, Fx):
        anti_noise_elements = einsum('...rsen,...rsn ->...rse', Fx, Weights)
        anti_noise          = einsum('...rse->...e', anti_noise_elements)
        
        return anti_noise
        
    def forward(self, Fx, Dis, Fx_extend):
        """ Computing the anti-noise based on the optimal control filter given by the intial filter weights. 

        Args:
            Fx (_float_): The filterd reference matrix, whose dimension is [ R x S x E x Len].
            Dis (_type_): The disturbance matrix, whose dimension is [E x Len].
            Fx_extend (_type_): The filtered reference matrix of delay-line at the whole duration, whose dimension is [T x R x S x E x Len].
            (R: number of references; S: number of secondary sources; E: number of the error; Len: length of the control filter; T: the number of samples during one iteration)
        Returns:
            _type_: Anti noise matrix, whose dimension is [T x E]
        """
        
        # Obtaining the gradint of the loss function
        weights_A_grad    = self.First_grad(initial_weigths=self.initial_weigths, Fx=Fx, Dis=Dis)
        
        # One step updationg 
        control_weights   = self.initial_weigths - 0.5*self.lr*weights_A_grad
        
        # Adaptive filtering 
        anti_noise_matrix = self.Adaptive_filtering(Weights=control_weights, Fx=Fx_extend)
        
        return anti_noise_matrix, self.Gam_vector

def Loss_Function_maml(anti_noise_matrix, Dis, Gam_vector):
    """ Sequared error loss function for modified MAML algorithm

    Args:
        anti_noise_matrix (_float_): The anti noise matrix, whose dimension is [T x E]
        Dis (_float_): The disturbance mateix, whose dimension is [E x T]
        Gam_vector (_folat_): The vector contains forgeting factors for different iteration, its dimension is [T]

    Returns:
        _type_: The summation of the error singals at the first Len iterations. 
    """
    error_vector = Dis - transpose(anti_noise_matrix, 1, 0) 
    loss = einsum('t,t->', einsum('st,st->t', error_vector, error_vector), Gam_vector)
    
    return loss 

if __name__=="__main__":
    
    Meta_model = Modified_MAML(num_ref=4, num_sec=4, Len_c=512, L_r=0.00001, Gamma=0.9)
    print(Meta_model)