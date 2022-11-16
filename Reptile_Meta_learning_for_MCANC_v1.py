from torch  import nn, float32, zeros, einsum

class Reptil_Meta(nn.Module):
    def __init__(self, num_ref, num_sec, Len_c):
        super().__init__()
        self.Control_filter  = nn.Parameter(zeros(size=(num_ref, num_sec, Len_c), dtype=float32))
        self.Lenc            = Len_c

    def Adaptive_filtering(self, Fx):
        anti_noise_elements = einsum('...rsen,...rsn ->...rse', Fx, self.Control_filter)
        anti_noise          = einsum('...rse->...e', anti_noise_elements)
        return anti_noise
    
    def forward(self, Fx):
        anti_noise = self.Adaptive_filtering(Fx)
        return anti_noise

def LossFunction_reptile(anti_noise, Dis):
    Error_signal = Dis - anti_noise 
    loss         = einsum('e, e ->', Error_signal, Error_signal)
    
    return loss 
