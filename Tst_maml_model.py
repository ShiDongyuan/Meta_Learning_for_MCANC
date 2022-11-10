from torch.utils.data         import Dataset, DataLoader
from Modified_MAMAL_for_MCANC import Modified_MAML, Loss_Function_maml, bulidng_data_by_DelayLine
from Buiding_custom_dataset   import Noise_Dataset

root_dir = 'noise_dataset'
csv_file = 'index.csv'
    
    
nois_dataset = Noise_Dataset(csv_file, root_dir)
    
Fx, Dis= nois_dataset[0]
print(Fx.shape)
print(Dis.shape)
    
train_data2 = DataLoader(dataset=nois_dataset, batch_size=1,shuffle=True)
    
Fxs, Diss = next(iter(train_data2))
print(f'Infor 1: Fxs dimension is {Fxs.shape} and Diss dimension is {Diss.shape}')

# Input data of the model
Fx  = Fxs[0]
Dis = Diss[0]

# Building the instance of the maml model
Meta_model = Modified_MAML(num_ref=4, num_sec=4, Len_c=512, L_r=0.00001, Gamma=0.99)

# Construting the delay-line based data 
Fx_extend = bulidng_data_by_DelayLine(Len_c=512, Fx=Fx)
print(f'Infor 2: Fx_extend dimenstion is {Fx_extend.shape}')

# Forward part 
out_1, out_2 = Meta_model(Fx, Dis, Fx_extend)

print(f' Infor 3: out dimension is {out_1.shape}')
print(f' Infor 4: forgeting factor vector is {out_2}')

# print(*Meta_model.parameters()[0].shape)

for i in Meta_model.parameters():
    
    print(i.shape)
    
model_dict = Meta_model.state_dict()

for k, v in model_dict.items():                                   
       print(k)
       print(v.shape)

