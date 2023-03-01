import numpy as np
import torch
import torch.nn as nn
import os
import argparse
from network import *

################################################# Helper Functions #################################################
def init_BF_CNN_RF(RF=43, coarse=True,num_channels=1, my_args=None):
    '''
    loads flat BF_CNN with RF flexibility.
    @my_args
    '''

    parser = argparse.ArgumentParser(description='flat BF_CNN')
    parser.add_argument('--kernel_size', default= 3)
    parser.add_argument('--padding', default= 1)
    parser.add_argument('--num_kernels', default= 64)
    parser.add_argument('--num_layers', default= 21)
    parser.add_argument('--RF',default = RF)
    parser.add_argument('--coarse', default= coarse) 
    parser.add_argument('--num_channels', default= num_channels)

    args = parser.parse_args('')
    
    if my_args is not None: ## update args with given args
        for key, value in vars(myargs).items():
                vars(args)[key] = value
                
    model = BF_CNN_RF(args)
    
    if torch.cuda.is_available():
        model = model.cuda()
    return model

def init_BF_CNN(my_args=None):
    '''
    loads flat BF_CNN with RF flexibility.
    @ grayscale: if True, number of input and output channels are set to 1. Otherwise 3
    '''

    parser = argparse.ArgumentParser(description='flat BF_CNN')
    parser.add_argument('--kernel_size', default= 3)
    parser.add_argument('--padding', default= 1)
    parser.add_argument('--num_kernels', default= 64)
    parser.add_argument('--num_layers', default= 20)
    parser.add_argument('--num_channels', default= 1)
    parser.add_argument('--coarse', default= True) 
    parser.add_argument('--first_layer_linear', default= True) 
    

    args = parser.parse_args('')

    if my_args is not None: ## update args with given args
        for key, value in vars(my_args).items():
                vars(args)[key] = value

    model = BF_CNN(args)
    if torch.cuda.is_available():
        model = model.cuda()
            
    return model


def read_trained_params(model, path): 
               
    if torch.cuda.is_available():
        learned_params =torch.load(path)
    else:
        learned_params =torch.load(path, map_location='cpu' )
        
    ## unwrap if in Dataparallel 
    new_state_dict = {}
    for key,value in learned_params.items(): 
        if key.split('.')[0] == 'module': 
            new_key = '.'.join(key.split('.')[1::])
            new_state_dict[new_key] = value

        else: 
            new_state_dict[key] = value
        

    model.load_state_dict(new_state_dict)        
    model.eval();

    return model



def load_multi_scale_denoisers_RF(base_path, training_data_name, training_noise,RF_low, RF, J): 
    models = {}
    
    ### load the low freq denoiser    
    if RF_low==43:
        init_coarse = init_BF_CNN_RF(RF=RF_low, coarse=True)
    elif RF_low==40:
        init_coarse = init_BF_CNN()
        
    path = os.path.join(base_path,'multi_scale',training_data_name, training_noise+'_RF_'+str(RF_low)+'x'+str(RF_low)+'_low','model.pt')
    models['low'] = read_trained_params(init_coarse, path)
    ### load the conditional denoisers
    for j in range(J):
        init_fine = init_BF_CNN_RF(RF=RF, coarse=False)
        path = os.path.join(base_path,'multi_scale',training_data_name, training_noise+'_RF_'+str(RF)+'x'+str(RF)+'_scale_'+str(j),'model.pt')
        models[j] = read_trained_params(init_fine, path)       

    return models


def load_BF_CNN_RF(base_path, training_data_name, training_noise, RF):

    init = init_BF_CNN_RF(RF=RF, coarse=True)
    path = os.path.join(base_path,'flat',training_data_name, training_noise+'_RF_'+str(RF)+'x'+str(RF),'model.pt')
    model = read_trained_params(init, path)
    return model

