import numpy as np
import torch
from dataloader_func import add_noise_torch
######################################################################################
################################ quality metrics #####################################
######################################################################################

def MSE(ref_im, im ):
    return ((ref_im - im)**2).mean()

def batch_ave_psnr_torch(ref_ims, images ,max_I):
    '''
    batch of ref_im and im are tensors of dims N,C, W, H
    '''
    mse = ((ref_ims - images)**2).mean(dim=(1,2,3))
    psnr_all =  10*(np.log10(max_I**2) - torch.log10(mse))
    return psnr_all.mean()



def batch_psnr_numpy(ref_ims, images ,max_I):
    '''
    batch of ref_im and im are tensors of dims N, W, H
    '''
    mse = ((ref_ims - images)**2).mean(axis=(1,2))
    psnr_all =  10*(np.log10(max_I**2) - np.log10(mse))
    return psnr_all




def calc_psnr_range(clean, noise_range, denoiser): 

    psnrs = {}

    for sigma in noise_range:     
        noise = torch.randn_like(clean)*sigma/255
        noisy = clean + noise.cuda()

        with torch.no_grad(): 
            denoised = noisy - denoiser(noisy).detach()    
        psnrs[sigma.item()] = batch_ave_psnr_torch(clean.cpu() , denoised.cpu(), 1)            

    return psnrs






