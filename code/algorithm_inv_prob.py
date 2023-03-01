import numpy as np
import torch
import time
from wavelet_func import reconstruct
from dataloader_func import rescale_image_range
### Takes a tensor of size (n_ch, im_d1, im_d2)
### and returns a tensor of size (n_ch, im_d1, im_d2)

def one_scale_synthesis(model, init_im , sig_0=1, sig_L=.01, h0=.01 , beta=.01 , freq=0,device=None, fixed_h = False):
    
    '''
    @model: denoiser to be used in the algorithm
    @init_im: either tuple=(C,H,W) representing size of low res image being synthesied, 
    or input tensor of size (3*C+1, H, W) if the goal is to generate the low pass as well
    @sig_0: initial sigma (largest)
    @sig_L: final sigma (smallest)
    @h0: 1st step size
    @beta:controls added noise in each iteration (0,1]. if 1, no noise is added. As it decreases more noise added.
    @freq: frequency at which intermediate steps will be logged 
    @fixed_h: if False, it uses an ascending h schedule to speed up convergence in later stages
    '''
    

        
    intermed_Ys=[]
    sigmas = []
    means = []
    
    ### initialization 
    if type(init_im) is tuple: ### If generating low resolution image. Unconditional synthesis from global prior
        C, H,W = init_im
        conditional=False

        
    else:  ### If generating details. Conditional  
        C, H,W = init_im.size()
        C = 3*C
        conditional=True
        
        
    e = torch.zeros((C ,H,W), requires_grad= False , device=device)
    N = C*H*W #correct?
    y = torch.normal(e, sig_0)      
    y = y.unsqueeze(0)
    y.requires_grad = False
        
        
    if freq > 0:
        intermed_Ys.append(y.squeeze(0))

    t=1
    sigma = torch.tensor(sig_0)
    start_time_total = time.time()
    snr = 20*torch.log10((y.std()/sigma)).item()
    snr_L = 20*torch.log10((torch.tensor([1])/sig_L)).item()    
    while snr < snr_L: 
        h = h0
        
        if fixed_h is False:
            h = h0*t/(1+ (h0*(t-1)) )

            
        with torch.no_grad():
            if conditional:
                f_y = model(torch.cat((init_im.unsqueeze(0), y), dim=1)) 
            else: 
                f_y = model(y)

        sigma = torch.norm(f_y)/np.sqrt(N)
        sigmas.append(sigma)
        gamma = sigma*np.sqrt(((1 - (beta*h))**2 - (1-h)**2 ))
        noise = torch.randn(C, H, W, device=device) 
        
        
        if freq > 0 and t%freq== 0:
            print('-----------------------------', t)
            print('sigma ' , sigma.item() )
            print('mean ', y.mean().item() )
            intermed_Ys.append(y.squeeze(0))
            
        y = y -  h*f_y + gamma*noise 
        means.append(y.mean().item())
        snr = 20*torch.log10((y.std()/sigma)).item()        
        
        
        t +=1
        
        if sigma > 1.5: 
            break
    print("-------- total number of iterations, " , t )
    print("-------- average time per iteration (s), " , np.round((time.time() - start_time_total)/(t)  ,4) )
    print("-------- final sigma, " , sigma.item() )
    print('-------- final mean ', y.mean().item() )
    print("-------- final snr, " , 20*torch.log10((y.std()/sigma)).item() )

    if conditional:
        denoised_y = y - model(torch.cat((init_im.unsqueeze(0), y), dim=1)) 
    else:             
        denoised_y = y - model(y)

    return denoised_y.squeeze(0), intermed_Ys, sigmas, means




def multi_scale_synthesis(models, init_im  , sig_0, sig_L, h0 , beta , freq,device = None, orth_forward=True, seeds =None, fixed_h = False):
    '''
    @model: denoiser to be used in the algorithm
    @init_im: either tuple=(C,H,W) representing size of low res image being synthesied, 
    or input tensor of size (3*C+1, H, W) if the goal is to generate the low pass as well
    @sig_0: initial sigma (largest)
    @sig_L: final sigma (smallest)
    @h0: 1st step size
    @beta:controls added noise in each iteration (0,1]. if 1, no noise is added. As it decreases more noise added.
    @freq: frequency at which intermediate steps will be logged 
    @seeds: if not none, it should be set to an integer to set the seeds manually 
    @orth_forward: True if wavelet coefficents are normalized. False if they have not (default in pytorch)
    @fixed_h: if False, it uses an ascending h schedule to speed up convergence in later stages
    '''
      
    J = len(models)-1
    if seeds is not None and len(models)!=len(seeds): 
        raise ValueError('len(seed) and number of models do not match!')
        

    all_out=[]
    all_im = []
    all_inter =[]
    if type(init_im) is tuple: ### If generating low resolution image. Unconditional synthesis from global prior
        print('-------------------- generating low pass image')
        if seeds['low'] is not None: 
            torch.manual_seed(seeds['low'])  
            
        low, inter,_,_ = one_scale_synthesis(model=models['low'], init_im=init_im , sig_0=sig_0['low'], 
                                         sig_L=sig_L['low'], h0=h0['low'], beta=beta['low'] , freq=freq['low'] 
                                         ,device=device, fixed_h = fixed_h['low'])
        all_inter.append(inter)   
        print('-------- im range: ', low.detach().min().item(), low.detach().max().item())
        im = rescale_image_range(low.detach(),.8 ) # rescale low pass to [0,1]
        im_max = im.max()
    else: 
        im = init_im 
        im_max = im.max()
        all_inter.append([])   
        
    all_out.append(im)
    all_im.append(im)
    
    
    print('--------', im.shape)
         
    for j in range(J-1,-1,-1): 
        print('-------------------- scale: ', j)
        if seeds[j] is not None: 
            torch.manual_seed(seeds[j])  
        else: 
            torch.random.seed()
        
        coeffs, inter,_,_ = one_scale_synthesis(models[j], im , sig_0=sig_0[j], sig_L=sig_L[j], h0=h0[j] ,
                                     beta=beta[j] , freq=freq[j] ,device=device, fixed_h = fixed_h[j] )
        all_out.append(coeffs.squeeze(0).detach())

        im = reconstruct( torch.cat([im,coeffs.detach()] ,dim = 0).unsqueeze(0),
                         device, 
                         orth_forward = orth_forward)
        
        
        im = im.squeeze(0)
        all_im.append(im)
        all_inter.append(inter)
        print('--------',im.shape)
        print('-------- im range: ', im.detach().min().item(), im.detach().max().item())
    

        
    return all_im, all_inter, all_out
