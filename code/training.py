import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import torch.nn as nn
import os
import time
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import argparse

from network import *
from dataloader_func import weights_init_kaiming, add_noise_torch, add_noise_torch_range,rescale_image_range
from quality_metrics_func import batch_ave_psnr_torch
from plotting_func import plot_loss,plot_psnr, plot_denoised_range

################################################# training #################################################
def train_entire_net(train_images, test_images,  args):
    start_time_total = time.time()

    trainloader = DataLoader(dataset=train_images, batch_size=args.batch_size, shuffle=True)
    testloader = DataLoader(dataset=test_images, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = BF_CNN_RF(args)

    model.apply(weights_init_kaiming)

    if torch.cuda.is_available():
        print('[ Using CUDA ]')
        model = nn.DataParallel(model).cuda()
        #model = model.cuda()


    print('number of parameters is ' , sum(p.numel() for p in model.parameters() if p.requires_grad))
    criterion = nn.MSELoss(reduction='sum')
    optimizer = Adam(filter(lambda p: p.requires_grad,model.parameters()), lr = args.lr)

    loss_list = []
    loss_list_test = []

    loss_list_epoch = []
    loss_list_test_epoch = []

    psnr_list_epoch = []
    psnr_list_test_epoch = []

    ############################## Train ##############################
    for h in range(args.num_epochs):
        print('epoch ', h )
        if h >= args.lr_freq and h%args.lr_freq==0:
            for param_group in optimizer.param_groups:
                args.lr = args.lr/2
                param_group["lr"] = args.lr
        #loop over images
        loss_sum = 0
        psnr_sum = 0
        for i, batch in enumerate(trainloader, 0):
            model.train()

            clean = batch.to(device)
            if args.rescale:
                clean = clean * torch.rand(size=(batch.size()[0], 1,1,1),device = device) #resize intensity of clean images
            optimizer.zero_grad()
            noisy , noise = add_noise_torch(clean, args.noise_level_range, 'B', device, args.quadratic_noise, args.coarse)
            output = model(noisy)
            if args.skip:
                target = noise
                if args.coarse:
                    denoised = noisy - output
                else:
                    denoised = noisy[:,1::] - output #C=3

            else:
                if args.coarse:
                    target = clean
                else:
                    target = clean[1::] #C=3

                denoised = output

            loss = criterion(output, target)/ (clean.size()[0])

            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            loss_sum += loss.item()
            if args.coarse:
                psnr_sum += batch_ave_psnr_torch(clean, denoised ,1.).item()
            else:
                psnr_sum += batch_ave_psnr_torch(clean[:,1::], denoised ,1.).item()


        loss_list_epoch.append(loss_sum/((i+1) ))
        psnr_list_epoch.append(psnr_sum/(i+1))

        print('model train loss ', loss_list_epoch[-1] )
        print('model train psnr ', psnr_list_epoch[-1] )


        ############################## Test ##############################
        loss_sum = 0
        psnr_sum = 0
        for i, batch in enumerate(testloader, 0):
            model.eval()
            clean = batch.to(device)
            if args.rescale:
                clean = clean * torch.rand(size=(batch.size()[0], 1,1,1),device = device) #added late, after I ran the job :(

            with torch.no_grad():
                noisy , noise = add_noise_torch(clean, args.noise_level_range, 'B', device, args.quadratic_noise, args.coarse) # add noise only to coeffs
                output = model(noisy)
                if args.skip:
                    target = noise
                    if args.coarse:
                        denoised = noisy - output
                    else:
                        denoised = noisy[:,1::] - output
                else:
                    if args.coarse:
                        target = clean
                    else:
                        target = clean[1::] #C=3

                    denoised = output

                loss = criterion(output, target)/ (clean.size()[0])
                loss_list_test.append(loss.item())
                loss_sum+= loss.item()
                if args.coarse:
                    psnr_sum += batch_ave_psnr_torch(clean, denoised ,1.).item()
                else:
                    psnr_sum += batch_ave_psnr_torch(clean[:,1::], denoised ,1.).item()

        loss_list_test_epoch.append(loss_sum/((i+1) ))
        psnr_list_test_epoch.append(psnr_sum/(i+1))
        print('model test loss ', loss_list_test_epoch[-1] )
        print('model test psnr ', psnr_list_test_epoch[-1] )


        ########################## save and plot ###########################
        plot_loss(loss_list_epoch, loss_list_test_epoch, args.dir_name+'/loss_epoch.png')
        plot_psnr(psnr_list_epoch , psnr_list_test_epoch ,args.dir_name+'/psnr_epoch.png' )

        noise_range = torch.logspace(0,2.5,4, device=device).reshape(4,1,1,1)
        with torch.no_grad():
            noisy , noise = add_noise_torch_range(clean[0], noise_range, device=device,coarse=args.coarse)
            output = model(noisy)
            if args.skip:
                if args.coarse:
                    denoised = noisy - output
                else:
                    denoised = noisy[:,1::] - output
            else:
                denoised = output
        if args.coarse:
            plot_denoised_range(clean[0], noisy, denoised, noise_range, args.dir_name, 1)

        else:
            plot_denoised_range(clean[0 ,1:2], noisy[:,1:2], denoised[:,0:1], noise_range, args.dir_name, 1)


        torch.save(model.state_dict(), args.dir_name  + '/model.pt')



    print("--- %s seconds ---" % (time.time() - start_time_total))


    ########### calculate average PSNR for Testset across a wide range
    psnr_range = {}
    for sigma in range(0,270, 20):
        psnr_range[sigma] = []
        print('calculating psnr for sigma  : ', sigma)
        psnr = 0
        for i, batch in enumerate(testloader,0):
            model.eval()
            clean = batch.to(device)
            noisy , noise = add_noise_torch(clean, sigma, 'S', device, args.quadratic_noise, args.coarse)
            with torch.no_grad():
                output = model(noisy)
                if args.skip:
                    target = noise
                    if args.coarse:
                        denoised = noisy - output
                    else:
                        denoised = noisy[:,1::] - output
                else:
                    if args.coarse:
                        target = clean
                    else:
                        target = clean[1::] #C=3

                    denoised = output

            loss = criterion(denoised, target)/ (clean.size()[0])
            if args.coarse:
                psnr += batch_ave_psnr_torch(clean, denoised ,1.).item()
            else:
                psnr += batch_ave_psnr_torch(clean[:,1::], denoised ,1.).item()

        psnr_range[sigma].append(psnr/(i+1))
        torch.save(psnr_range, args.dir_name + '/psnr_test_list.pt')
    print('psnr for Test set over the range of 0 to 260: ', psnr_range)

    return model
################################################# main #################################################


def main():
    parser = argparse.ArgumentParser(description='training BFCNN with chosen RF')
    ### architecture variables
    parser.add_argument('--kernel_size', default= 3)
    parser.add_argument('--padding', default= 1)
    parser.add_argument('--num_kernels', default= 64)
    parser.add_argument('--num_layers', default= 21)
    parser.add_argument('--num_channels', default= 1)#set 1 for grayscale and 3 for color
    parser.add_argument('--skip', default= True)
    
    ### architecture variables to be set
    parser.add_argument('--coarse', help = 'denoiser for coarse or fine coefficients')
    parser.add_argument('--RF',type=int , help='receptive field of the network') # only values accepted in this set {5,9,13, 23, 43}


    ### optimization variables
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--lr_freq', type=int, default=50)

    
    ### dataset variables 
    parser.add_argument('--noise_level_range', default= [0,255])
    parser.add_argument('--quadratic_noise', default=True) # noise std will be drawn from 1/sqrt(std) distribution instead of uniform distribution
    parser.add_argument('--SLURM_ARRAY_TASK_ID',type=int ) #this determines j or scale     
    parser.add_argument('--rescale', default=True) # rescale intensities. Don't rescale for conditional denoisers. 
    
    ### directory-related variables 
    parser.add_argument('--dir_name', default= '/mnt/home/zkadkhodaie/ceph/multi_scale_synthesis/denoisers/celebAtwo_gray/noise_range_')
    parser.add_argument('--data_path', default= '/mnt/home/zkadkhodaie/ceph/datasets/img_align_celeba_320x320_coeffs_num_scales_3/') #data_path must end with a / because this string is used to infer coarse scale in the pyramid 
    parser.add_argument('--debug', default=False)

    args = parser.parse_args()

    if args.coarse: 
        j =  int(args.data_path.split('/')[-2].split('_')[-1]) - 1 #depth of pyramid - 1 (because j starts from 0)
        args.dir_name = args.dir_name + str(args.noise_level_range[0])+'to'+ str(args.noise_level_range[1]) + str(args.RF)+'x'+str(args.RF) + '_low'
        args.rescale=True
        
    else:
        j = args.SLURM_ARRAY_TASK_ID 
        args.dir_name = args.dir_name + str(args.noise_level_range[0])+'to'+ str(args.noise_level_range[1]) + str(args.RF)+'x'+str(args.RF) + '_scale_'+ str(j)
        args.rescale=False

    #compensate for smaller image size (1/4) at each higher scale (j=0,500, j=1,500*4, j = 2,500*16)
    args.num_epochs = args.num_epochs * (4**j)
    args.lr_freq = args.lr_freq * (4**j)

    ########## load data ##########
    train_path = args.data_path + 'train_scale'+str(j)+'.pt'
    test_path = args.data_path + 'test_scale'+str(j)+'.pt'

    if args.debug:
        train_coeffs = torch.load(train_path)[0:args.batch_size]
        test_coeffs = torch.load(test_path)[0:args.batch_size]
    else:
        train_coeffs = torch.load(train_path) 
        test_coeffs = torch.load(test_path)

    # correct for the missing normalization factor in the default pywavelet Haar decompose function
    train_coeffs = train_coeffs/(2**(j+1))
    test_coeffs = test_coeffs/(2**(j+1))

    print('train: ', train_coeffs.size(), 'test: ', test_coeffs.size() )
    print('train low mean: ', train_coeffs[:,0].mean().item(), 'test low mean: ', test_coeffs[:,0].mean().item() )
    print('train high mean: ', train_coeffs[:,1::].mean().item(), 'test high mean: ', test_coeffs[:,1::].mean().item() )

    print('train low std: ', train_coeffs[:,0].std().item(), 'test low std: ', test_coeffs[:,0].std().item() )
    print('train high std: ', train_coeffs[:,1::].std().item(), 'test high std: ', test_coeffs[:,1::].std().item() )

    print('train low max: ', abs(train_coeffs[:,0]).max().item(), 'test high max: ', abs(test_coeffs[:,0]).max().item() )
    print('train high max: ', abs(train_coeffs[:,1::]).max().item(), 'test high max: ', abs(test_coeffs[:,1::]).max().item() )

    if args.coarse:
        train_coeffs = train_coeffs[:,0:1]
        test_coeffs = test_coeffs[:,0:1]

    # adjust max noise level to max signal value
    if args.coarse is False:
        args.noise_level_range[1] = int(args.noise_level_range[1] * abs(train_coeffs[:,1::]).max().item())

    print('true noise range: ', args.noise_level_range )

    if not os.path.exists(args.dir_name):
        os.makedirs(args.dir_name)

    train_entire_net(train_coeffs, test_coeffs, args)


if __name__ == "__main__" :
    main()
