import numpy as np
import os
import time
import torch
from dataloader_func import load_CelebA_HQ_dataset, prep_celeba_for_wavelet
from wavelet_func import multi_scale_decompose
import argparse
import pywt

def main():
    start_time_total = time.time()

    parser = argparse.ArgumentParser(description='training multiple models')
    parser.add_argument('--dir_name', default= '/mnt/home/zkadkhodaie/ceph/datasets/img_align_celeba_320x320_coeffs_num_scales_')
    parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument('--k', default= None, help='number of replica of each image with intensity changed randomly')
    parser.add_argument('--J', default=3, help='number of scales')
    parser.add_argument('--wavelet', default='db1')
    parser.add_argument('--boundary_mode', default='symmetric' )

    parser.add_argument('--debug', default=False)
    args = parser.parse_args()


    print('coeffs ready ')
    args.dir_name = args.dir_name + str(args.J)
    if not os.path.exists(args.dir_name):
        os.makedirs(args.dir_name)


    print('loading images ... ')
    train_folder_path = '/mnt/home/zkadkhodaie/ceph/datasets/img_align_celeba_512x512/train/'
    test_folder_path = '/mnt/home/zkadkhodaie/ceph/datasets/img_align_celeba_512x512/test/'

    if args.debug:
        all_images= load_CelebA_HQ_dataset( train_folder_path, test_folder_path, s=.625, n=args.batch_size)
    else:
        all_images= load_CelebA_HQ_dataset( train_folder_path, test_folder_path, s=.625)

    print('images loaded')
    train_images, test_images = prep_celeba_for_wavelet(all_images)
    print('train: ', train_images.shape, 'test: ', test_images.shape )
    print('train mean: ', train_images.mean().item(), 'test mean: ', test_images.mean().item() )
    print('images preped')

    
    low = train_images
    for j in range(args.J):
        low, high = pywt.dwt2(low,wavelet= args.wavelet, mode=args.boundary_mode)
        coeffs = np.concatenate((low,) + high, axis=-3)  # (*, 4, L/2, L/2)
        coeffs = torch.from_numpy(coeffs).to(dtype=torch.float32)
        torch.save(coeffs, args.dir_name + '/train_scale'+str(j)+'.pt')
        print('train shape at ', j , coeffs.shape)


    low = test_images
    for j in range(args.J):
        low, high = pywt.dwt2(low,wavelet= args.wavelet, mode=args.boundary_mode)
        coeffs = np.concatenate((low,) + high, axis=-3)  # (*, 4, L/2, L/2)
        coeffs = torch.from_numpy(coeffs).to(dtype=torch.float32)
        torch.save(coeffs, args.dir_name + '/test_scale'+str(j)+'.pt')
        print('test shape at ', j , coeffs.shape)


    print('coeffs ready ')

    print("--- %s seconds ---" % (time.time() - start_time_total))



if __name__ == "__main__" :
    main()

