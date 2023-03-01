import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import torch.nn as nn
import torch
import os
from PIL import Image
import gzip

######################################################################################
################################ data loaders ################################
######################################################################################



def load_CelebA_dataset( train_folder_path, test_folder_path, s=1,n=-1):
    '''
    return images in a dict of two arrays. all images horizontal. Output arrays are in dims: B, H, W, C
    This function does not change image range.
    '''
    image_dict = {};
    train_names = os.listdir(train_folder_path)[0:n]
    test_names = os.listdir(test_folder_path)[0:n]

    # read and prep train images
    train_images = []
    for file_name in train_names:
        im = plt.imread(train_folder_path + file_name)
        im = im[29:-29, 9:-9] #crop to 160x160
        if s != 1:
            im =  resize_image(im, s)
        train_images.append(im);

    # read and prep test images
    test_images = []
    for file_name in test_names:
        im = plt.imread(test_folder_path + file_name)
        im = im[29:-29, 9:-9]
        if s != 1:
            im =  resize_image(im, s)
        test_images.append(im);


    image_dict['train'] = np.array(train_images)
    image_dict['test'] = np.array(test_images)

    return image_dict

def load_set14( path):
    test_names = os.listdir(path)
    # read and prep test images
    test_images = []
    for file_name in test_names:
        image = plt.imread(path + file_name)
        test_images.append(image );
    return np.array(test_images)


def load_CelebA_HQ_dataset( train_folder_path, test_folder_path, s=1,n=-1):
    '''
    return images in a dict of two arrays. all images horizontal. Output arrays are in dims: B, H, W, C
    This function does not change image range.
    '''
    image_dict = {};
    train_names = os.listdir(train_folder_path)[0:n]
    test_names = os.listdir(test_folder_path)[0:n]

    # read and prep train images
    train_images = []
    for file_name in train_names:
        im = plt.imread(train_folder_path + file_name)
        if s != 1:
            im =  resize_image(im, s)
        train_images.append(im);

    # read and prep test images
    test_images = []
    for file_name in test_names:
        im = plt.imread(test_folder_path + file_name)
        if s != 1:
            im =  resize_image(im, s)
        test_images.append(im);


    image_dict['train'] = np.array(train_images)
    image_dict['test'] = np.array(test_images)

    return image_dict





######################################################################################
################################ image pre-processing ################################
######################################################################################
def prep_celeba(all_images,k=None, mean_zero=False):
    '''
    @k: number of replica of each image with a different intensity
    '''
    all_images_tensor = {}

    train_images = int_to_float(all_images['train'])
    train_images = rgb_to_gray(train_images) # convert to grayscale
    if mean_zero:
        #train_images = remove_mean(train_images)
        train_images = train_images - train_images.mean()
    if k is not None:
        train_images = change_intensity_dataset(train_images,k)
    train_images = torch.FloatTensor(train_images).permute(0,3,1,2).contiguous() # (B, C, H, W)

    test_images = int_to_float(all_images['test'])
    test_images = rgb_to_gray(test_images)
    if mean_zero:
        #test_images = remove_mean(test_images)
        test_images = test_images - test_images.mean()
    if k is not None:
        test_images = change_intensity_dataset(test_images ,k )
    test_images = torch.FloatTensor(test_images).permute(0,3,1,2).contiguous() # (B, C, H, W)

    return train_images, test_images



def resize_image(im, s):
    image_pil = Image.fromarray(im) # data type needed is uint8
    newsize = (int(image_pil.size[0] * s), int(image_pil.size[1] * s))
    image_pil_resize = image_pil.resize(newsize, resample=Image.BICUBIC)
    image_re = np.array(image_pil_resize)
    return image_re

def float_to_int(X):
    if X.dtype != 'uint8':
        return X
    else:
        return (X*255).astype('uint8')


def int_to_float(X):
    if X.dtype == 'uint8':
        return (X/255).astype('float32')
    else:
        return X

def rgb_to_gray(data):
    n, h, w, c = data.shape
    return data.mean(3).reshape(n,h,w,1)

def change_intensity(  im , k):
    temp = np.zeros((k,im.shape[0], im.shape[1], im.shape[2])).astype('float32')
    for i in range(k):
        temp[i] = np.random.rand(1).astype('float32') * im
    return temp

def change_intensity_dataset(dataset, k):
    temp = []
    for im in dataset:
        temp.append(change_intensity(im,k))
    return np.concatenate(temp)

def rescale_image_range(im, max_I):

    return ((im - im.min()) * (1/(im.max() - im.min()) * max_I))

def rescale_image(im):
    if type(im) == torch.Tensor:
        im = im.numpy()
    return ((im - im.min()) * (1/(im.max() - im.min()) * 255)).astype('uint8')


def remove_mean(images):
    '''
    remove mean of images in a numpy batch of size N,W,H,1
    '''
    return images - images.mean(axis=(1,2)).reshape(-1,1,1,1)

def patch_generator(all_images, patch_size, stride):
    '''images: a 4D tensor of image: B, C, H, W
    patch_size: a tuple indicating the size of patches
    stride: a tuple indicating the size of the strides
    '''
    im_height = all_images.shape[2]
    im_width = all_images.shape[3]

    h = int(im_height/stride[0]) * stride[0]
    w = int(im_width/stride[1]) * stride[1]

    all_patches = []
    for x in range(0,h- patch_size[0] + 1, stride[0]):
        for y in range(0,w - patch_size[1] + 1, stride[1]):
            patch = all_images[:,:, x:x+patch_size[0] , y:y+patch_size[1]]
            all_patches.append(patch)

    return torch.cat(all_patches)



def patch_generator_np(all_images, patch_size, stride):
    '''images: a 4D numpy of image: B, C, H, W
    patch_size: a tuple indicating the size of patches
    stride: a tuple indicating the size of the strides
    '''
    im_height = all_images.shape[2]
    im_width = all_images.shape[3]

    h = int(im_height/stride[0]) * stride[0]
    w = int(im_width/stride[1]) * stride[1]

    all_patches = []
    for x in range(0,h- patch_size[0] + 1, stride[0]):
        for y in range(0,w - patch_size[1] + 1, stride[1]):
            patch = all_images[:,:, x:x+patch_size[0] , y:y+patch_size[1]]
            all_patches.append(patch)

    return np.concatenate(all_patches)





def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    #elif classname.find('BatchNorm') != -1:
    #    m.weight.data.normal_(mean=0, std=sqrt(2./9./64.)).clamp_(-0.025,0.025)

    #    nn.init.constant(m.bias.data, 0.0)



###########################################################################
################################ wavelet coeffs ###########################
###########################################################################

def prep_celeba_for_wavelet(all_images, k=None):
    '''returns numpy arrays'''
    all_images_tensor = {}

    train_images = int_to_float(all_images['train'])
    train_images = rgb_to_gray(train_images) # convert to grayscale
    if k is not None:
        train_images = change_intensity_dataset(train_images,k)
    
    train_images = train_images.transpose(0,3,1,2)
    test_images = int_to_float(all_images['test'])
    test_images = rgb_to_gray(test_images)
    if k is not None:
        test_images = change_intensity_dataset(test_images ,k )
    test_images = test_images.transpose(0,3,1,2) 
    return train_images, test_images

###########################################################################
################################ add noise ################################
###########################################################################


def add_noise_torch(all_patches, noise_level, mode='B', device=None, quadratic=False, coarse=True):
    '''
    For images and wave coeffs 
    Gets images in the form of torch  tensors of size (B, C, H, W)
    '''
    N, C, H, W = all_patches.size()


    #for blind denoising
    if mode == 'B':
        std = torch.randint(low = noise_level[0], high=noise_level[1] ,
                            size = (N,1,1,1) ,device = device )/255

        if quadratic is True:
            std = std ** 2
    #for specific noise
    else:
        std = torch.ones(N,1,1,1,  device = device) * noise_level/255

    if coarse:
        noise_samples = torch.randn(size = all_patches.size() , device = device) * std
        noisy = noise_samples+ all_patches
    else:
        noise_samples = torch.randn(size = (N,3,H,W) , device = device) * std
        noisy = torch.cat((all_patches[:, 0:1, :,:], all_patches[:, 1::, :,:] + noise_samples), dim=1)

    return noisy, noise_samples

def add_noise_torch_range(im, noise_range, device=None, coarse=True):
    '''
    For images and wave coeffs
    Gets image- torch  tensors of size (1, H, W)
    @noise_rangel: sigmas for images in range 0-255 -tensor
    '''
    images = torch.stack([im]*noise_range.shape[0])
    noise_range = noise_range/255
    noise_samples = torch.randn(size = images.size() , device = device) * noise_range

    if coarse:
        noisy = images + noise_samples
    else:
        noise_samples = noise_samples[:,0:3]
        noisy = torch.cat((images[:, 0:1, :,:], images[:, 1::, :,:] + noise_samples), dim=1)

    return noisy , noise_samples


def add_noise_coeffs_torch_range(im, noise_range, device=None):
    '''
    Gets image- torch  tensors of size (1, H, W)
    @noise_rangel: sigmas for images in range 0-255 -tensor
    '''
    images = torch.stack([im]*noise_range.shape[0])
    noise_range = noise_range/255
    noise_samples = torch.randn(size = images.size() , device = device) * noise_range
    noise_samples = noise_samples[:,0:3]
    noisy = torch.cat((images[:, 0:1, :,:], images[:, 1::, :,:] + noise_samples), dim=1)
    return noisy , noise_samples



def add_noise_coeffs_torch(all_patches, noise_level, mode='B', device=None, quadratic=False):
    '''
    Gets images in the form of torch  tensors of size (B, C, H, W)
    '''
    N, C, H, W = all_patches.size()
    #if (mode == 'S' and type(noise_level) != int):
    #    raise TypeError('noise level needs to be in range 0 to 255')
    #elif(mode == 'B' and type(noise_level[0]) != int):
    #    raise TypeError('noise level needs to be in range 0 to 255')

    #if all_patches.dtype == torch.uint8:
    #    raise TypeError('Data type required is float32')

    #for blind denoising
    if mode == 'B':
        std = torch.randint(low = noise_level[0], high=noise_level[1] ,
                            size = (N,1,1,1) ,device = device )/255

        if quadratic is True:
            std = std ** 2
    #for specific noise
    else:
        std = torch.ones(N,1,1,1,  device = device) * noise_level/255

    noise_samples = torch.randn(size = (N,3,H,W) , device = device) * std
    noisy = torch.cat((all_patches[:, 0:1, :,:], all_patches[:, 1::, :,:] + noise_samples), dim=1)

    return noisy, noise_samples





