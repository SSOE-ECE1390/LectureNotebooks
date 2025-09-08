import cv2
import numpy as np
from random import randint

"""
 This library implements tools to add different types of noise to 
 images using openCV

 Implements:
    image_noise = add_gaussian_noise(image,mean=0,std=25)
        Adds gaussian random noise to the image
    image_noise = add_salt_noise(image,fraction=0.005)
        Adds salt (max value) noise to the image
    image_noise = add_pepper_noise(image,fraction=0.005)
        Adds pepper (value=0) noise to the image
 
        
 """


def add_gaussian_noise(image, mean=0, std=25):
    # adds gaussian random noise
    # mean and std are for the distribution of the noise
    # returns copy of image with noise added

    noise = np.random.normal(mean, std, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image

def add_salt_noise(image,fraction=0.005):
    # adds salt (max value) noise
    # fraction is the / of pixels to contaminate
    # returns copy of image with noise added

    image2 = image.copy()
    number = np.int32(np.round(image.shape[0]*image.shape[1]*fraction))
    for idx in range(0,number):
        i=randint(0,image.shape[0]-1)
        j=randint(0,image.shape[1]-1)
        image2[i,j]=255
    return image2

def add_pepper_noise(image,fraction=0.005):
     # adds pepper (value=0) noise
    # fraction is the / of pixels to contaminate
    # returns copy of image with noise added

    image2 = image.copy()
    number = np.int32(np.round(image.shape[0]*image.shape[1]*fraction))
    for idx in range(0,number):
        i=randint(0,image.shape[0]-1)
        j=randint(0,image.shape[1]-1)
        image2[i,j]=0
    return image2