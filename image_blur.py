import numpy as np
from skimage.transform import rescale
import scipy
import scipy.ndimage
import cv2
import os

FILE_NAME = 'teste_img.jpg'
PATH = os.path.join('assets', FILE_NAME)

img = cv2.imread(PATH)
cv2.imshow('Original', img)

blue  = rescale(img[:,:,0], 0.5)
green = rescale(img[:,:,1], 0.5)
red   = rescale(img[:,:,2], 0.5)
img_scaled = np.stack([blue, green, red], axis=2)

# Box Blur 3x3
box_blur = (1/9) * np.ones((3,3))

# Box Blur 10x10
box_blur_big1 = (1/100) * np.ones((10,10))
box_blur_big = np.ones((10,10))
box_blur_big = box_blur_big / box_blur_big.sum()

# Gaussian Blur 3x3
gaussian_blur = (1/16) * np.array([[1,2,1],
                                   [2,4,2], 
                                   [1,2,1]])

# Gaussian Blur 5x5
gaussian_blur_5x5 = (1/256) * np.array([[1,4,6,4,1],
                                        [4,16,24,16,4],
                                        [6,24,36,24,6],
                                        [4,16,24,16,4],
                                        [1,4,6,4,1]])

kernels = [box_blur, box_blur_big, gaussian_blur, gaussian_blur_5x5]
kernels_names = ['Box Blur', 'Box Blur Big', 'Gaussian Blur 3x3', 'Gaussian Blur 5x5']
kernels_dict = {kernels_names[i]: kernels[i] for i in range(len(kernels))}

for kernel_name, kernel in kernels_dict.items():  
    conv_im1 = scipy.ndimage.convolve(img, np.atleast_3d(kernel), mode='nearest')
    cv2.imshow(kernel_name, conv_im1)

cv2.waitKey(0)