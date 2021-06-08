# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 16:05:02 2021

@author: jstar
"""

import numpy as np 
from matplotlib import pyplot as plt 
# simple averaging filter without scaling parameter 
mean_filter = np.ones((3,3)) 

# different edge detecting filters 
# scharr in x-direction 
scharr = np.array([[-3, 0, 3], [-10,0,10], [-3, 0, 3]]) 
# sobel in x direction 
sobel_x= np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) 
# sobel in y direction 
sobel_y= np.array([[-1,-2,-1], [0, 0, 0], [1, 2, 1]]) 
# laplacian 
laplacian=np.array([[0, 1, 0], [1,-4, 1], [0, 1, 0]]) 

filters = [mean_filter, laplacian, sobel_x, sobel_y, scharr] 
filter_name = ['mean_filter', 'laplacian', 'sobel_x', 'sobel_y', 'scharr_x'] 
fft_filters = [np.fft.fft2(x) for x in filters] 
fft_shift = [np.fft.fftshift(y) for y in fft_filters]
mag_spectrum = [np.log(np.abs(z)+1) for z in fft_shift] 

for i in range(5): 
    plt.subplot(2,3,i+1),plt.imshow(mag_spectrum[i],cmap = 'gray') 
    plt.title(filter_name[i]), plt.xticks([]), plt.yticks([]) 
    
plt.show()

