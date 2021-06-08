"""
Comparison between Spatial and Frequency Domain Filtering

Conduct the comparison between Spatial and Frequency Domain Filtering about the attached Rose image i 
via experiments of following procedure. 
h(x,y) below is the Sobel mask for horizontal edge detection. 

1) Conduct the Fourier Transform about the image i. Let its result be I(u,v). 
2) Conduct Fourier transform about the Sobel mask h(x,y). Let the result be H(u,v). 
3) Conduct the element-wise multiplication between I(u,v) and H(u,v). Let its result be I’(u,v). 
      I’(u,v)=I(u,v)H(u,v)
   Then, show the I’(u,v) in an image. 
4) Conduct the inverse Fourier Transform of I’(u,v). Let its result be i’(x,y). 
   Then, show the image  i’(x,y).
5) Conduct a mask processing of  h(x,y) on the Spatial domain about the original image i. 
   Let the resultant image be i’’(x,y).  
   And, make a comparison between the above two images, i’(x,y) and i’’(x,y).

                    h(x,y)=
                            1    2    1
                            0    0    0
                           -1   -2   -1
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

#%% read image

img = Image.open('Rose-BMP.bmp').convert('L')
i = np.array(img)

if i.shape[0]%2 == 0:
    i = i[:i.shape[0]-1,:]
if i.shape[1]%2 == 0:
    i = i[:,:i.shape[1]-1]

plt.figure()
plt.imshow(i, cmap='gray')
plt.show()

#%% fft

I = np.fft.fft2(i)
I = np.fft.fftshift(I)
magnitude_I = np.log(np.abs(I)+1)

plt.figure()
plt.imshow(magnitude_I, cmap='gray')
plt.show()

#%% sobel

h = np.array([[ 1, 2, 1],
              [ 0, 0, 0],
              [-1,-2,-1]])

#plt.figure()
#plt.imshow(abs(h), cmap='gray')
#plt.show()

h_p = np.pad(h, ((1000,1000),(1000,1000)), 'constant', constant_values=0)

H = np.fft.fft2(h_p)
H = np.fft.fftshift(H)
magnitude_H = np.log(np.abs(H)+1)

plt.figure()
plt.imshow(magnitude_H[490:1513,490:1513], cmap='gray')
plt.show()

#%% element-wise multiplication

def cal_D(c_row, c_col, r, c):
    s = (c_row-r)**2 + (c_col-c)**2
    return s**(1/2)

def filter_radius(fshift, rad, low=True):
    rows, cols = fshift.shape
    c_row, c_col = int(rows/2), int(cols/2)    # center
    
    filter_fshift = fshift.copy()
    
    for r in range(rows):
        for c in range(cols):
            if low:    # low-pass filter
                if cal_D(c_row, c_col, r, c) > rad:
                    filter_fshift[r,c] = 1
            else:      # high-pass filter
                if cal_D(c_row, c_col, r, c) < rad:
                    filter_fshift[r,c] = 1
    
    return filter_fshift

low_fshift = filter_radius(I, rad=50, low=True)

I_prime = low_fshift
plt.figure()
plt.imshow(np.log(abs(I_prime)+1), cmap='gray')
plt.show()

#%% inverse FFT

f_i_prime = np.fft.ifftshift(I_prime)
i_prime = np.fft.ifft2(f_i_prime)
i_prime = np.abs(i_prime)*1000000
i_prime[i_prime>255] = 255

plt.figure()
plt.imshow(np.log(i_prime+1), cmap='gray')
plt.show()

#%% Spatail domain filtering

def conv_2d(img, i_filter, stride):
    
    result_shape = tuple( np.int64( 
        (np.array(img.shape)-np.array(i_filter.shape))/stride+1 
        ) )
    
    result = np.zeros(result_shape)
    
    for h in range(0, result_shape[0], stride):
        for w in range(0, result_shape[1], stride):
            tmp = img[h:h+i_filter.shape[0],w:w+i_filter.shape[1]]*i_filter
            result[h,w] = np.abs(np.sum(tmp))
            
    result[result>255] = 255
    
    return result

result = conv_2d(i, h, stride=1)

plt.figure()
plt.imshow(result, cmap='gray')
plt.show()