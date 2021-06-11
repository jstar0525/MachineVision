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

                     h(x,y) =
                            1    2    1
                            0    0    0
                           -1   -2   -1
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# read image
img = Image.open('Rose-BMP.bmp').convert('L')
i = np.array(img)

# make a odd shape image
if i.shape[0]%2 == 0:
    i = i[:i.shape[0]-1,:]
if i.shape[1]%2 == 0:
    i = i[:,:i.shape[1]-1]

plt.imshow(i, cmap='gray')
plt.show()

#%% fft

I = np.fft.fft2(i)
I_s = np.fft.fftshift(I)

plt.imshow(np.log(np.abs(I_s)+1), cmap='gray')
plt.show()

#%% sobel

h = np.array([[ 1, 2, 1],
              [ 0, 0, 0],
              [-1,-2,-1]])

def fft_filter(h, filter_size=1023, crop_size=1023):

    if crop_size >= filter_size:
        filter_size = crop_size

    p = int((filter_size-h.shape[0])/2)
    h_p = np.pad(h, ((p,p),(p,p)), 'constant', constant_values=0)
    
    H = np.fft.fft2(h_p)
    
    if crop_size >= filter_size:
        return H
    else:
        start = int((filter_size-crop_size)/2)
        last = int(start+crop_size)
        
        return H[start:last,start:last]
    
H = fft_filter(h, filter_size=1023, crop_size=1023)
H_s = np.fft.fftshift(H)

plt.imshow(np.log(np.abs(H_s)+1), cmap='gray')
plt.show()

#%% element-wise multiplication

I_prime = I_s*np.log(np.abs(H_s)+1)

plt.imshow(np.log(abs(I_prime)+1), cmap='gray')
plt.show()

#%% inverse FFT

i_prime = np.fft.ifftshift(I_prime)
i_prime = np.fft.ifft2(i_prime)
i_prime[i_prime>255] = 255

plt.imshow(np.abs(i_prime), cmap='gray')
plt.show()

#%% Spatail domain filtering

def conv(img, i_filter, stride):
    
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

result = conv(i, h, stride=1)

plt.imshow(result, cmap='gray')
plt.show()

