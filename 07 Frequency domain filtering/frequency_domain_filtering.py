"""
Perform following filtering using the radius of 50 pixels.

    a) Low Pass Filtering 
        : Transform the image into Frequency Domain
        : Manipulate the frequency components
        : Transform the manipulated freq. Components to the spatial domain. 
    b) High Pass Filtering. 
    c) Compare the above two  processing results.
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def cal_D(c_row, c_col, r, c):
    s = (c_row-r)**2 + (c_col-c)**2
    return s**(1/2)

def filter_radius(gray_img, fshift, rad, low=True):
    rows, cols = gray_img.shape
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

# read image
img = Image.open('Rose-BMP.bmp').convert('L')
gray_img = np.array(img)

plt.figure()
plt.imshow(gray_img, cmap='gray')
plt.show()

# fft
f = np.fft.fft2(gray_img)
magnitude_f = 20*np.log(np.abs(f))

plt.figure()
plt.imshow(magnitude_f, cmap='gray')
plt.show()

# fftshift
fshift = np.fft.fftshift(f)
magnitude_fshift = 20*np.log(np.abs(fshift))

plt.figure()
plt.imshow(magnitude_fshift, cmap='gray')
plt.show()

# low-pass filter
low_fshift = filter_radius(gray_img, fshift, rad=50, low=True)
low_pass_magnitude = 20*np.log(np.abs(low_fshift))

plt.figure()
plt.imshow(low_pass_magnitude, cmap='gray')
plt.show()

# low-pass filter inverse fft
low_ishift = np.fft.ifftshift(low_fshift)
low_img = np.fft.ifft2(low_ishift)
low_img = np.abs(low_img)

plt.figure()
plt.imshow(low_img, cmap='gray')
plt.show()

# high-pass filter
high_fshift = filter_radius(gray_img, fshift, rad=50, low=False)
high_pass_magnitude = 20*np.log(np.abs(high_fshift))

plt.figure()
plt.imshow(high_pass_magnitude, cmap='gray')
plt.show()

# high-pass filter inverse fft
high_ishift = np.fft.ifftshift(high_fshift)
high_img = np.fft.ifft2(high_ishift)
high_img = np.abs(high_img)

plt.figure()
plt.imshow(high_img, cmap='gray')
plt.show()