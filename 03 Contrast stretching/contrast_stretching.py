"""
Perform simulations about the attached two images as following. 

    Enhance the images using Contrast stretching 
"""


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def contrast_stretching(img, newMin=0, newMax=75):
    
    ch_img = img.copy()
    
    ch_img[ch_img<=newMin] = newMin
    ch_img[ch_img>=newMax] = newMax
    
    result = (ch_img - newMin) / (newMax - newMin) * 255
    
    return result

def show(img, max_ylim=12500):
    
    # show image
    plt.imshow(img, cmap='gray')
    plt.show()
    
    # show histogram
    if max_ylim != 'none':
        axes = plt.axes()
        axes.set_ylim([0, max_ylim])
    plt.hist(img.ravel(), bins=256, range=[0,256])
    plt.show()

# image 1
img_1 = Image.open('paperPhoto20210402174743810.bmp').convert('L')

npimg_1 = np.array(img_1)
show(npimg_1, max_ylim=12500)
   
stimg_1 = contrast_stretching(npimg_1, newMin=0, newMax=75)
show(stimg_1, max_ylim=12500)


# image 2
img_2 = Image.open('paperPhoto20210402174743817.bmp').convert('L')

npimg_2 = np.array(img_2)
show(npimg_2, max_ylim=4500)

stimg_2 = contrast_stretching(npimg_2, newMin=100, newMax=255)
show(stimg_2, max_ylim=4500)

