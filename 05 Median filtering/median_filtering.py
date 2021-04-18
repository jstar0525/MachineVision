"""
Perform simulations about the attached two images as following. 

    Apply median filtering about two images. 
    Then, compare the results before and after the filtering.
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

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

def median_filter(img, filter_size=(3, 3), stride=1):
    
    img_shape = np.shape(img)

    result_shape = tuple( np.int64( 
        (np.array(img_shape)-np.array(filter_size))/stride+1 
        ) )

    result = np.zeros(result_shape)

    for h in range(0, result_shape[0], stride):
        for w in range(0, result_shape[1], stride):
            tmp = img[h:h+filter_size[0],w:w+filter_size[1]]
            tmp = np.sort(tmp.ravel())
            result[h,w] = tmp[int(filter_size[0]*filter_size[1]/2)]
    
    return result

# image 1
img_1 = Image.open('paperPhoto20210402174743810.bmp').convert('L')
npimg_1 = np.array(img_1)
show(npimg_1, max_ylim=12500)

med_img_1 = median_filter(npimg_1)
show(med_img_1, max_ylim=12500)

med_img_1 = median_filter(npimg_1, (5,5))
show(med_img_1, max_ylim=12500)

# image 2
img_2 = Image.open('paperPhoto20210402174743817.bmp').convert('L')
npimg_2 = np.array(img_2)
show(npimg_2, max_ylim=4500)

med_img_2 = median_filter(npimg_2)
show(med_img_2, max_ylim=4500)

med_img_2 = median_filter(npimg_2, (5,5))
show(med_img_2, max_ylim=4500)
