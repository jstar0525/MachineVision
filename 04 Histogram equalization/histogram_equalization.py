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

def contrast_stretching(img, newMin=0, newMax=75):
    
    ch_img = img.copy()
    
    ch_img[ch_img<=newMin] = newMin
    ch_img[ch_img>=newMax] = newMax
    
    result = (ch_img - newMin) / (newMax - newMin) * 255
    
    return result

def histogram_equalization(img):
    
    # 1
    (N, M) = img.shape
  
    G = 256 # gray levels
    H = np.zeros(G) # initialize an array Histogram
    
    # 2    
    for g in img.ravel():
        H[g] += 1
        
    g_min = np.min(np.nonzero(H))
    
    # 3
    H_c = np.zeros_like(H) # cumulative image histogram
    H_c[0] = H[0]
    for g in range(1,G):
        H_c[g] = H_c[g-1] + H[g]
    
    H_min = H_c[g_min]
    
    # 4
    T = np.round( (H_c - H_min) / (M*N - H_min) * (G-1) )
    
    # 5
    result = np.zeros_like(img)
    for n in range(N):
        for m in range(M):
            result[n,m] = T[img[n,m]]
        
    return result

# image 1
img_1 = Image.open('paperPhoto20210402174743810.bmp')

np_img_1 = np.array(img_1)[:,:,0]
show(np_img_1, max_ylim=12500)
   
stretched_img_1 = contrast_stretching(np_img_1, newMin=0, newMax=75)
show(stretched_img_1, max_ylim=12500)

eq_img_1 = histogram_equalization(np_img_1)
show(eq_img_1, max_ylim=12500)

# image 2
img_2 = Image.open('paperPhoto20210402174743817.bmp')

np_img_2 = np.array(img_2)[:,:,0]
show(np_img_2, max_ylim=4500)

stretched_img_2 = contrast_stretching(np_img_2, newMin=100, newMax=255)
show(stretched_img_2, max_ylim=3500)

eq_img_2 = histogram_equalization(np_img_2)
show(eq_img_2, max_ylim=3500)

#%%

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

med_img_1 = median_filter(np_img_1)
show(med_img_1, max_ylim=12500)

med_img_2 = median_filter(np_img_2)
show(med_img_2, max_ylim=4500)

#%% 

axes = plt.axes()
axes.set_ylim([0, 3500])
plt.bar(range(len(H)), H)
plt.show()

axes = plt.axes()
axes.set_ylim([0, 3500])
plt.bar(range(len(H_c)), H_c)
plt.show()

axes = plt.axes()
axes.set_ylim([0, 3500])
plt.plot(range(len(T)), T)
plt.show()
