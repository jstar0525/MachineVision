"""
Perform simulations about the attached two images as following. 

    Enhance the images using Histogram equalization method 
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

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
        
    return result, T
    
def show_Hc(img, max_ylim=12500):
    
    # show image
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.show()
    
    G = 256 # gray levels
    H = np.zeros(G)
    
    for g in img.ravel():
        H[g] += 1
    
    H_c = np.zeros_like(H) # cumulative image histogram
    H_c[0] = H[0]
    for g in range(1,G):
        H_c[g] = H_c[g-1] + H[g]
        
    # show histogram
    fig, ax1 = plt.subplots()
    if max_ylim != 'none':
        ax1.set_ylim([0, max_ylim])
    ax1.set_ylabel('histogram', color='C0')
    ax1.hist(img.ravel(), bins=256, range=[0,256], color='C0')
    ax1.tick_params(axis='y', labelcolor='C0')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('cumulative histogram', color='C1')
    ax2.plot(H_c, color='C1')
    ax2.tick_params(axis='y', labelcolor='C1')
    plt.show()


# image 1
img_1 = Image.open('paperPhoto20210402174743810.bmp').convert('L')

npimg_1 = np.array(img_1)
show_Hc(npimg_1, max_ylim=12500)

eqimg_1, T1 = histogram_equalization(npimg_1)
plt.figure()
plt.title('monotonic pixel brightness transformation T')
plt.plot(T1)
plt.show()
show_Hc(eqimg_1, max_ylim=12500)


# image 2
img_2 = Image.open('paperPhoto20210402174743817.bmp').convert('L')

npimg_2 = np.array(img_2)
show_Hc(npimg_2, max_ylim=3500)

eqimg_2, T2 = histogram_equalization(npimg_2)
plt.figure()
plt.title('monotonic pixel brightness transformation T')
plt.plot(T2)
plt.show()
show_Hc(eqimg_2, max_ylim=3500)
