"""
1. Following is an edge image. 
   Draw the 10 most explicit straight lines using Hough Transform on the edge image. 
   
2. Choose the straight lines which are within +- 10 degree 
   from the horizontal axis among the above 10 lines. 
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

#%%  read image

img = Image.open('fig.png').convert('L')
# img = Image.open('test.png').convert('L')
bin_img = np.array(img)
bin_img[bin_img>0] = 1

plt.figure()
plt.imshow(bin_img, cmap='gray')
plt.title('Input Image')
plt.show()

#%%  make hough space

def make_hough_space(bin_img, theta_resolution = 0.1, rho_resolution = 1):

    global D, R, C
    
    R, C = bin_img.shape
    
    n_theta = int(180 / theta_resolution + 1)
    theta = np.linspace(-90, 90, n_theta)
    rad_theta = np.deg2rad(theta)
    
    D = int( np.sqrt(R**2 + C**2) )
    n_rho = int(D*2 / rho_resolution + 1)
    rho = np.linspace(-D, D, n_rho)
    
    hough_space = np.zeros((n_rho,n_theta))
    for r in range(R):
        for c in range(C):
            if bin_img[r,c]:
                for theta_idx, rad in enumerate(rad_theta):
                    tmp = c*np.cos(rad) + r*np.sin(rad)
                    rho_idx = np.argmin(abs(rho-tmp))
                    hough_space[rho_idx, theta_idx] += 1
                    
    return hough_space, rad_theta, rho

hough_space, rad_theta, rho = make_hough_space(bin_img, theta_resolution = 0.1, rho_resolution = 1)
                
plt.figure(figsize=(6,9))
str_hough_space = hough_space*3 #3, 200
str_hough_space[str_hough_space>255] = 255
plt.imshow(str_hough_space, extent=[-90, 90, -D, D], cmap='jet', aspect=1 / 10)
plt.title('Hough Space')
plt.xlabel('Angles (degrees)')
plt.ylabel('Distance (pixels)')
plt.show()

#%% take intersection points in hough space

def select_lines(hough_space, rad_theta, rho, num=10, threshold=20):
    
    hough = hough_space.copy()
    idx = []
    while(len(idx) < num):
        rho_idx, theta_idx = np.unravel_index(hough.argmax(), hough.shape)
        if not idx:
            idx.append([rho_idx, theta_idx])
        else:
            np_idx = np.array(idx, dtype='int64')
            thr = abs(rho[np_idx[:,0]]-rho[rho_idx]) \
                    +abs(rad_theta[np_idx[:,1]]-rad_theta[theta_idx])*5
            if np.min(thr) > threshold :
                idx.append([rho_idx, theta_idx])
            else:
                hough[rho_idx, theta_idx] = 0
    idx = np.array(idx, dtype='int64')
    
    element_rho = rho[idx[:,0]]
    element_theta = rad_theta[idx[:,1]]
            
    return idx, element_rho, element_theta
        
idx, element_rho, element_theta = select_lines(hough_space, rad_theta, rho, num=10, threshold=20)

plt.figure(figsize=(6,9))
str_hough_space = hough_space*3 #3, 200
str_hough_space[str_hough_space>255] = 255
plt.imshow(str_hough_space, cmap='jet')
plt.plot(idx[:,1], idx[:,0], 'wo')
plt.title('Hough Space')
plt.xlabel('Angles (degrees)')
plt.ylabel('Distance (pixels)')
plt.axis('off')
plt.show()

#%% draw lines

m = - np.cos(element_theta)/np.sin(element_theta)
b = element_rho / np.sin(element_theta)

plt.figure()
plt.imshow(bin_img, cmap='gray')
for i in range(len(m)):
    for c in range(C):
        y = int(m[i]*c+b[i])
        if y >= 0 and y < R:
            plt.plot(c, y, marker='.', color='red')
    for r in range(R):
        x = int((r-b[i])/m[i])
        if x >=0 and x < C:
            plt.plot(x, r, marker='.', color='red')
plt.title('Detecting lines')
plt.show()

#%% filter lines

grad = []

for i in range(len(m)):
    if np.tan(np.deg2rad(-10)) < m[i] and m[i] < np.tan(np.deg2rad(10)):
        grad.append([m[i], b[i]])
        

plt.figure()
plt.imshow(bin_img, cmap='gray')
for i in range(len(grad)):
    for c in range(C):
        y = int(grad[i][0]*c+grad[i][1])
        if y >= 0 and y < R:
            plt.plot(c, y, marker='.', color='red')
    for r in range(R):
        x = int((r-grad[i][1])/grad[i][0])
        if x >=0 and x < C:
            plt.plot(x, r, marker='.', color='red')
plt.title('Filter lines')
plt.show()


