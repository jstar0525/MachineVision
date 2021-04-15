"""
Conduct a distance transform 
for the following obstacle image. 
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 1. read image

img = Image.open('paperPhoto2021031918372826.bmp').convert('L')
gray_img = gray_img = np.array(img)

plt.figure(1)
plt.imshow(gray_img, cmap='gray')
plt.show()

# 2. make distance metrics

def De(a,b):
    
    r_square = (a[0]-b[0])**2
    c_square = (a[1]-b[1])**2
    
    return (r_square+c_square)**0.5

def D4(a,b):
    
    r_abs = abs(a[0]-b[0])
    c_abs = abs(a[1]-b[1])
    
    return r_abs+c_abs

def D8(a,b):
    
    r_abs = abs(a[0]-b[0])
    c_abs = abs(a[1]-b[1])
    
    return max(r_abs,c_abs)
    
def D(D_type=D4):
    
    result_D = np.zeros((3,3))
    
    for i in range(3):
        for j in range(3):
            result_D[i,j] = D_type([i,j],[1,1])
            
    return result_D
    

# 3. Conduct a distance transform for the following obstacle image

def distance_transform(gray_img, D_type=D4):
    
    al = np.array([[1,1,0],
                   [1,0,0],
                   [1,0,0]])

    br = np.array([[0,0,1],
                   [0,0,1],
                   [0,1,1]])

    # 1
    row, col = gray_img.shape
    N_max = row+col
    F = np.zeros_like(gray_img, dtype='float64')
    F[gray_img==0] = N_max
    
    while(True):
        # 2
        for r in range(1,row-1):
            for c in range(1,col-1):
                df = ( D(D_type)+F[r-1:r+2,c-1:c+2] )*al
                df[1,1] = F[r,c]
                F[r,c] = np.min(df[np.nonzero(df)])
        # 3
        for r in reversed(range(1,row-1)):
            for c in reversed(range(1,col-1)):
                df = ( D(D_type)+F[r-1:r+2,c-1:c+2] )*br
                df[1,1] = F[r,c]
                F[r,c] = np.min(df[np.nonzero(df)])
        # 4
        if np.any(F!=N_max):
            break
            
    return F

F_De = distance_transform(gray_img, D_type=De)

plt.figure(2)
plt.title('F_De')
plt.imshow(F_De)
plt.show()

F_D4 = distance_transform(gray_img, D_type=D4)

plt.figure(3)
plt.title('F_D4')
plt.imshow(F_D4)
plt.show()

F_D8 = distance_transform(gray_img, D_type=D8)

plt.figure(4)
plt.title('F_D8')
plt.imshow(F_D8)
plt.show()


