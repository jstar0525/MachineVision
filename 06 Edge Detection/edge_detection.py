"""
Following is black and white Lena image. 

1. Compare the Edge detection performances with Sobel, Robert, Prewitt, LOG operators: 
2. Describe their characteristic differences based on your simulation results 
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

#%% read image and operator

img = Image.open('lena.png').convert('L')
img = np.array(img)

roberts_1 = np.array([[ 1, 0],
                      [ 0,-1]])

roberts_2 = np.array([[ 0, 1],
                      [-1, 0]])

sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_y = np.array([[ 1, 2, 1],
                    [ 0, 0, 0],
                    [-1,-2,-1]])

prewitt_x = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]])

prewitt_y = np.array([[ 1, 1, 1],
                      [ 0, 0, 0],
                      [-1,-1,-1]])

LoG_3_1 = np.array([[ 0,-1, 0],
                    [-1, 4,-1],
                    [ 0,-1, 0]])

LoG_3_2 = np.array([[-1,-1,-1],
                    [-1, 8,-1],
                    [-1,-1,-1]])

LoG_5 = np.array([[ 0, 0,-1, 0, 0],
                  [ 0,-1,-2,-1, 0],
                  [-1,-2,16,-2,-1],
                  [ 0,-1,-2,-1, 0],
                  [ 0, 0,-1, 0, 0]])

LoG_9 = np.array([[ 0, 1, 1,  2,  2,  2, 1, 1, 0],
                  [ 1, 2, 4,  5,  5,  5, 4, 2, 1],
                  [ 1, 4, 5,  3,  0,  3, 5, 4, 1],
                  [ 2, 5, 3,-12,-24,-12, 3, 5, 2],
                  [ 2, 5, 0,-24,-40,-24, 0, 5, 2],
                  [ 2, 5, 3,-12,-24,-12, 3, 5, 2],
                  [ 1, 4, 5,  3,  0,  3, 5, 4, 1],
                  [ 1, 2, 4,  5,  5,  5, 4, 2, 1],
                  [ 0, 1, 1,  2,  2,  2, 1, 1, 0]])

#%% function

def show(img, result1, result2, result, thr_result):
    
    plt.imshow(img, cmap='gray')
    plt.show()
    
    plt.imshow(result1, cmap='gray')
    plt.show()
    
    plt.imshow(result2, cmap='gray')
    plt.show()
    
    plt.imshow(result, cmap='gray')
    plt.show()
    
    plt.imshow(thr_result, cmap='gray')
    plt.show()

def edge_detection(img, mask1, mask2, threshold, show_img=True):
    
    img_shape = img.shape
    
    try:
        if mask1.shape != mask2.shape:
            raise Exception('마스크의 크기가 서로 다릅니다.')
        filter_size = mask1.shape
    except Exception as e:
        print('예외가 발생했습니다.', e)

    result_shape = tuple(np.array(img_shape)-np.array(filter_size)+1)

    result1 = np.zeros(result_shape)
    result2 = np.zeros(result_shape)

    for h in range(0, result_shape[0]):
        for w in range(0, result_shape[1]):
            tmp = img[h:h+filter_size[0],w:w+filter_size[1]]
            result1[h,w] = np.abs(np.sum(tmp*mask1))
            result2[h,w] = np.abs(np.sum(tmp*mask2))
            
    result = result1 + result2
    
    thr_result = np.zeros(result_shape)
    thr_result[result>threshold] = 1
    
    if show_img:
        show(img, result1, result2, result, thr_result)
    
    return result1, result2, result, thr_result

#%% 

edge_detection(img, roberts_1, roberts_2, threshold=50)

edge_detection(img, sobel_x, sobel_y, threshold=140)

edge_detection(img, prewitt_x, prewitt_y, threshold=100)

edge_detection(img, LoG_3_1, np.zeros_like(LoG_3_1), threshold=70)
edge_detection(img, LoG_3_2, np.zeros_like(LoG_3_2), threshold=150)
edge_detection(img, LoG_5, np.zeros_like(LoG_5), threshold=300)
edge_detection(img, LoG_9, np.zeros_like(LoG_9), threshold=2000)


