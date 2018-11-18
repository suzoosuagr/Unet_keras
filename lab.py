import numpy as np
import skimage.filters as filters
import skimage.io as io
import skimage.transform as trans
import cv2

def normalize_L(laplacian):
    max_val = laplacian.max()
    min_val = laplacian.min()
    domain = max_val - min_val + 1e-7
    laplacian = laplacian - min_val
    laplacian = laplacian / domain
    # return np.clip(laplacian, 0, 1)
    return laplacian

# img = cv2.imread('test.jpg', 0)
# print('helsk')


img = io.imread('test.jpg', as_gray=True)
# img = trans.resize(img, (256,256))
# laplacian = cv2.Laplacian(img, cv2.CV_32F)
laplacian = filters.laplace(img, 3)
laplacian = normalize_L(laplacian)
laplacian = np.resize(laplacian, (256,256,1))
print(laplacian.shape)
print(laplacian.max())
io.imshow(laplacian)
io.show()
print('testpoint')