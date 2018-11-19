import numpy as np
import skimage.feature as feature
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
c_img = feature.canny(img) * 1.0
io.imshow(c_img)
io.show()
print('finished')