import skimage.io as io
import skimage.transform as trans
import os
import numpy as np

data_dir = '/Users/jiyangwang/desktop/ISBI2018/train/image/'
# filenames = [f for f in os.listdir(data_dir) if f.endswith('.tif')]
filenames = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.jpg')]
save_dir = '/Users/jiyangwang/desktop/ISBI2018_resize/train/image/'

for f in filenames:
    splits = f.split('.')
    splits = splits[0].split('/')
    ids = splits[-1]
    save_file = os.path.join(save_dir, ids+'.jpg')
    img = io.imread(f, as_gray = False)
    resize = trans.resize(img, (256,256,3))
    resize_max = resize.max()
    print(resize_max)
    resize = resize/resize_max
    io.imsave(save_file, resize)




# img = io.imread('IDRiD_55_OD.tif', as_gray=True)
# trans = trans.resize(img, (256,256))
# trans_max = trans.max()
# trans = trans/trans_max
# io.imsave('ISBI_55.png', trans)
print('Finished')