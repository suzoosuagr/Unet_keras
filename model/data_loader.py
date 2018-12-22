import scipy
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

class DataLoader():
    def __init__(self, dataset_name='ISICKeras', img_res=(256, 256)):
        self.dataset_name = dataset_name
        self.img_res = img_res

    def load_data(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "eval"
        path_images = glob('../../data/%s/%s/image/*' % (self.dataset_name, data_type))
        path_labels = glob('../../data/%s/%s/label/*' % (self.dataset_name, data_type))

        batch_images = np.random.choice(path_images, size=batch_size)
        batch_labels = np.random.choice(path_labels, size=batch_size)

        imgs_A = [] # This is the label
        imgs_B = [] # This is the image of nevus 
        for img_path, lab_path in zip(batch_images, batch_labels):
            img = self.imread(img_path)
            lab = self.labread(lab_path)

            img_A = np.resize(lab, (self.img_res[0], self.img_res[1], 1))
            img_B = np.resize(img, (self.img_res[0], self.img_res[1], 1)) # 3 for rgb 1 for gray

            # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                img_A = np.fliplr(img_A)
                img_B = np.fliplr(img_B)

            imgs_A.append(img_A)
            imgs_B.append(img_B)

        imgs_A = np.array(imgs_A)/127.5 - 1.
        imgs_B = np.array(imgs_B)/127.5 - 1.

        return imgs_A, imgs_B

    def load_batch(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "eval"
        path_images = glob('../../data/%s/%s/image/*' % (self.dataset_name, data_type))
        path_labels = glob('../../data/%s/%s/label/*' % (self.dataset_name, data_type))

        self.n_batches = int(len(path_images) / batch_size)

        for i in range(self.n_batches-1):
            batch_images = path_images[i*batch_size:(i+1)*batch_size]
            batch_labels = path_labels[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for img, lab in zip(batch_images, batch_labels):
                img = self.imread(img)
                lab = self.labread(lab)

                # img_A = scipy.misc.imresize(lab, self.img_res)
                # img_B = scipy.misc.imresize(img, self.img_res)
                img_A = np.resize(lab, (self.img_res[0], self.img_res[1], 1))
                img_B = np.resize(img, (self.img_res[0], self.img_res[1], 1)) # 3 for rgb 1 for gray

                if not is_testing and np.random.random() > 0.5:
                        img_A = np.fliplr(img_A)
                        img_B = np.fliplr(img_B)

                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A)/127.5 - 1.
            imgs_B = np.array(imgs_B)/127.5 - 1.

            yield imgs_A, imgs_B

    def imread(self, path):
        return scipy.misc.imread(path, mode='L').astype(np.float) # RGB or L(gray)
    
    def labread(self, path):
        return scipy.misc.imread(path, mode='L').astype(np.float)