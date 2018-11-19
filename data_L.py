from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
import skimage.filters as filters
import skimage.color as color
import skimage.feature as feature
import keras.backend as K



Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]

COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])


def normalize(laplacian):
    max_val = laplacian.max()
    min_val = laplacian.min()
    domain = max_val - min_val + 1e-7
    laplacian = laplacian - min_val
    return laplacian / domain
    

def adjustData_L_gray(img, mask):
    if(np.max(img) > 1):
        img = img / 255
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        img_size = img.shape
        img_r = np.resize(img, (256,256))
        laplacian = filters.laplace(img_r, 3)
        laplacian = normalize(laplacian)
        laplacian = np.resize(laplacian, img_size)
    return (img, laplacian, mask)

def adjustData_L_rgb(img, mask):
    gray = color.rgb2gray(img)
    if(np.max(img) > 1):
        img = img / 255
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        img_r = np.resize(gray, (256,256))
        laplacian = filters.laplace(img_r, 3)
        laplacian = normalize(laplacian)
        laplacian = np.resize(laplacian, mask.shape)
    return (img, laplacian, mask)

def adjustData_Canny_gray(img, mask):
    if(np.max(img) > 1):
        img = img / 255
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        img_size = img.shape
        img_c = np.resize(img, (256,256))
        img_c = feature.canny(img_c) * 1.0
        img_c = np.resize(img_c, img_size)
    return (img, img_c, mask)
    

def testGenerator(test_path,num_image = 30,target_size = (256,256),flag_multi_class = False,as_gray = True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d.png"%i),as_gray = as_gray)
        img = img / 255
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img

def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255

def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)

def trainGenerator_L(batch_size,train_path,image_folder,mask_folder,aug_dict, image_color_mode = 'grayscale',
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode, # this line is for membrane
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,laplacian,mask = adjustData_Canny_gray(img,mask)
        yield [img, laplacian],[mask]

def evalGenerator_L(batch_size, train_path, image_folder, mask_folder, image_color_mode = 'grayscale',mask_color_mode = 'grayscale', 
                    image_save_prefix='image', mask_save_prefix = 'mask', flag_multi_class = False, num_class = 2, save_to_dir = None,
                    target_size = (256, 256), seed = 1):
    image_datagen = ImageDataGenerator()
    mask_datagen = ImageDataGenerator()
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode=image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    eval_generator = zip(image_generator, mask_generator)
    for (img,mask) in eval_generator:
        img,laplacian,mask = adjustData_Canny_gray(img,mask)
        yield [img, laplacian], [mask]