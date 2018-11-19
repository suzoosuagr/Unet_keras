import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from skimage import filters
from keras import losses
from keras.engine.topology import Layer

def IoU(y_true, y_pred):
    y_pred = K.cast(y_pred > 0.5, 'float32')
    H, W, _ = y_pred.get_shape().as_list()[1:]
    pred_flat = K.reshape(y_pred, [-1, H * W])
    true_flat = K.reshape(y_true, [-1, H * W])

    intersection = K.sum(pred_flat * true_flat, axis=1) + 1e-7
    denominator = K.sum(
        pred_flat, axis=1) + K.sum(
            true_flat, axis=1) + 1e-7 - intersection

    return K.mean(intersection / denominator)

def IoU_bce_loss(y_true, y_pred):
    ''' IoU loss + binary_cross_entropy
    called by function name by `model.compile(loss=IoU_bce_loss)` 
    Return:
        loss op
    '''
    bce_loss = losses.binary_crossentropy(y_true, y_pred)
    H, W, _ = y_pred.get_shape().as_list()[1:]
    pred_flat = K.reshape(y_pred, [-1, H * W])
    true_flat = K.reshape(y_true, [-1, H * W])
    intersection = K.sum(pred_flat * true_flat, axis=1)
    denominator = K.sum(
        pred_flat, axis=1) + K.sum(
            true_flat, axis=1) + 1e-7 - intersection

    iou_loss = K.mean(intersection / denominator)
    return bce_loss - iou_loss + 1.0


def unet_L(pretrained_weights = None,input_size = (256,256,3), laplacian_size = (256,256,1)):
    inputs = Input(input_size, name='input_1')
    laplacian = Input(laplacian_size, name='input_2')
    inputs_1 = concatenate([inputs, laplacian], axis=3)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs_1)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    laplacian1 = MaxPooling2D(pool_size=(2, 2))(laplacian)

    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    laplacian2 = MaxPooling2D(pool_size=(2, 2))(laplacian1)

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    laplacian3 = MaxPooling2D(pool_size=(2, 2))(laplacian2)
    
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    merge6 = concatenate([laplacian3, merge6], axis=3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    merge7 = concatenate([laplacian2, merge7], axis=3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    merge8 = concatenate([laplacian1, merge8], axis=3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9],axis = 3)
    merge9 = concatenate([laplacian, merge9], axis=3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs = [inputs,laplacian], output = conv10)

    model.compile(optimizer = Adam(lr = 1e-3), loss = IoU_bce_loss, metrics = ['accuracy', IoU])
    
    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model


