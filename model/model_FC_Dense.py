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

def BN_ReLU_Conv(inputs, n_filters, filter_size=3, dropout=0.2):
    inputs = BatchNormalization()(inputs)
    inputs = Conv2D(n_filters, filter_size, activation='relu', padding='same',kernel_initializer='he_normal')(inputs)
    inputs = Dropout(dropout)(inputs)
    return inputs

def TransitionDown(inputs, n_filters, dropout=0.2):
    inputs =  BN_ReLU_Conv(inputs, n_filters, filter_size=1, dropout=dropout)
    inputs = MaxPool2D(pool_size=(2,2))(inputs)
    return inputs

def TransitionUp(skip_connection, block_to_upsample, n_filters):
    inputs = concatenate(block_to_upsample, axis=3)
    inputs = Conv2D(n_filters, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(inputs))
    inputs = concatenate([inputs, skip_connection])
    return inputs

def FC_Dense(pretrained_weights=None, input_size=(256,256,1), n_filters_first=32):
    inputs = Input(input_size)
    growth_rate = 16
    # First Conv
    # all feature maps store in `stack`
    stack = Conv2D(n_filters_first, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    # The number of feature maps store in n_filters
    n_filters = n_filters_first 

    #####################
    # Downsampling path #
    #####################
    skip_connection_list = []
    for i in range(4):
        for j in range(5):
            conv = BN_ReLU_Conv(stack, growth_rate, dropout=0.2)
            stack = concatenate([conv, stack], axis=3)
            n_filters += growth_rate
        skip_connection_list.append(stack)
        stack = TransitionDown(stack, n_filters, dropout=0.2)

    # skip_connection_list = skip_connection_list[::-1]

    ###############
    # Bottleneck #
    ############## 

    block_to_upsample = []

    for i in range(5):
        conv = BN_ReLU_Conv(stack, growth_rate, dropout=0.2)
        block_to_upsample.append(conv)
        stack = concatenate([stack, conv], axis=3)
    
    ###################
    # Upsampling path #
    ###################

    for i in range(4):
        n_filters_keep = growth_rate * 5
        stack = TransitionUp(skip_connection_list[3-i], block_to_upsample, n_filters_keep)

        block_to_upsample = []
        for j in range(5):
            conv = BN_ReLU_Conv(stack, growth_rate, dropout=0.2)
            block_to_upsample.append(conv)
            stack = concatenate([stack, conv], axis=3)
    ############
    # Sigmoid #
    ###########
    for j in range(3):
        conv = BN_ReLU_Conv(stack, 8, dropout=0.2)
        block_to_upsample.append(conv)
        stack = concatenate([stack, conv], axis=3)

    conv1 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(stack)
    conv2 = Conv2D(1, 1, activations='relu', padding='same', kernel_initializer='he_normal')(conv1)

    model = Model(inputs=inputs, output=conv2)
    model.compile(optimizer = Adam(lr = 1e-4), loss = IoU_bce_loss, metrics = ['accuracy', IoU])

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model