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

def Pix2Pix(img_rows=256, img_cols=256, img_channels=3, pretrained_weights=None, ):
    # Params Setting
    df = 64
    gf = 64
    img_shape = (img_rows, img_cols, img_channels)
    # Set basic params

    discriminator = build_discriminator(img_shape, df)
    discriminator.compile(loss='mse', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
    generator = build_generator(img_shape, gf)

    img_A = Input(shape=(256,256,1))
    img_B = Input(shape=img_shape)

    fake_A = generator(img_B)
    
    discriminator.trainable=False

    valid = discriminator([fake_A, img_B])

    combined = Model(inputs=[img_A, img_B], output=[valid, fake_A])
    combined.compile(loss=['mse', 'mae'], loss_weights=[1,100], optimizer=Adam(0.0002, 0.5))

    return generator, discriminator, combined

def build_generator(img_shape, gf):
    def conv2d(layer_input, filters, f_size=4, bn=True):
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d

    def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = BatchNormalization(momentum=0.8)(u)
        u = Concatenate()([u, skip_input])
        return u

    #Image input
    d0 = Input(shape=img_shape)

    # Downsampling
    d1 = conv2d(d0, gf, bn=False)
    d2 = conv2d(d1, gf*2)
    d3 = conv2d(d2, gf*4)
    d4 = conv2d(d3, gf*8)
    d5 = conv2d(d4, gf*8)
    d6 = conv2d(d5, gf*8)
    d7 = conv2d(d6, gf*8)

    # Upsampling
    u1 = deconv2d(d7, d6, gf*8)
    u2 = deconv2d(u1, d5, gf*8)
    u3 = deconv2d(u2, d4, gf*8)
    u4 = deconv2d(u3, d3, gf*4)
    u5 = deconv2d(u4, d2, gf*2)
    u6 = deconv2d(u5, d1, gf)

    u7 = UpSampling2D(size=2)(u6)
    output_img = Conv2D(1, kernel_size=1, strides=1, padding='same', activation='tanh')(u7)

    return Model(d0, output_img)

def build_discriminator(img_shape, df):
    def d_layer(layer_input, filters, f_size=4, bn=True):
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d
    
    img_A = Input(shape=(256,256,1))
    img_B = Input(shape=img_shape)

    combined_imgs = Concatenate(axis = -1)([img_A, img_B])

    d1 = d_layer(combined_imgs, df, bn=False)
    d2 = d_layer(d1, df*2)
    d3 = d_layer(d2, df*4)
    d4 = d_layer(d3, df*8)

    validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

    return Model([img_A, img_B], validity)

    

