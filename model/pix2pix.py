from __future__ import print_function, division
import scipy

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
from keras import backend as K
import matplotlib.pyplot as plt
import sys
from data_loader import DataLoader
import numpy as np
import os
from keras import callbacks
import logging
import tensorflow as tf
from keras import losses



def set_logger(log_path):
    """Sets the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

class Pix2Pix():
    """ -- Generator generate mask 1 for nevus 0 for normal skin
        -- Discriminator determin the reality of mask its inputs including two patterns:
            -- [real_mask, ori_img] with ground truth all one
            -- [fake_mask, ori_img] with ground truth all zero 
    """
    def __init__(self):
        # Input shape
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 1
        self.g_channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.lab_shape = (self.img_rows, self.img_cols, self.g_channels)

        # Summary
        self.writer = tf.summary.FileWriter('../experiment/ISIC_gray_cGAN/logs/{}'.format(datetime.datetime.now()))

        # Configure data loader ISICKeras 
        self.dataset_name = 'ISICKeras'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.img_rows, self.img_cols))


        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D 
        self.gf = 64
        self.df = 64

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generator
        #-------------------------

        # Build the generator
        self.generator = self.build_generator()

        # Input images and their conditioning images
        img_A = Input(shape=self.lab_shape)
        img_B = Input(shape=self.img_shape)

        # By conditioning on B generate a fake version of A
        fake_A = self.generator(img_B)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_A, img_B])

        self.combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
        self.combined.compile(loss=['mse', 'mae'],
                              loss_weights=[1, 100],
                              optimizer=optimizer)

    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=3, bn=True): # original f_size = 3
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=3, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)

        # Downsampling
        d1 = conv2d(d0, self.gf, bn=False)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)
        d5 = conv2d(d4, self.gf*8)
        d6 = conv2d(d5, self.gf*8)
        d7 = conv2d(d6, self.gf*8)

        # Upsampling
        u1 = deconv2d(d7, d6, self.gf*8)
        u2 = deconv2d(u1, d5, self.gf*8)
        u3 = deconv2d(u2, d4, self.gf*8)
        u4 = deconv2d(u3, d3, self.gf*4)
        u5 = deconv2d(u4, d2, self.gf*2)
        u6 = deconv2d(u5, d1, self.gf)

        u7 = UpSampling2D(size=2)(u6)
        output_img = Conv2D(self.g_channels, kernel_size=1, strides=1, padding='same', activation='tanh')(u7)

        return Model(d0, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=3, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_A = Input(shape=self.lab_shape)
        img_B = Input(shape=self.img_shape)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])

        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=1, strides=1, padding='same')(d4)

        return Model([img_A, img_B], validity)

    def train(self, epochs, batch_size=1, sample_interval=50):

        # tensorboard.set_model(self.generator)

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)
        self.best_iou = 0
        self.best_g_loss = 0

        for epoch in range(epochs):
            self.test_iou_ls = []
            self.test_g_loss = []            
            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Condition on B and generate a translated version
                fake_A = self.generator.predict(imgs_B)

                # Train the discriminators (original images = real / generated = Fake)
                d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], valid)
                d_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_B], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                iou_acc = self.IoU(imgs_A, fake_A)
                # -----------------
                #  Train Generator
                # -----------------

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])

                # Plot the progress
                if batch_i % 100 == 0:
                    elapsed_time = datetime.datetime.now() - start_time
                    print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] [IoU: %f] time: %s" % (epoch, epochs,
                                                                            batch_i, self.data_loader.n_batches,
                                                                            d_loss[0], 100*d_loss[1],
                                                                            g_loss[0],
                                                                            iou_acc,
                                                                            elapsed_time))
            for batch_j, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size, is_testing=True)):
                fake_A = self.generator.predict(imgs_B)
                iou_acc = self.IoU(imgs_A, fake_A)
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])
                test_g_loss = g_loss[0]
                self.test_iou_ls.append(iou_acc)
                self.test_g_loss.append(test_g_loss)
            
            avg_test_iou = np.mean(self.test_iou_ls)
            avg_test_g_loss = np.mean(self.test_g_loss)
            summary_test_iou = tf.Summary(value=[
                tf.Summary.Value(tag='avg_test_iou', simple_value=avg_test_iou),
                tf.Summary.Value(tag='avg_test_g_loss', simple_value=avg_test_g_loss)
            ])
            self.writer.add_summary(summary_test_iou)
            if avg_test_iou > self.best_iou:
                self.best_iou = avg_test_iou
                self.best_g_loss = avg_test_g_loss
                logging.info('The best test_iou is {:.2f} test_g_loss is {:.2f} at {} epoch '.format(self.best_iou, self.best_g_loss ,epoch))
                logging.info('New best iou updated, now the best test iou is {:.4f}'.format(avg_test_iou))
                self.combined.save(filepath='../../model_weights/ISIC_gray_cGAN.hdf5')
            else:
                logging.info('No new best iou, and current best test iou is {:.2f}'.format(self.best_iou))
                # If at save interval => save generated image samples
                # if batch_i % sample_interval == 0:
                #     self.sample_images(epoch, batch_i)
        logging.info("[{}] Training begin at {} end at {}".format(self.img_shape, start_time, datetime.datetime.now()))
    # def sample_images(self, epoch, batch_i):
    #     os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
    #     r, c = 3, 3

    #     imgs_A, imgs_B = self.data_loader.load_data(batch_size=3, is_testing=True)
    #     fake_A = self.generator.predict(imgs_B)

    #     gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])

    #     # Rescale images 0 - 1
    #     gen_imgs = 0.5 * gen_imgs + 0.5

    #     titles = ['Condition', 'Generated', 'Original']
    #     fig, axs = plt.subplots(r, c)
    #     cnt = 0
    #     for i in range(r):
    #         for j in range(c):
    #             axs[i,j].imshow(gen_imgs[cnt])
    #             axs[i, j].set_title(titles[i])
    #             axs[i,j].axis('off')
    #             cnt += 1
    #     fig.savefig("images/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i))
    #     plt.close()

    def IoU(self, y_true, y_pred):
        y_pred = (y_pred > 0) * 1.0
        y_true = (y_true > 0) * 1.0
        _, H, W, _ = y_pred.shape
        pred_flat = np.reshape(y_pred, [-1, H * W])
        true_flat = np.reshape(y_true, [-1, H * W])

        intersection = np.sum(pred_flat * true_flat, axis=1) + 1e-7
        denominator = np.sum(
            pred_flat, axis=1) + np.sum(
                true_flat, axis=1) + 1e-7 - intersection

        return np.mean(intersection / denominator)

if __name__ == '__main__':
    set_logger('../train_cGAN.log')
    logging.info('Creating model ...')
    gan = Pix2Pix()
    gan.train(epochs=200, batch_size=1, sample_interval=200)