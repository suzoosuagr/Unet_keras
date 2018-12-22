from model.model_cGAN import *
from data import *
from keras import backend as k
import tensorflow as tf
from keras.callbacks import TensorBoard
from time import strftime

config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.8
session = tf.Session(config=config)
K.set_session(session)

batch_size = 2
epochs = 200
steps_per_epoch = 25
steps = 6

img_rows = 256
img_cols = 256
img_channels = 3
patch =  img_rows // 2**4
disc_patch = (patch, patch, 1)

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(batch_size=batch_size,
                        train_path='../data/ISICKeras/train',
                        image_folder='image',
                        mask_folder='label',
                        image_color_mode='rgb',
                        aug_dict=data_gen_args,
                        save_to_dir = None)
evalGene = evalGenerator(batch_size=batch_size,
                        train_path='../data/ISICKeras/eval',
                        image_folder='image',
                        mask_folder='label',
                        image_color_mode='rgb',
                        save_to_dir = None)
generator, discriminator, combined = Pix2Pix(img_rows, img_cols, img_channels)
# tensorboard = TensorBoard(log_dir='experiment/ISIC_rgb_inputs_iou_bce_GAN/logs/{}'.format(strftime('%Y-%m-%d_%H:%M:%S')), write_graph=False)
# model_checkpoint = ModelCheckpoint('../model_weights/ISIC_rgb_inputs_iou_bce_GAN.hdf5', monitor='val_IoU',verbose=1, save_best_only=True, mode='max')
writer = tf.summary.FileWriter('experiment/ISIC_rgb_inputs_GAN/logs/{}'.format(strftime('%Y-%m-%d_%H:%M:%S')))

valid = np.ones((batch_size,) + disc_patch)
fake = np.zeros((batch_size,) + disc_patch)
names = ['d_loss', 'acc', 'g_loss']

for epoch in range(epochs):
    for batch_i, (imgs_B, imgs_A) in enumerate(myGene):
        fake_A = generator.predict(imgs_B)
        # original images = real / generated = Fake
        d_loss_real = discriminator.train_on_batch([imgs_A, imgs_B], valid)
        d_loss_fake = discriminator.train_on_batch([fake_A, imgs_B], fake)
        d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
        # Train the generators
        g_loss = combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])
        if batch_i > steps_per_epoch - 2:
            break
    print("[Epoch {:d}/{:d} [Steps {:d}/{:d}] [D loss: {:f}] [acc: {:.2f}] [G loss: {:f}]]".format(epoch, epochs, batch_i, steps_per_epoch, d_loss[0], d_loss[1], g_loss[0]))
    logs = [d_loss[0], d_loss[1], g_loss[0]]
    summary = tf.Summary(value=[tf.Summary.Value(tag='d_loss',simple_value=)])
    