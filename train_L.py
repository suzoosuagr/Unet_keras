# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from model.model_Unet_Laplacian import *
# from model.model_FC_Dense import *
from data_L import *
from keras import backend as k
import tensorflow as tf
from keras.callbacks import TensorBoard
from time import strftime

config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.8
session = tf.Session(config=config)
K.set_session(session)

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# batch_size=16
# steps_per_epoch = 2075/batch_size
# steps = 259/batch_size


# Unet_ batch = 16, steps oer epoch = 25 steps=6
# for membrane run
batch_size = 4
# steps_per_epoch = 2075//batch_size
# steps = 259//batch_size

steps_per_epoch = 55//batch_size
steps = 28//batch_size

data_gen_args = dict(vertical_flip=True,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator_L(batch_size=batch_size,
                        train_path='../data/ISBI2018/train',
                        image_folder='image',
                        mask_folder='label',
                        image_color_mode='rgb',
                        aug_dict=data_gen_args,
                        save_to_dir = None)
evalGene = evalGenerator_L(batch_size=batch_size,
                        train_path='../data/ISBI2018/eval',
                        image_folder='image',
                        mask_folder='label',
                        image_color_mode='rgb',
                        save_to_dir = None,)
model = unet_L()
tensorboard = TensorBoard(log_dir='experiment/test_L_unet_ISBI_rgb_1Dlaplacian_extra/logs/{}'.format(strftime('%Y-%m-%d_%H:%M:%S')), write_graph=False, update_freq='epoch')
model_checkpoint = ModelCheckpoint('../model_weights/test_L_unet_ISBI_rgb_1DLaplacian_extra.hdf5', monitor='val_IoU',verbose=1, save_best_only=True, mode='max')
model.fit_generator(myGene,steps_per_epoch=steps_per_epoch,epochs=200,callbacks=[model_checkpoint, tensorboard],
                    validation_data=evalGene, validation_steps=steps)
# model.fit(x=inputs, y=mask, batch_size=batch_size, epochs=20, verbose=1, callbacks=[tensorboard, model_checkpoint])
# model.evaluate_generator(evalGene, verbose=1, steps = steps)

# testGene = testGenerator("data/membrane/test")
# results = model.predict_generator(testGene,30,verbose=1)
# saveResult("data/membrane/test",results)