# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from model.model_Unet import *
from data import *
from keras import backend as k
import tensorflow as tf
from keras.callbacks import TensorBoard
from time import time

config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.8
session = tf.Session(config=config)
K.set_session(session)

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# batch_size=16
# steps_per_epoch = 2075/batch_size
# steps = 259/batch_size

# for membrane run
batch_size = 16
steps_per_epoch = 30

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(batch_size=batch_size,
                        train_path='membrane/train',
                        image_folder='image',
                        mask_folder='label',
                        aug_dict=data_gen_args,
                        save_to_dir = None)
# evalGene = evalGenerator(batch_size=batch_size,
#                         train_path='../data/ISICKeras/eval',
#                         image_folder='image',
#                         mask_folder='label',
#                         save_to_dir = None,)
model = unet()
tensorboard = TensorBoard(log_dir='experiment/membrane/logs/{}'.format(time()), write_graph=False, update_freq='epoch')
model_checkpoint = ModelCheckpoint('membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=steps_per_epoch,epochs=20,callbacks=[model_checkpoint, tensorboard])
#                     validation_data=evalGene, validation_steps=steps)
# model.evaluate_generator(evalGene, verbose=1, steps = steps)

# testGene = testGenerator("data/membrane/test")
# results = model.predict_generator(testGene,30,verbose=1)
# saveResult("data/membrane/test",results)