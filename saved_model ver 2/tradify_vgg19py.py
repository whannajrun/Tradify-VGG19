from cutmix_keras import CutMixImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os
import zipfile
import tensorflow as tf

local_zip = 'jajanan_indonesia_final.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')

zip_ref.extractall('./')
zip_ref.close()

base_dir = './jajanan_indonesia_final'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 64

def train_val_generators(dir_train, dir_validation, dir_test):
    train_datagen = ImageDataGenerator(rescale = 1./255.,
                                       rotation_range=40,
                                       width_shift_range=.2,
                                       height_shift_range=.2,
                                       shear_range=.2,
                                       zoom_range=.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest')
    
    train_generator1 = train_datagen.flow_from_directory(dir_train, batch_size=16, 
                                                        class_mode = 'categorical', 
                                                        seed=100, shuffle=True,
                                                        target_size=(64, 64))
    train_generator2 = train_datagen.flow_from_directory(dir_train, batch_size=16, 
                                                        class_mode = 'categorical', 
                                                        seed=100, shuffle=True,
                                                        target_size=(64, 64))
    train_generator = CutMixImageDataGenerator(
    generator1=train_generator1,
    generator2=train_generator2,
    img_size=IMG_SIZE,
    batch_size=16)
    
    validation_datagen = ImageDataGenerator(rescale = 1./255.)
    validation_generator = validation_datagen.flow_from_directory(dir_validation, batch_size=16, class_mode = 'categorical', seed=100, target_size=(64, 64))
    
    test_datagen = ImageDataGenerator(rescale = 1./255)
    test_generator = test_datagen.flow_from_directory(dir_test, batch_size=16,
                                                      class_mode='categorical', 
                                                      seed=100, target_size=(64, 64))
    
    return train_generator, validation_generator, test_generator

train_generator, validation_generator, test_generator = train_val_generators(train_dir, validation_dir, test_dir)

from tensorflow.keras.applications import VGG19
from tensorflow.keras import layers
from tensorflow.keras import Model

base_model = VGG19(weights='imagenet', input_shape=(64, 64, 3), include_top=False)
base_model.trainable = False

last = base_model.get_layer('block4_pool').output

x = layers.Flatten()(last)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.1)(x)                  
x = layers.Dense(8, activation='softmax')(x)

from tensorflow.keras.optimizers import RMSprop, Adam

model = Model(inputs= base_model.input,outputs= x) 
model.compile(optimizer = Adam(lr=0.0001),
              loss = 'categorical_crossentropy', 
              metrics = ['accuracy'])

history = model.fit(
            train_generator,
            validation_data = validation_generator,
            epochs = 20,
            steps_per_epoch = 64,
            verbose = 1)

results = model.evaluate(test_generator)

model_dir = './model2_frompy/model'
export_dir = os.path.join(model_dir, 'saved_model2')
tf.saved_model.save(model,export_dir=export_dir)

mode = "Speed" 

if mode == 'Storage':
    optimization = tf.lite.Optimize.OPTIMIZE_FOR_SIZE
elif mode == 'Speed':
    optimization = tf.lite.Optimize.OPTIMIZE_FOR_LATENCY
else:
    optimization = tf.lite.Optimize.DEFAULT

converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)

converter.optimizations = [optimization]

tflite_model = converter.convert();

import pathlib

tflite_model_file = pathlib.Path('./model2_frompy/model2.tflite')
tflite_model_file.write_bytes(tflite_model)

from tensorflow.keras.models import load_model
model.save("model2.h5")
loaded_model = load_model("model2.h5")
loss, accuracy = loaded_model.evaluate(test_generator)