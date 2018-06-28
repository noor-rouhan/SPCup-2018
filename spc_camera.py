# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 23:28:17 2017

@author: User
"""

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # see issue #152
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # -1 !!!!

# Initialising the CNN
classifier = Sequential()

#convultion
classifier.add(Convolution2D(32, 3, 3, input_shape = (64,64,3), activation = 'relu'))
#pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))
#second convolution net
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
#flattening
classifier.add(Flatten())

classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))
#compile
classifier.compile(optimizer = 'rmsprop', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

#image augmentation
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('train',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('train',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode = 'categorical')

classifier.fit_generator(training_set,
                         steps_per_epoch=4000,
                         epochs=5,
                         validation_data= test_set,
                         nb_val_samples=2000)
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat4.jpg', target_size = (64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
if result[0][0] == 1:
    prediction = 'dog' 
else:
    prediction = 'cat'
