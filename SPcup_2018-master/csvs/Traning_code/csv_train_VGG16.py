from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from PIL import Image
from skimage.transform import resize
from random import shuffle
import numpy as np
i = 0
list_paths = []
for subdir, dirs, files in os.walk("./input"):
    for file in files:
        #print os.path.join(subdir, file)
        filepath = subdir + os.sep + file
        list_paths.append(filepath)
print(list_paths)
list_train = [filepath for filepath in list_paths if "\\train" in filepath]
print(list_train)
# print(list_train)
#shuffle(list_train)
list_test = [filepath for filepath in list_paths if "\\test" in filepath]
print(list_test)
 
list_train = list_train
list_test = list_test
#index = [os.path.basename(filepath) for filepath in list_test]
list_classes = list(set([os.path.dirname(filepath).split(os.sep)[-1] for filepath in list_paths if "train" in filepath]))
print('list class')
list_classes = ['Sony-NEX-7',
 'Motorola-X',
 'HTC-1-M7',
 'Samsung-Galaxy-Note3',
 'Motorola-Droid-Maxx',
 'iPhone-4s',
 'iPhone-6',
 'LG-Nexus-5x',
 'Samsung-Galaxy-S4',
 'Motorola-Nexus-6']

 
def get_class_from_path(filepath):
    return os.path.dirname(filepath).split(os.sep)[-1]
 
def read_and_resize(filepath):
    global i
    im_array = pd.DataFrame.as_matrix(pd.read_csv(filepath))
 #   pil_im = Image.fromarray(im_array)
  #  pil_im = Image.fromarray(im_array.astype('uint8'), 'RGB')
    pil_im = Image.fromarray(im_array.astype('uint8'))
    new_array = np.array(pil_im.resize((256, 256)))
    i = i + 1
    print(i)
    return new_array/255

def label_transform(labels):
    labels = pd.get_dummies(pd.Series(labels))
    label_index = labels.columns.values
    return labels, label_index
 
print('xtrain')
X_train = np.array([read_and_resize(filepath) for filepath in list_train])
print(X_train)
print(X_train.shape)
print('xtrain complete')
#reshaping
backup = X_train
X_train = X_train.reshape(X_train.shape[0], 256, 256,1)
X_test = np.array([read_and_resize(filepath) for filepath in list_test])
#df = pd.DataFrame(backup)
X_test = X_train.reshape(X_train.shape[0], 256, 256,1)
labels = [get_class_from_path(filepath) for filepath in list_train]
y, label_index = label_transform(labels)
y = np.array(y)

from keras import backend as K
from keras.utils.layer_utils import convert_all_kernels_in_model
def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224,3)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))
    model.summary()

    if weights_path:
        model.load_weights(weights_path)

    return model


file_path="best.hdf5"
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras import optimizers, losses, activations, models
from keras.layers import Input, BatchNormalization, GlobalMaxPool2D, Concatenate
input_shape = (256, 256,1)
nclass = len(label_index)
print('model')
def get_model():
 
    nclass = len(label_index)
    inp = Input(shape=input_shape)
    norm_inp = BatchNormalization()(inp)
    img_1 = Convolution2D(16, kernel_size=3, activation=activations.relu, padding="same")(norm_inp)
    img_1 = Convolution2D(16, kernel_size=3, activation=activations.relu, padding="same")(img_1)
    img_1 = MaxPooling2D(pool_size=(3, 3))(img_1)
    img_1 = Dropout(rate=0.2)(img_1)
    img_1 = Convolution2D(32, kernel_size=3, activation=activations.relu, padding="same")(img_1)
    img_1 = Convolution2D(32, kernel_size=3, activation=activations.relu, padding="same")(img_1)
    img_1 = MaxPooling2D(pool_size=(3, 3))(img_1)
    img_1 = Dropout(rate=0.2)(img_1)
    img_1 = Convolution2D(64, kernel_size=2, activation=activations.relu, padding="same")(img_1)
    img_1 = Convolution2D(20, kernel_size=2, activation=activations.relu, padding="same")(img_1)
    img_1 = GlobalMaxPool2D()(img_1)
    img_1 = Dropout(rate=0.2)(img_1)
    dense_1 = Dense(128, activation=activations.relu)(img_1)
    dense_1 = Dense(nclass, activation=activations.softmax)(dense_1)
 
    model = models.Model(inputs=inp, outputs=dense_1)
    opt = optimizers.Adam()
 
    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
    model.summary()
    return model
model_norm = get_model()
checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
history = model_norm.fit(X_train, y, validation_split=0.1, epochs=100, shuffle=True, verbose=2)
print('history')
# #
#from keras import ModelCheckpoint

#callbacks=[ModelCheckpoint('VGG16-transferlearning.model', monitor='val_acc', save_best_only=True)]

    # Test pretrained model vgg 16 ########
    ########
model = VGG_16('vgg16_weights_tf_dim_ordering_tf_kernels(1).h5')
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')
out = model.predict(X_train)
print (np.argmax(out))
 #### END ########
 
model.load_weights(file_path)
predicts = model.predict(X_test)
predicts = np.argmax(predicts, axis=1)
predicts = [label_index[p] for p in predicts]

df = pd.DataFrame(columns=['fname', 'camera'])
df['fname'] = index
df['camera'] = predicts
df.to_csv("sub.csv", index=False)