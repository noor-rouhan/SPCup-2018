import numpy as np  # linear algebra
import pandas as pd  # data processi
import os
from PIL import Image
from skimage.transform import resize
from random import shuffle

list_paths = []
for subdir, dirs, files in os.walk("./input"):
    for file in files:
        #print os.path.join(subdir, file)
        filepath = subdir + os.sep + file
        list_paths.append(filepath)
print(list_paths)

list_train = [filepath for filepath in list_paths]
shuffle(list_train)

list_test = [filepath for filepath in list_paths if "/test" in filepath]
# print(list_train)
print(list_test)
list_train = list_train
list_test = list_test
index = [os.path.basename(filepath) for filepath in list_test]
list_classes = list(set([os.path.dirname(filepath).split(os.sep)[-1] for filepath in list_paths if "train" in filepath]))


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
    im_array = np.array(Image.open((filepath)), dtype="uint8")
    pil_im = Image.fromarray(im_array)
    new_array = np.array(pil_im.resize((256, 256)))
    return new_array / 255


def label_transform(labels):
    labels = pd.get_dummies(pd.Series(labels))
    label_index = labels.columns.values

    return labels, label_index
X_train = np.array([read_and_resize(filepath) for filepath in list_train])
print(list_train)