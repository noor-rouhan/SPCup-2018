import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

base = 'C:\\Users\\misbah\\Documents\\Python\\SPcup\\input\\train'
inp = 'C:\\Users\\misbah\\Documents\\Python\\SPcup\\input'
temp = os.listdir('C:\\Users\\misbah\\Documents\\Python\\SPcup\\input\\test')
image = []
threshold = .8
for file in os.listdir(base):
    for im in os.listdir(base+'\\'+file):
        image.append(base+'\\'+file+'\\'+im)

for im in image:
    i = cv2.imread(im, 0)

    tm = cv2.imread('template.jpg',0)
    res = cv2.matchTemplate(i, tm, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res>threshold)[1]

    if loc.shape[0]>0:
        print(im)
        print("matched")



# print(image)