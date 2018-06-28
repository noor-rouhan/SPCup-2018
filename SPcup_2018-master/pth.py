import cv2
import os
import numpy as np


base = 'C:\\Users\\misbah\\Documents\\Python\\SPcup\\input\\train'
inp = 'C:\\Users\\misbah\\Documents\\Python\\SPcup\\input'
temp = os.listdir('C:\\Users\\misbah\\Documents\\Python\\SPcup\\input\\test')
image = []
threshold = .8


for file in os.listdir(base):
    for im in os.listdir(base+'\\'+file):
        image.append(base+'\\'+file+'\\'+im)



img = cv2.imread(image[0], 0)
template = cv2.imread(inp + '\\test\\' + temp[0], 0)

res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
loc = np.where(res>threshold)[1]
if loc.shape[0]>0:
    print("matched")


