import cv2
import numpy as np
from numba import vectorize
img = cv2.imread('(HTC-1-M7)1.jpg')
den = cv2.fastNlMeansDenoising(img,3,7,21)
noise = img - den



print(noise)
print('std mean noise')
print(cv2.meanStdDev(noise))
print('mean')

@vectorize(["float32(float32)"],target = 'gpu')
def mean_noise(n):
	return np.std(n)

print('cuda')
print(mean_noise(noise))

#noise1 = noise_mat(cv2.imread(,0))

