import numpy as np
import cv2
import matplotlib.pyplot as plt


#1 - Reading image and show hist
file = "exercises\\ex3-PixelwiseOperations\\data\\vertebra.png"
im = cv2.imread(file)

hist = cv2.calcHist([im], [0], None, [255], [0,255])

plt.figure()
plt.plot(hist)
plt.xlim([0,255])
plt.show()

#2 min max value

print(np.max(im))
print(np.min(im))


#3

from skimage.util import img_as_float
from skimage.util import img_as_ubyte

im_float = img_as_float(im)

print(np.max(im_float))
print(np.min(im_float))

#4

im_byte = img_as_ubyte(im_float)

print(np.max(im_byte))
print(np.min(im_byte))

#5
