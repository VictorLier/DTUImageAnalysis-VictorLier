import numpy as np
import cv2
import matplotlib.pyplot as plt


#1 - Reading image and show hist
file = "exercises\\ex3-PixelwiseOperations\\data\\vertebra.png"
im = cv2.imread(file)

hist = cv2.calcHist([im], [0], None, [255], [0,255])

#plt.figure()
#plt.plot(hist)
#plt.xlim([0,255])
#plt.show()

#2 min max value

#print(np.max(im))
#print(np.min(im))


#3

from skimage.util import img_as_float
from skimage.util import img_as_ubyte

im_float = img_as_float(im)

#print(np.max(im_float))
#print(np.min(im_float))

#4

im_byte = img_as_ubyte(im_float)

#print(np.max(im_byte))
#print(np.min(im_byte))

#5

def histogram_stretch(img_in):
    #img_float = img_as_float(img_in)

    min_val = np.min(img_in)
    max_val = np.max(img_in)
    min_desired = 0
    max_desired = 255

    print(max_val)

    img_out = (max_desired - min_desired) // (max_val - min_val) * (img_in - min_val) + min_desired

    print(np.max(img_out))
    #print(np.min(img_out))


    return (img_out)

stretch_im = histogram_stretch(im)
hist2 = cv2.calcHist([stretch_im], [0], None, [255], [0,255])

plt.figure()
plt.plot(hist2)
plt.xlim([0,255])
plt.show()
