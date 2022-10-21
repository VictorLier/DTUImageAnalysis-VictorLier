import numpy as np
import cv2
import matplotlib.pyplot as plt


#1 - Reading image and show hist
#file = "exercises\\ex3-PixelwiseOperations\\data\\vertebra.png"
#im = cv2.imread(file)

#hist = cv2.calcHist([im], [0], None, [255], [0,255])

#plt.figure()
#plt.plot(hist)
#plt.xlim([0,255])
#plt.show()

#2 min max value

#print(np.max(im))
#print(np.min(im))


#3

#from skimage.util import img_as_float
#from skimage.util import img_as_ubyte

#im_float = img_as_float(im)

#print(np.max(im_float))
#print(np.min(im_float))

#4

#im_byte = img_as_ubyte(im_float)

#print(np.max(im_byte))
#print(np.min(im_byte))

#5

'''
def histogram_stretch(img_in):
    img_float = img_as_float(img_in)

    min_val = img_float.min()
    max_val = img_float.max()
    min_desired = 0.0
    max_desired = 1.0

    print("MaxVal Before")
    print(max_val)
    print("MinVal before")
    print(min_val)

    print((max_desired - min_desired) / (max_val - min_val))

    #img_out = (max_desired - min_desired) / (max_val - min_val) * (img_float - min_val) + min_desired

    img_out = np.multiply((img_float - min_val), (max_desired - min_desired) / (max_val - min_val))

    print("MaxVal After")
    print(np.max(img_out))
    print("MinVal After")
    print(np.min(img_out))


    return img_as_ubyte(img_out)
'''
#stretch_im = histogram_stretch(im)
#hist2 = cv2.calcHist([stretch_im], [0], None, [256], [0,256])

#plt.figure()
#plt.plot(hist2)
#plt.xlim([0,255])
#plt.show()


#6

#cv2.imshow('image window', im_float)
# add wait key. window waits until user presses a key
#cv2.waitKey(0)
# and finally destroy/close all open windows
#cv2.imshow('image window', stretch_im)
#cv2.waitKey(0)

#cv2.destroyAllWindows()



#7
'''
#def gamma_map(img, gamma):
    img_float = img_as_float(img)
    img_float = img_float**gamma

    return img_float
'''

#8

#img = gamma_map(im, 2)

#cv2.imshow('image window', img)
# add wait key. window waits until user presses a key
#cv2.waitKey(0)
#cv2.destroyAllWindows()


#9
'''
def threshold_image(img_in, thres):
    img_in[img_in > thres]  = 255
    img_in[img_in < thres] = 0
    img = img_in

    return(img)
'''


# 10

'''
img_thrs = threshold_image(im,230)

cv2.imshow('image window', img_thrs)
# add wait key. window waits until user presses a key
cv2.waitKey(0)
cv2.destroyAllWindows()

'''


#11

from skimage.filters import threshold_otsu
'''
thrs = threshold_otsu(im)

img_thrs = threshold_image(im,thrs)

cv2.imshow('image window', img_thrs)
# add wait key. window waits until user presses a key
cv2.waitKey(0)
cv2.destroyAllWindows()

'''

#12

#ligemeget

#13

file = "exercises\\ex3-PixelwiseOperations\\data\\DTUSigns2.jpg"
im = cv2.imread(file)
'''
cv2.namedWindow("output", cv2.WINDOW_NORMAL)
cv2.imshow("output", im)
cv2.waitKey(0)
'''

def detect_dtu_signs(img_in):
    r_comp = img_in[:, :, 0]
    g_comp = img_in[:, :, 1]
    b_comp = img_in[:, :, 2]

    segm_blue = (r_comp < 10) & (g_comp > 85) & (g_comp < 105) & (b_comp > 180) & (b_comp < 200)

    img_in[(img_in[:, :, 1] > 85) & (img_in[:, :, 1] < 105) & (img_in[:, :, 0] < 10) & (img_in[:, :, 2] > 180) & (img_in[:, :, 2] < 200)] = 255

    img_out = img_in

    return img_out

img = detect_dtu_signs(im)

cv2.namedWindow("output", cv2.WINDOW_NORMAL)
cv2.imshow("output", img)
cv2.waitKey(0)