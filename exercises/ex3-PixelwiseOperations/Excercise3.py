import numpy as np
import cv2
import matplotlib.pyplot as plt


#1 - Reading image and show hist

def FileImport(FilSti):
    im = cv2.imread(FilSti)

    return im

im = FileImport(r'exercises\ex3-PixelwiseOperations\data\vertebra.png')

def show_in_moved_window(win_name, img, x, y):
    """
    Show an image in a window, where the position of the window can be given
    """
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(win_name, x, y)
    cv2.imshow(win_name, img)


#show_in_moved_window("Vertavra", im, 0, 0)
#cv2.waitKey()
#cv2.destroyAllWindows()

def Histogram(Image):
    if len(Image.shape) == 2:
        Channels = 0
    else:
        Channels = Image.shape[2]

    if Image.dtype == 'uint8':
        histSize = [255]
        range = [0,255]

    elif Image.dtype == 'uint16':
        histSize = [65535]
        range = [0,65535]
    
    elif Image.dtype == 'uint32':
        histSize = [2**32 - 1]
        range = [0,2**32 - 1]

    elif Image.dtype == 'float':
        histSize = [1]
        range = [0]

    else:  
        print("Billede type eksistere ikke")

    Hist = cv2.calcHist(Image,[Channels], None,histSize, range)


    plt.plot(Hist)
    plt.xlim(range)
    plt.show() 

#Histogram(im)



#2 min max value

def MinMax(image):
    min = np.min(image)
    max = np.max(image)

    return min, max

min, max = MinMax(im)

print(min, max)

#3

from skimage.util import img_as_float
from skimage.util import img_as_ubyte

im_float = img_as_float(im)

print(np.max(im_float))
print(np.min(im_float))

#4

im_byte = img_as_ubyte(im_float)

print(MinMax(im_byte))

print(im_byte.dtype)
#5

def histogram_stretch(img):
    #Convert til float
    if img.dtype == 'float':
        img_in = img
    else:
        img_in = img_as_float(img)

    min_desired = 0
    max_desired = 1

    #Min of max værdi findes
    min_val, max_val = MinMax(img_in)

    print("MinVal before: ", min_val)
    print("MaxVal Before: ", max_val)


    img_out = (max_desired - min_desired) / (max_val - min_val) * (img_in - min_val) + min_desired

    Min_after, Max_after = MinMax(img_out)

    print("MinVal After: ", Min_after)
    print("MaxVal After: ", Max_after)

    return img_as_ubyte(img_out)

im_stretch = histogram_stretch(im)

#Histogram(im_stretch)





#6

#show_in_moved_window("Normal",im,400,0)
#show_in_moved_window("Stretch", im_stretch,0,0)

#cv2.waitKey()
#cv2.destroyAllWindows()


#7

def gamma_map(img, gamma):
    img_float = img_as_float(img)
    img_float = np.power(img_float,gamma)

    return img_float

#8

img = gamma_map(im, 2)
#show_in_moved_window("Gamma = 2", img, 0,0)

img = gamma_map(im, 0.5)
#show_in_moved_window("Gamma = 0.5", img, 1000,0)

#show_in_moved_window("ingen map",im,2000,0)

#cv2.waitKey()
#cv2.destroyAllWindows()



#9
def threshold_image(img_in, thres):
    img_in[img_in > thres] = 255
    img_in[img_in < thres] = 0
    img = img_in

    return img_as_ubyte(img)

# 10


img_thrs = threshold_image(im,160)

#show_in_moved_window("Threshold", img_thrs,0,0)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


#11

from skimage.filters import threshold_otsu
thrs = threshold_otsu(im)
img_thrs = threshold_image(im,thrs)

#show_in_moved_window("Selv fundne thres",img_thrs,0,0)
#cv2.waitKey(0)
#cv2.destroyAllWindows()




#12

#ligemeget

#13

im = FileImport(r'exercises\ex3-PixelwiseOperations\data\DTUSigns2.jpg')

show_in_moved_window("Skilt",im,0,0)
cv2.waitKey(0)
cv2.destroyAllWindows()


def detect_dtu_signs(img_in):
    r_comp = img_in[:, :, 0]
    g_comp= img_in[:, :, 1]
    b_comp = img_in[:, :, 2]

    segm_blue = (r_comp < 10) & (g_comp > 85) & (g_comp < 105) & (b_comp > 180) & (b_comp < 200)

    red = r_comp*segm_blue
    green = g_comp*segm_blue
    blue = b_comp*segm_blue

    img_in[:, :, 0] = red
    img_in[:, :, 1] = green
    img_in[:, :, 2] = blue

    img_out = img_in

    return img_as_ubyte(img_out)

img = detect_dtu_signs(im)
show_in_moved_window("Blå skilt",img,0,0)
cv2.waitKey(0)
cv2.destroyAllWindows()




#14

