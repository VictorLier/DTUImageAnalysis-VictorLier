import numpy as np
import cv2
from scipy.ndimage import correlate
from skimage.filters import median
from skimage.filters import gaussian
from skimage.util import img_as_float
from skimage.util import img_as_ubyte
import time


'''
input_img = np.arange(25).reshape(5, 5)
print(input_img)

weights = [[0, 1, 0],
		   [1, 2, 1],
		   [0, 1, 0]]


res_img = correlate(input_img, weights)

#1

print(res_img[3,3])


#2 

res_img_cons = correlate(input_img, weights, mode='constant', cval=10)

print(res_img_cons)

res_img_refl = correlate(input_img, weights, mode='reflect', cval=10)

print(res_img_refl)
'''

#3
def FileImport(FilSti):
    #Importere billede - husk at starte r'filsti'

    im = cv2.imread(FilSti)

    return im

im = FileImport(r'exercises\ex4-ImageFiltering\data\Gaussian.png')

def show_in_moved_window(win_name, img, x, y):
    """
    Show an image in a window, where the position of the window can be given
    """
    cv2.namedWindow(win_name,cv2.WINDOW_NORMAL)
    cv2.moveWindow(win_name, x, y)
    cv2.imshow(win_name, img)

    '''
    husk 
    
    '''

show_in_moved_window("Gussian", im, 0,0)


'''
size = 10
weights = np.ones([size, size])
weihgts = weights / np.sum(weights)
print(weights)

res_img = correlate(im, weights, mode='constant', cval=10)

show_in_moved_window("Filtered", res_img, 1000,0)


size = 10
footprint = np.ones([size,size])
med_img = median(im,footprint)

show_in_moved_window("Median", med_img, 1000,0)

cv2.waitKey()
cv2.destroyAllWindows()
'''

#5
im = FileImport(r'exercises\ex4-ImageFiltering\data\SaltPepper.png')

def MedianFilter(Image, Size):
	if len(Image.shape) == 3:
		Image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
	else:
		Image = Image
	footprint = np.ones([Size,Size])
	Image_out = median(Image,footprint)

	return Image_out

def MeanFilter(Image, Size):
	if len(Image.shape) == 3:
		Image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
	else:
		Image = Image

	weights = np.ones([Size, Size])
	Nweihgts = weights / np.sum(weights)

	res_img = correlate(Image, Nweihgts, mode='constant', cval=10)

	return res_img

Med_im = MedianFilter(im,10)

Mean_im = MeanFilter(im,10)

show_in_moved_window("No Filter", im,0,0)
show_in_moved_window("Med filter", Med_im,1000,0)
show_in_moved_window("Mean filter",Mean_im,2000,0)
cv2.waitKey()
cv2.destroyAllWindows()



#6



def GaussianFilter(Image,sigma):
	if len(Image.shape) == 3:
		Image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
	else:
		Image = Image
	
	gauss_img = gaussian(Image, sigma)

	return gauss_img


#7
im = FileImport(r'exercises\ex4-ImageFiltering\data\car.png')
show_in_moved_window("No Filter", im,0,0)

im_med = MedianFilter(im,10)
im_mean = MeanFilter(im,10)
im_gaus = GaussianFilter(im,10)

show_in_moved_window("Median Filter", im_med,400,0)
show_in_moved_window("Mean Filter", im_mean,800,0)
show_in_moved_window("Gaussian Filter", im_gaus,1200,0)

cv2.waitKey()
cv2.destroyAllWindows()



#8

from skimage.filters import prewitt_h
from skimage.filters import prewitt_v
from skimage.filters import prewitt

im = FileImport(r'exercises\ex4-ImageFiltering\data\donald_1.png')
show_in_moved_window("No Filter", im,0,0)


def Prewitt_h(Image):
	if len(Image.shape) == 3:
		Image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
	else:
		Image = Image

	return (prewitt_h(Image))	

def Prewitt_v(Image):
	if len(Image.shape) == 3:
		Image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
	else:
		Image = Image

	return (prewitt_v(Image))	

def Prewitt(Image):
	if len(Image.shape) == 3:
		Image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
	else:
		Image = Image

	return (prewitt(Image))	


im_prewh = Prewitt_h(im)
im_prewv = Prewitt_v(im)
im_prew = Prewitt(im)



#9
show_in_moved_window("Prewh Filter", im_prewh,400,0)
show_in_moved_window("Prewv Filter", im_prewv,800,0)
show_in_moved_window("Prew Filter", im_prew,1200,0)

cv2.waitKey()
cv2.destroyAllWindows()


#
im = FileImport(r'exercises\ex4-ImageFiltering\data\ElbowCTSlice.png')
show_in_moved_window("No Filter", im,0,0)

im_gaus = GaussianFilter(im,3)

im_prew = Prewitt(im_gaus)

from skimage.filters import threshold_otsu

def Threshold_otsu(Image):
	if len(Image.shape) == 3:
		Image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
	else:
		Image = Image
	return threshold_otsu(Image)
 
thres = Threshold_otsu(im_prew)
print(thres)


def threshold_image(img_in, thres):
    if img_in.dtype == 'uint8':
        max = 255
        min = 0

    elif img_in.dtype == 'uint16':
        max = 65535
        min = 0
    
    elif img_in.dtype == 'uint32':
        max = 2**32 - 1
        min = 0

    elif img_in.dtype == 'float':
        max = 1
        min = 0

    else:  
        print("Billede type eksistere ikke")

    img_in[img_in > thres] = max
    img_in[img_in < thres] = min
    img = img_in

    return img_as_ubyte(img)

thres_im =threshold_image(im_prew,thres)
show_in_moved_window("Masse filtre", thres_im,400,0)

cv2.waitKey()
cv2.destroyAllWindows()



