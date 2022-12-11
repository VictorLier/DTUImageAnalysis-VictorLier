from skimage.morphology import erosion, dilation, opening, closing
from skimage.morphology import disk
import cv2
from skimage.util import img_as_float
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt
from skimage import io
import numpy as np

def plot_comparison(original, filtered, filter_name):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True, sharey=True)
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title('original')
    ax1.axis('off')
    ax2.imshow(filtered, cmap=plt.cm.gray)
    ax2.set_title(filter_name)
    ax2.axis('off')
    io.show()


#1

def FileImport(FilSti):
    #Importere billede - husk at starte r'filsti'

    im = cv2.imread(FilSti)

    return im

im = FileImport(r'exercises\ex4b-ImageMorphology\data\lego_5.png')

im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

from skimage.filters import threshold_otsu
def Threshold_otsu(Image):
	if len(Image.shape) == 3:
		Image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
	else:
		Image = Image
	return threshold_otsu(Image)

thres = Threshold_otsu(im)

def threshold_image(img_in, thres):
    imgcopy = img_in.copy()
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
    
    imgcopy[imgcopy > thres] = max
    imgcopy[imgcopy < thres] = min

    return img_as_ubyte(imgcopy)

bin_img = threshold_image(im, thres)

#plot_comparison(im, bin_img, "otsu")


#2

def Erosion(img,DiskSize):
    footprint = disk(DiskSize)
    eroded = erosion(img, footprint)
    return eroded

footprint = disk(4)
print(footprint)

eroded = Erosion(bin_img, 4)
#plot_comparison(bin_img, eroded, 'erosion')


#3

def Dilation(img, DiskSize):
    footprint = disk(DiskSize)
    dilated = dilation(img, footprint)
    return dilated

dilated = Dilation(bin_img, 4)
#plot_comparison(bin_img,dilated,'dilated')

#4
def Opening(img, DiskSize):
    footprint = disk(DiskSize)
    opened = opening(img, footprint)
    return opened

opened = Opening(bin_img,4)

#plot_comparison(bin_img,dilated,'Opened')

#5

def Closing(img, DiskSize):
    footprint = disk(DiskSize)
    closed = closing(img, footprint)
    return closed

closed = Closing(bin_img,4)
#plot_comparison(bin_img,closed,'Closed')



#6
from skimage.filters import prewitt
def Prewitt(Image):
	if len(Image.shape) == 3:
		Image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
	else:
		Image = Image

	return (prewitt(Image))	

Outline = Prewitt(bin_img)
#plot_comparison(bin_img,Outline,'Outline')


#7
opened = Opening(bin_img,1)
closed = Closing(opened,15)
opg7 = Prewitt(closed)

#plot_comparison(bin_img,opg7,'Multi')


#8
im = FileImport(r'exercises\ex4b-ImageMorphology\data\lego_7.png')

im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
thres = Threshold_otsu(im)
bin_img = threshold_image(im,thres)

#plot_comparison(im,bin_img,'threshold')

Outline = Prewitt(bin_img)

#plot_comparison(bin_img,Outline,'Outline')

#9
Closed = Closing(bin_img,3)

ClosedOutline = Prewitt(Closed)
#plot_comparison(Outline,ClosedOutline,'Close Outline')

#10
im = FileImport(r'exercises\ex4b-ImageMorphology\data\lego_3.png')
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
thres = Threshold_otsu(im)
bin_img = threshold_image(im,thres)
Outline = Prewitt(bin_img)
Closed = Closing(bin_img,2)
ClosedOutline = Prewitt(Closed)
#plot_comparison(Outline,ClosedOutline,'Close Outline')


#11

im = FileImport(r'exercises\ex4b-ImageMorphology\data\lego_9.png')
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#thres = Threshold_otsu(im)
#bin_img = threshold_image(im,thres)
#plot_comparison(im,bin_img,'Binary')

#Outline = Prewitt(bin_img)
#plot_comparison(bin_img,Outline,'Ouline')

#12
#closed = Closing(bin_img,10)
#ClosedOutline = Prewitt(closed)
#plot_comparison(Outline,ClosedOutline,'ClosedOutline')


#13
#eroded = Erosion(closed,10)
#ErodedOutline = Prewitt(eroded)
#plot_comparison(ClosedOutline,ErodedOutline,'ErodedClosedOutline')


#15
im = FileImport(r'exercises\ex4b-ImageMorphology\data\puzzle_pieces.png')
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
thres = Threshold_otsu(im)
bin_img = threshold_image(im,thres)
plot_comparison(im,bin_img,'Binary')

opened = Opening(bin_img,20)

Outline = Prewitt(opened)

plot_comparison(bin_img,Outline,'Outline')
