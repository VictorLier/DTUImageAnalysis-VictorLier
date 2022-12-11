import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
from skimage import io, color, morphology
from skimage import segmentation
from skimage import measure
from skimage.color import label2rgb

def Covariance(a, b):
    # Finder covariance mellem a og b
    if len(a) == len(b):
        CoVa = 1/(len(a)-1) * np.sum(np.multiply(a,b))

    else:
        print("Kan ikke regne Covariance. Vektor er ikke samme størrelse")

    return CoVa

def FileImport(FilSti):
    #Importere billede - husk at starte r'filsti'

    im = cv2.imread(FilSti)

    return im

def show_in_moved_window(win_name, img, x, y):
    """
    Show an image in a window, where the position of the window can be given
    """
    cv2.namedWindow(win_name,cv2.WINDOW_NORMAL)
    cv2.moveWindow(win_name, x, y)
    cv2.imshow(win_name, img)

    '''
    husk 
    cv2.waitKey()
    cv2.destroyAllWindows()
    '''


def Histogram(Image):
    #Finder og viser histogram for billedet

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
        range = [0,1]

    else:  
        print("Billede type eksistere ikke")

    Hist = cv2.calcHist(Image,[Channels], None,histSize, range)


    plt.plot(Hist)
    plt.xlim(range)
    plt.show() 

'''
Kan også gøres sådan:
plt.hist(img_gray.ravel(), bins=256, range=(1, 100))
io.show()
'''

'''
Plot:
io.imshow(img,vmin=0, vmax=150, cmap ='gray')
io.show()
'''




def GaussLensMM(FocalLength, ObjectDistance, CCDdistance):
    #Udregner GaussLens ligningen - alle mål i mm

    f = FocalLength
    g = ObjectDistance 
    b = CCDdistance

    if g != None and b != None:
        f = (g*b)/(b+g)
    
    elif f != None and b != None:
        g = (b*f)/(b-f)

    elif f != None and g != None:
        b = -(g*f)/(f-g)

    else:
        print("Gauss information passer ikke")

    return f, g, b

def ObjectSizeMM(ObjectDistance, CCDdistance, ObjectHeight, ObjectSensorHeight):
    #Udregner størrelse af objekt

    g = ObjectDistance
    b = CCDdistance
    G = ObjectHeight
    B = ObjectSensorHeight

    if b != None and G != None and B != None:
        g = b*G/B

    elif g != None and G != None and B != None:
        b = g*B/G

    elif g != None and b != None and B != None:
        G = g*B/b 

    elif g != None and b != None and G != None:
        B = b*G/g
    
    else:
        print("ObjectSize forket information")

    return g, b, G, B

def PixelSize(XRes, YRes, Width, Height):
    #Finder størrelse i pixels

    PixelWidth = Width/XRes
    PixelHeight = Height/YRes

    return PixelWidth, PixelHeight

def FieldOfView(Focallength, SensorWidth, SensorHeight):
    #Finder Filed of view i grader

    HorizontalAngle = 2 * (math.atan2((SensorWidth/2),Focallength)) * 180 / math.pi
    VerticalAngle = 2 * (math.atan2((SensorHeight/2),Focallength)) * 180 / math.pi

    return HorizontalAngle, VerticalAngle

def MinMax(image):
    #Finder min og mac værdi af billede

    min = np.min(image)
    max = np.max(image)

    return min, max

from skimage.util import img_as_float

from skimage.util import img_as_ubyte

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

def gamma_map(img, gamma):
    #Gamma map. kan tage alle billeder. Kommer ud som float
    if img.dtype == 'float':
        img_float = img
    else:
        img_float = img_as_float(img)

    img_float = np.power(img_float,gamma)

    return img_float

def threshold_image(img_in, thres, INV):
    img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    if img_in.dtype == 'uint8':
        max = 255

    elif img_in.dtype == 'uint16':
        max = 65535
    
    elif img_in.dtype == 'uint32':
        max = 2**32 - 1

    elif img_in.dtype == 'float':
        max = 1

    else:  
        print("Billede type eksistere ikke")
    if INV:
        ret,img = cv2.threshold(img_in,thres,max,cv2.THRESH_BINARY_INV)
    else:
        ret,img = cv2.threshold(img_in,thres,max,cv2.THRESH_BINARY)
    return img_as_ubyte(img)

#Read the documentation of [Otsu's method](https://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.threshold_otsu) and use it to compute and apply a threshold to the vertebra image.
from skimage.filters import threshold_otsu
def Threshold_otsu(Image):
	if len(Image.shape) == 3:
		Image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
	else:
		Image = Image
	return threshold_otsu(Image)

from skimage.filters import median
def MedianFilter(Image, Size):
	if len(Image.shape) == 3:
		Image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
	else:
		Image = Image
	footprint = np.ones([Size,Size])
	Image_out = median(Image,footprint)

	return Image_out

from scipy.ndimage import correlate
def MeanFilter(Image, Size):
	if len(Image.shape) == 3:
		Image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
	else:
		Image = Image

	weights = np.ones([Size, Size])
	Nweihgts = weights / np.sum(weights)

	res_img = correlate(Image, Nweihgts, mode='constant', cval=10)

	return res_img

from skimage.filters import gaussian
def GaussianFilter(Image,sigma):
	if len(Image.shape) == 3:
		Image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
	else:
		Image = Image
	
	gauss_img = gaussian(Image, sigma)

	return gauss_img

from skimage.filters import prewitt_h
def Prewitt_h(Image):
	if len(Image.shape) == 3:
		Image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
	else:
		Image = Image

	return (prewitt_h(Image))	

from skimage.filters import prewitt_v
def Prewitt_v(Image):
	if len(Image.shape) == 3:
		Image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
	else:
		Image = Image

	return (prewitt_v(Image))

from skimage.filters import prewitt
def Prewitt(Image):
	if len(Image.shape) == 3:
		Image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
	else:
		Image = Image

	return (prewitt(Image))	

def plot_comparison(original, filtered, filter_name):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True, sharey=True)
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title('original')
    ax1.axis('off')
    ax2.imshow(filtered, cmap=plt.cm.gray)
    ax2.set_title(filter_name)
    ax2.axis('off')
    io.show()


from skimage.morphology import erosion, dilation, opening, closing
from skimage.morphology import disk
#De skal være GreyScale:
def Erosion(img,DiskSize):
    footprint = disk(DiskSize)
    eroded = erosion(img, footprint)
    return eroded

def Dilation(img, DiskSize):
    footprint = disk(DiskSize)
    dilated = dilation(img, footprint)
    return dilated

def Opening(img, DiskSize):
    footprint = disk(DiskSize)
    opened = opening(img, footprint)
    return opened

def Closing(img, DiskSize):
    footprint = disk(DiskSize)
    closed = closing(img, footprint)
    return closed


from skimage import measure
from skimage.color import label2rgb

def Label(Image):
    label_img = measure.label(Image)
    n_labels = label_img.max()
    print(f"Number of labels: {n_labels}")    
    Overlay = label2rgb(label_img, Image)

    return label_img, Overlay


from skimage import segmentation
segmentation.clear_border(img_bin)

def Areas(label_img):
    region_props = measure.regionprops(label_img)
    areas = np.array([prop.area for prop in region_props])

    return areas

