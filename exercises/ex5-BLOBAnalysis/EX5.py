from skimage import io, color, morphology
from skimage.util import img_as_float, img_as_ubyte
import matplotlib.pyplot as plt
import numpy as np
import math
from skimage.filters import threshold_otsu
from skimage import segmentation
from skimage import measure
from skimage.color import label2rgb
import cv2

#1

def FileImport(FilSti):
    #Importere billede - husk at starte r'filsti'

    im = cv2.imread(FilSti)

    return im
#im = FileImport(r'exercises\ex5-BLOBAnalysis\data\lego_4_small.png')

#im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

def Threshold_otsu(Image):
	if len(Image.shape) == 3:
		Image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
	else:
		Image = Image
	return threshold_otsu(Image)
#thres = Threshold_otsu(im)

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
#img_bin = threshold_image(im,thres,True)


def plot_comparison(original, filtered, filter_name):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True, sharey=True)
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title('original')
    ax1.axis('off')
    ax2.imshow(filtered, cmap=plt.cm.gray)
    ax2.set_title(filter_name)
    ax2.axis('off')
    io.show()
#plot_comparison(im,img_bin,'Thres')



#2 
#img_binNo = segmentation.clear_border(img_bin)
#plot_comparison(img_bin,img_binNo,'Border')


#3
from skimage.morphology import erosion, dilation, opening, closing
from skimage.morphology import disk

def Closing(img, DiskSize):
    footprint = disk(DiskSize)
    closed = closing(img, footprint)
    return closed

def Opening(img, DiskSize):
    footprint = disk(DiskSize)
    opened = opening(img, footprint)
    return opened

#closed = Closing(img_binNo,5)
#img_open = Opening(closed,5)

#4

def Label(Image):
    label_img = measure.label(Image)
    n_labels = label_img.max()
    print(f"Number of labels: {n_labels}")    
    Overlay = label2rgb(label_img, Image)
    
    return label_img, Overlay

#label_img, LabelCompare = Label(img_open)

#5

#plot_comparison(im,LabelCompare,'label')


#6
#region_props = measure.regionprops(label_img)
#areas = np.array([prop.area for prop in region_props])
#plt.hist(areas, bins=50)
#plt.show()


#7
'''
import plotly
import plotly.express as px
import plotly.graph_objects as go
from skimage import data, filters, measure, morphology, io, color


def interactive_blobs():
    img_org = io.imread(r'exercises\ex5-BLOBAnalysis\data\lego_4_small.png')
    img = color.rgb2gray(img_org)
    # Binary image, post-process the binary mask and compute labels
    threshold = filters.threshold_otsu(img)
    mask = img < threshold
    mask = morphology.remove_small_objects(mask, 50)
    mask = morphology.remove_small_holes(mask, 50)
    labels = measure.label(mask)

    fig = px.imshow(img, binary_string=True)
    fig.update_traces(hoverinfo='skip') # hover is only for label info

    props = measure.regionprops(labels, img)
    properties = ['area', 'eccentricity', 'perimeter', 'intensity_mean']

    # For each label, add a filled scatter trace for its contour,
    # and display the properties of the label in the hover of this trace.
    for index in range(1, labels.max()):
        label_i = props[index].label
        contour = measure.find_contours(labels == label_i, 0.5)[0]
        y, x = contour.T
        hoverinfo = ''
        for prop_name in properties:
            hoverinfo += f'<b>{prop_name}: {getattr(props[index], prop_name):.2f}</b><br>'
        fig.add_trace(go.Scatter(
            x=x, y=y, name=label_i,
            mode='lines', fill='toself', showlegend=False,
            hovertemplate=hoverinfo, hoveron='points+fills'))

    plotly.io.show(fig)


if __name__ == '__main__':
    interactive_blobs()

'''

#8
img_org = FileImport(r'exercises\ex5-BLOBAnalysis\data\Sample E2 - U2OS DAPI channel.tiff')

def show_in_moved_window(win_name, img, x, y):
    """
    Show an image in a window, where the position of the window can be given
    """
    cv2.namedWindow(win_name)
    cv2.moveWindow(win_name, x, y)
    cv2.imshow(win_name, img)

    '''
    husk 
    cv2.waitKey()
    cv2.destroyAllWindows()
    '''

#show_in_moved_window("Bacteria",img_org,0,0)
#cv2.waitKey()
#cv2.destroyAllWindows()

img_small = img_org[700:1200, 900:1400]
img_gray = img_as_ubyte(img_small) 
#io.imshow(img_gray, vmin=0, vmax=150)
#plt.title('DAPI Stained U2OS cell nuclei')
#io.show()

#plt.hist(img_gray.ravel(), bins=256, range=(1, 100))
#io.show()

#8
thres = Threshold_otsu(img_gray)

img_bin = threshold_image(img_gray, thres,False)

#plot_comparison(img_gray,img_bin,'threshold')

#9

img_binNo = segmentation.clear_border(img_bin)
#plot_comparison(img_binNo,img_binNo,'Border')

label_img, Overlay = Label(img_binNo)

#plot_comparison(img_binNo,Overlay,'label')


#10

def Areas(label_img):
    region_props = measure.regionprops(label_img)
    areas = np.array([prop.area for prop in region_props])

    return areas

areas = Areas(label_img)
print(areas)

print(len(areas))
print(np.min(areas))
print(np.max(areas))

plt.hist(img_gray.ravel(), bins=100, range=(np.min(areas), np.max(areas)))
io.show()

#11
min_area = 55
max_area = 100
region_props = measure.regionprops(label_img)

# Create a copy of the label_img
label_img_filter = label_img
for region in region_props:
	# Find the areas that do not fit our criteria
	if region.area > max_area or region.area < min_area:
		# set the pixels in the invalid areas to background
		for cords in region.coords:
			label_img_filter[cords[0], cords[1]] = 0
# Create binary image from the filtered label image
i_area = label_img_filter > 0
plot_comparison(img_small, i_area, 'Found nuclei based on area')



#12
perimeters = np.array([prop.perimeter for prop in region_props])


#13


f = (4 * math.pi * region.area) / (perimeters**2)

print(f)
