import cv2
import matplotlib.pyplot as plt

def FileImport(FilSti):
    im = cv2.imread(FilSti, cv2.IMREAD_ANYDEPTH)

    return im

def ShowImage(Name,Image):

    cv2.namedWindow(Name, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(Name, Image)
    cv2.waitKey()
    cv2.destroyAllWindows()

    return

def Histogram(Image):
    if len(Image.shape) == 2:
        Channels = 0
    else:
        Channels = Image.shape[3]

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
    

