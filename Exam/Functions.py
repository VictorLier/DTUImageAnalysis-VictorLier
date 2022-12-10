import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

def Covariance(a, b):
    if len(a) == len(b):
        CoVa = 1/(len(a)-1) * np.sum(np.multiply(a,b))

    else:
        print("Kan ikke regne Covariance. Vektor er ikke samme størrelse")

    return CoVa

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

def GaussLensMM(FocalLength, ObjectDistance, CCDdistance):
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
    PixelWidth = Width/XRes
    PixelHeight = Height/YRes

    return PixelWidth, PixelHeight

def FieldOfView(Focallength, SensorWidth, SensorHeight):
    HorizontalAngle = 2 * (math.atan2((SensorWidth/2),Focallength)) * 180 / math.pi
    VerticalAngle = 2 * (math.atan2((SensorHeight/2),Focallength)) * 180 / math.pi

    return HorizontalAngle, VerticalAngle














