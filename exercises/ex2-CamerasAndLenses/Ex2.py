import math

#1

a = 10
b = 3

theta = math.atan2(b,a)

degree = theta *  180 / math.pi

print(theta)
print(degree)



#2

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


OBDistanceM = [0.1, 1, 5, 15]

OBDistanceMM = [i*1000 for i in OBDistanceM]

print(OBDistanceMM)


Focallength = 15

output = [GaussLensMM(Focallength, i, None) for i in OBDistanceMM]

print(output)


#3

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

Focallength = 5

OBDistanceMM = 5000

Focallength, OBDistanceMM, CCDdistance = GaussLensMM(Focallength, OBDistanceMM, None)
#1
print(CCDdistance)


#2
ObjectHeight = 1800
OBDistanceMM, CCDdistance, ObjectHeight, ObjectSensorHeight = ObjectSizeMM(OBDistanceMM, CCDdistance, ObjectHeight, None)
print(ObjectSensorHeight)


#3

def PixelSize(XRes, YRes, Width, Height):
    PixelWidth = Width/XRes
    PixelHeight = Height/YRes

    return PixelWidth, PixelHeight

PixelSize = PixelSize(640,480,6.4,4.8)

print(PixelSize)

#4

ThomasHeightOnSensor = ObjectSensorHeight/PixelSize[1]

print(ThomasHeightOnSensor)


#5

def FieldOfView(Focallength, SensorWidth, SensorHeight):
    HorizontalAngle = 2 * (math.atan2((SensorWidth/2),Focallength)) * 180 / math.pi
    VerticalAngle = 2 * (math.atan2((SensorHeight/2),Focallength)) * 180 / math.pi

    return HorizontalAngle, VerticalAngle

Vinkel = FieldOfView(Focallength, 7, 4.8)

print(Vinkel)






