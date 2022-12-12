from Functions import *
import matplotlib.pyplot as plt
import math
from skimage.transform import rotate
from skimage.transform import EuclideanTransform
from skimage.transform import SimilarityTransform
from skimage.transform import warp
from skimage.transform import swirl


img = FileImport(r'Exam\Data\CPHSun.png')


img = rotate(img,16,False,(20,20))

img = img_as_ubyte(img)

print(img[200,200])