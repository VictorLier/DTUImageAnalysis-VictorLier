import matplotlib.pyplot as plt
import math
from skimage.transform import rotate
from skimage.transform import EuclideanTransform
from skimage.transform import SimilarityTransform
from skimage.transform import warp
from skimage.transform import swirl
import numpy as np
from Functions import *

src_img = FileImport(r'Exam\Data\rocket.png')

src = np.array([[220, 55], [105, 675], [315, 675]])
dst = np.array([[100,165], [200,605], [379,525]])


plt.imshow(src_img)
plt.plot(src[:, 0], src[:, 1], '.r', markersize=12)
plt.show()


e_x = src[:, 0] - dst[:, 0]
error_x = np.dot(e_x, e_x)
e_y = src[:, 1] - dst[:, 1]
error_y = np.dot(e_y, e_y)
f = error_x + error_y
print(f"Landmark alignment error F: {f}")

tform = EuclideanTransform()
tform.estimate(src, dst)

#src_transform = matrix_transform(src, tform.params)

warped = warp(src_img, tform.inverse)

warped = img_as_ubyte(warped)

print(warped[150,150])

WapredPoints = warp(src, tform)

e_x = src[:, 0] - dst[:, 0]
error_x = np.dot(e_x, e_x)
e_y = src[:, 1] - dst[:, 1]
error_y = np.dot(e_y, e_y)
f = error_x + error_y
print(f"Landmark alignment error F: {f}")



plt.imshow(warped)
plt.plot(WapredPoints[:, 0], WapredPoints[:, 1], '.r', markersize=12)
plt.show()