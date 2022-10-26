import numpy as np
from scipy.ndimage import correlate

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

file = "exercises\\ex4-ImageFiltering\\data\\Gaussian.png"
im = cv2.imread(file)




size = 5

weights = np.ones([size, size])

weihgts = weights / np.sum(weights)