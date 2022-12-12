from Functions import *
from skimage.filters import prewitt
import numpy as np

img = FileImport(r'Exam\Data\rocket.png')

show_in_moved_window("Racket", img, 0,0)


img_in = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

Outline = prewitt(img_in)

show_in_moved_window("Outline",Outline, 200,0)

threshold = threshold_image(Outline, 0.06,False)

show_in_moved_window("threshold", threshold, 800 ,0)
cv2.waitKey()
cv2.destroyAllWindows()

print(threshold)

print(np.max(threshold))

print(np.count_nonzero(threshold == 255))
