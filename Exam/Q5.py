from Functions import *

img = FileImport(r'Exam\Data\pixelwise.png')

show_in_moved_window("Original",img,0,0)



img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

show_in_moved_window("Grey",img,800,0)



img = histogram_stretch(img,0.1,0.6)

show_in_moved_window("Stretch",img,1200,0)

Thres = Threshold_otsu(img)

print(Thres/255)

img = threshold_image(img,Thres,False)

show_in_moved_window("Threshold",img,1600,0)

cv2.waitKey()
cv2.destroyAllWindows()