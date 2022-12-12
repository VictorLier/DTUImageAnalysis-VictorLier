from Functions import *

img = FileImport(r'Exam\Data\pixelwise.png')

img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

img_S = img_HSV[:,:,1]

show_in_moved_window("S",img_S,0,0)

Thres = Threshold_otsu(img_S)

print(Thres)

ThresImage = threshold_image(img_S, Thres,False)

show_in_moved_window("Threshold",ThresImage,800,0)


Eroded = Erosion(ThresImage,4)

show_in_moved_window("Eroded",Eroded,1200,0)

print(np.count_nonzero(Eroded == 255))



cv2.waitKey()
cv2.destroyAllWindows()