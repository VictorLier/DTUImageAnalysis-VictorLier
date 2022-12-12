from Functions import *
from skimage import segmentation


img = FileImport(r'Exam\Data\figures.png')

Thres = Threshold_otsu(img)

print(Thres)

ThresImage = threshold_image(img, Thres, True)

show_in_moved_window("ThresImage",ThresImage,0,0)

UdenKant = segmentation.clear_border(ThresImage)

show_in_moved_window("KantFjerne",UdenKant,800,0)

Label_img, Overlay = Label(UdenKant)

show_in_moved_window("MedBlob",Overlay,1200,0)


BlobArea = Areas(Label_img)

print(BlobArea)

BigLoc = BlobArea.argmax(axis=0)
print(BigLoc)


Perim = Perimeter(Label_img)

print(len(Perim))
print(Perim[BigLoc])


SumOver = (BlobArea > 13000).sum()
print(SumOver)



cv2.waitKey()
cv2.destroyAllWindows()
