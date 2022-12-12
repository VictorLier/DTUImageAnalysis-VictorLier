from Functions import * 

img = FileImport(r'Exam\Data\rocket.png')

img = GaussianFilter(img,3)

img = img_as_ubyte(img)

print(img[100,100])