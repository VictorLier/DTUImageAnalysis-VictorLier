from Functions import *

IMG1 = FileImport(r'Exam\Data\change1.png')
IMG2 = FileImport(r'Exam\Data\change2.png')

IMG1 = color.rgb2gray(IMG1)
IMG2 = color.rgb2gray(IMG2)


Diff = np.abs(IMG1-IMG2)

NChangPixel = np.count_nonzero(Diff > 0.3)




print(NChangPixel)

TotalPix = Diff.shape[0] * Diff.shape[1]

percen = NChangPixel / TotalPix * 100

print(percen)


