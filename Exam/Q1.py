from Functions import *

im = FileImport("exercises\ex3-PixelwiseOperations\data\dark_background.png")

ShowImage('Billede1', im)

print(im.dtype)

print(len(im.shape))

Histogram(im)

