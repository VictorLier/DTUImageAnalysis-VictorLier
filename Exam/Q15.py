import numpy as np
from skimage import io, color
from skimage.morphology import binary_closing, binary_opening
from skimage.morphology import disk
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure
from skimage.color import label2rgb
import pydicom as dicom
from scipy.stats import norm
from scipy.spatial import distance

Cows = [26, 46, 33, 23, 35, 28, 21, 30, 38, 43]
Sheep = [67, 27, 40, 60, 39, 45, 27, 67, 43, 50, 37, 100]


std_Cows = np.std(Cows)
std_Sheep = np.std(Sheep)

mu_Cows = np.average(Cows)
mu_Sheep = np.average(Sheep)


test_value = 38



min = 0
max = 100
range = np.arange(min, max, 1.0)
pdf_Cows = norm.pdf(range, mu_Cows, std_Cows)
pdf_Sheep = norm.pdf(range, mu_Sheep, std_Sheep)
plt.plot(range, pdf_Cows, 'r--', label="Cows")
plt.plot(range, pdf_Sheep, 'g', label="Sheep")
plt.title("Fitted Gaussians")
plt.legend()
plt.show()

print(((mu_Sheep-mu_Cows) / 2) + mu_Cows)