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


def show_comparison(original, modified, modified_name):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                   sharey=True)
    ax1.imshow(original, cmap="gray", vmin=-200, vmax=500)
    ax1.set_title('Original')
    ax1.axis('off')
    ax2.imshow(modified)
    ax2.set_title(modified_name)
    ax2.axis('off')
    io.show()



ct = dicom.read_file(r'Exam\Data\1-162.dcm')
img = ct.pixel_array
print(img.shape)
print(img.dtype)

io.imshow(img,vmin=0, vmax=150, cmap ='gray')
io.show()


def ROI_std_mu(Filsti, Image):
    roi = io.imread(Filsti)
    mask = roi > 0
    values = Image[mask]
    std = np.std(values)
    mu = np.average(values)

    return values, std, mu


liver_values, std_liver, mu_liver = ROI_std_mu(r'Exam\Data\LiverROI.png',img)
kidney_values, std_kidney, mu_kidney = ROI_std_mu(r'Exam\Data\KidneyROI.png',img)
aorta_values, std_aorta, mu_aorta = ROI_std_mu(r'Exam\Data\AortaROI.png',img)


min_hu = -200
max_hu = 1000
hu_range = np.arange(min_hu, max_hu, 1.0)
pdf_aorta = norm.pdf(hu_range, mu_aorta, std_aorta)
pdf_kidney = norm.pdf(hu_range, mu_kidney, std_kidney)
pdf_liver = norm.pdf(hu_range, mu_liver, std_liver)
plt.plot(hu_range, pdf_aorta, 'b', label="aorta")
plt.plot(hu_range, pdf_kidney, 'b--', label="kidney")
plt.plot(hu_range, pdf_liver, 'g--', label="liver")
plt.title("Fitted Gaussians")
plt.legend()
plt.show()






















test_value = 203.3

print(norm.pdf(test_value, mu_aorta, std_aorta), norm.pdf(test_value, mu_kidney, std_kidney))

test_value = 145.7
#135.5 - 0.007731951497272039 0.007978073872141056
print(norm.pdf(test_value, mu_liver, std_liver), norm.pdf(test_value, mu_kidney, std_kidney))


