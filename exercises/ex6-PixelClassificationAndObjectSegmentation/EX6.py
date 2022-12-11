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
from math import isclose


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



ct = dicom.read_file(r'exercises\ex6-PixelClassificationAndObjectSegmentation\data\Training.dcm')
img = ct.pixel_array
print(img.shape)
print(img.dtype)


#1
io.imshow(img,vmin=0, vmax=150, cmap ='gray')
io.show()


spleen_roi = io.imread(r'exercises\ex6-PixelClassificationAndObjectSegmentation\data\SpleenROI.png')

def ROI_std_mu(Filsti, Image):
    roi = io.imread(Filsti)
    mask = roi > 0
    values = Image[mask]
    std = np.std(values)
    mu = np.average(values)

    return values, std, mu

spleen_values, std_spleen, mu_spleen = ROI_std_mu(r'exercises\ex6-PixelClassificationAndObjectSegmentation\data\SpleenROI.png', img)

print(std_spleen, mu_spleen)


#2

plt.hist(spleen_values, bins=256, range=(np.min(spleen_values), np.max(spleen_values)))
io.show()


#3
n, bins, patches = plt.hist(spleen_values, 60, density=1)
pdf_spleen = norm.pdf(bins, mu_spleen, std_spleen)
plt.plot(bins, pdf_spleen)
plt.xlabel('Hounsfield unit')
plt.ylabel('Frequency')
plt.title('Spleen values in CT scan')
plt.show()


#4

bone_values, std_bone, mu_bone = ROI_std_mu(r'exercises\ex6-PixelClassificationAndObjectSegmentation\data\BoneROI.png', img)

print(std_bone, mu_bone)


min_hu = -200
max_hu = 1000
hu_range = np.arange(min_hu, max_hu, 1.0)
pdf_spleen = norm.pdf(hu_range, mu_spleen, std_spleen)
pdf_bone = norm.pdf(hu_range, mu_bone, std_bone)
plt.plot(hu_range, pdf_spleen, 'r--', label="spleen")
plt.plot(hu_range, pdf_bone, 'g', label="bone")
plt.title("Fitted Gaussians")
plt.legend()
plt.show()




#5
fat_values, std_fat, mu_fat = ROI_std_mu(r'exercises\ex6-PixelClassificationAndObjectSegmentation\data\FatROI.png', img)
kidney_values, std_kidney, mu_kidney = ROI_std_mu(r'exercises\ex6-PixelClassificationAndObjectSegmentation\data\KidneyROI.png', img)
liver_values, std_liver, mu_liver = ROI_std_mu(r'exercises\ex6-PixelClassificationAndObjectSegmentation\data\LiverROI.png', img)


min_hu = -200
max_hu = 1000
hu_range = np.arange(min_hu, max_hu, 1.0)
pdf_spleen = norm.pdf(hu_range, mu_spleen, std_spleen)
pdf_bone = norm.pdf(hu_range, mu_bone, std_bone)
pdf_fat = norm.pdf(hu_range, mu_fat, std_fat)
pdf_kidney = norm.pdf(hu_range, mu_kidney, std_kidney)
pdf_liver = norm.pdf(hu_range, mu_liver, std_liver)
plt.plot(hu_range, pdf_spleen, 'r--', label="spleen")
plt.plot(hu_range, pdf_bone, 'g', label="bone")
plt.plot(hu_range, pdf_fat, 'b', label="fat")
plt.plot(hu_range, pdf_kidney, 'b--', label="kidney")
plt.plot(hu_range, pdf_liver, 'g--', label="liver")
plt.title("Fitted Gaussians")
plt.legend()
plt.show()


#6
#----


#7
mu_soft = (mu_spleen+mu_kidney+mu_liver) / 3
std_soft = (std_spleen+std_kidney+std_liver) / 3

t_backgorund = -200
t_fat = mu_soft / 3 - mu_fat
t_bone = mu_bone - mu_soft

print(t_backgorund, t_fat, t_bone)


#8
background_img = (img < t_backgorund)
fat_img = (img > t_backgorund) & (img <= t_fat)
soft_img = (img > t_fat) & (img <= t_bone)
bone_img = (img > t_bone)

#9
label_img = fat_img + 2 * soft_img + 3 * bone_img
image_label_overlay = label2rgb(label_img)
show_comparison(img, image_label_overlay, 'Classification result')

#10
#----

#11
test_value = 400
if norm.pdf(test_value, mu_soft, std_soft) > norm.pdf(test_value, mu_bone, std_bone):
	print(f"For value {test_value} the class is soft tissue")
else:
	print(f"For value {test_value} the class is bone")

i=-500
while i <=2000:
    pdf_fat = norm.pdf(i, mu_fat, std_fat)
    pdf_soft = norm.pdf(i, mu_soft, std_soft)
    pdf_bone = norm.pdf(i, mu_bone, std_bone)

    if isclose(pdf_fat, pdf_soft, abs_tol=1e-10):
        print("Grænsen mellem fat og soft: ", i)
        i += 1
    
    elif isclose(pdf_soft, pdf_bone, abs_tol=1e-10):
        print("Grænsen mellem soft og bone: ", i)
        i += 1

    i += 1





