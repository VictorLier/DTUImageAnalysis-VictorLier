import numpy as np 
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt 
from sklearn import decomposition

carData = np.loadtxt(r'Exam\Data\car_data.txt',comments="%")
x = carData

i = 0
while i<=7:
    carData[:,i] = carData[:,i] / np.std(carData[:,i])
    i += 1


mn = np.mean(carData, axis=0)
data = carData - mn

print(data[0,0])


pca = decomposition.PCA()
pca.fit(x)
values_pca = pca.explained_variance_
exp_var_ratio = pca.explained_variance_ratio_
vectors_pca = pca.components_
data_transform = pca.transform(x)


data_transform2 = data_transform[:,[1,2,3]]

print(data_transform2.shape)

plt.figure() 
d = pd.DataFrame(data_transform2, columns=['PC1', 'PC2', 'PC3'])
sns.pairplot(d)
plt.show()

print(vectors_pca[0,0])
