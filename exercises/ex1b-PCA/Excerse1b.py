import numpy as np
import matplotlib.pyplot as plt


#1

iris_data = np.loadtxt("exercises\ex1b-PCA\data\irisdata.txt", comments="%")

x = iris_data[0:50, 0:4]

n_feat = x.shape[1]
n_obs = x.shape[0]
print(f"Number of features: {n_feat} and number of observations: {n_obs}")


#2
sep_l = x[:, 0]
sep_w = x[:, 1]
pet_l = x[:, 2]
pet_w = x[:, 3]

Var_sep_1 = sep_l.var(ddof=1)
Var_sep_w = sep_w.var(ddof=1)
Var_pet_1 = pet_l.var(ddof=1)
Var_pet_w = pet_w.var(ddof=1)


#3
def Covariance(a, b):
    if len(a) == len(b):
        CoVa = 1/(len(a)-1) * np.sum(np.multiply(a,b))

    else:
        print("Kan ikke regne Covariance. Vektor er ikke samme størrelse")

    return CoVa


print(Covariance(sep_l,pet_w))

print(Covariance(sep_l, sep_w))



#4

import seaborn as sns
import pandas as pd

plt.figure()

d = pd.DataFrame(x, columns=['Sepal length', 'Sepal width',
							 'Petal length', 'Petal width'])
sns.pairplot(d)
plt.show()



#5

mn = np.mean(x, axis =0)
data = x -mn

c_x = np.cov(x)

print(c_x)

#6

values, vectors = np.linalg.eig(c_x)

#7
v_norm = values / values.sum() * 100
plt.plot(v_norm)
plt.xlabel('Principal component')
plt.ylabel('ercent explained variance')
plt.ylim([0, 100])

plt.show()


#8




#9
from sklearn import decomposition

pca = decomposition.PCA()
pca.fit(x)
values_pca = pca.explained_variance_
exp_var_ratio = pca.explained_variance_ratio_
vectors_pca = pca.components_

data_transform = pca.transform(data)
