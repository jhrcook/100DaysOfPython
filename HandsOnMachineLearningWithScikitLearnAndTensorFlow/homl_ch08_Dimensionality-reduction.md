# Chapter 8. Dimensionality Reduction


```python
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

np.random.seed(0)

plt.style.use('seaborn-whitegrid')
```


```python
%matplotlib inline
```


```python
# %load_ext ipycache
```

We will consider two main approaches of dimensionality reduction, *project* and *manifold learning*, by learning about **PCA**, **kernel PCA**, and **LLE**.

## The curse of dimensionality

As the number of dimensions grow, the volume grows exponentially.
Therefore, high-dimension data sets will be relatively sparse with most data points being very far away from each other.
This could easily result in overfitting.
Assuming even distribution, the number of training instances required to reach a given density grows exponentially wit the number of dimensions.

## Main approaches for dimensionality reduction

### Projection

In real data sets, the training instances are not evenly distributed throughout the feature space, instead some features contain greater variability than others.
Thus, the training instances lie within a subspace of the feature space.
The number of dimensions can be reduced if the subspace is identified and the data points are projected onto it.

### Manifold learning

A $d$-dimensional manifold is a part of an $n$-dimensional space (where $d<n$)that locally resembles a $d$-dimensional hyperplane.
The *manifold assumption* states that most real-world high-dimensional datasets lie close to a lower-dimensional manifold.

## Principal Component Analysis (PCA)

PCA rotates the axes such that they contain the most variance in the data.
*Singular Value Decomposition* is used to identify these new "principal components."
Below is an example of manually using SVD to find the principal component vectors and project the original data onto the new axes.


```python
n=1000
X = np.array((
    (np.random.randn(n) * 2) + 5,
    (np.random.randn(n) * 0.5) - 2
)).T

# Rotated by 30 degrees
rot = np.pi * 30 / 180
X[:, 0] = X[:, 0] * np.cos(rot) + X[:, 1] * np.sin(rot)
X[:, 1] = X[:, 1] * np.cos(rot) - X[:, 0] * np.sin(rot)

fig = plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], X[:, 1], color='blue', alpha=0.2)
plt.xlabel('$x_1$', fontsize=16)
plt.ylabel('$x_2$', fontsize=16)
plt.title('Rotated 2D data', fontsize=18)
plt.axis('equal')
plt.show()
```


![png](homl_ch08_Dimensionality-reduction_files/homl_ch08_Dimensionality-reduction_5_0.png)



```python
# The data must be centered before SVD.
X_centered = X - X.mean(axis=0)

# SVD of centered data.
U, s, Vt = np.linalg.svd(X_centered)

# First two components.
c1, c2 = Vt.T[:, 0], Vt.T[:, 0]

print(f'compoent 1: {np.round(c1[0], 4)}, {np.round(c1[0], 4)}')
print(f'compoent 2: {np.round(c1[1], 4)}, {np.round(c1[1], 4)}')
```

    compoent 1: -0.8955, -0.8955
    compoent 2: 0.445, 0.445



```python
# Project the data onto the PCs.
X_2d = X_centered.dot(Vt.T)

fig = plt.figure(figsize=(8, 5))
plt.scatter(X_2d[:, 0], X_2d[:, 1], color='blue', alpha=0.2)
plt.xlabel('PC 1', fontsize=16)
plt.ylabel('PC 2', fontsize=16)
plt.title('Rotated 2D data', fontsize=18)
plt.axis('equal')
plt.show()
```


![png](homl_ch08_Dimensionality-reduction_files/homl_ch08_Dimensionality-reduction_7_0.png)


There is also a use `PCA` class in Scikit-Learn.


```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)
```

The `pca` object holds relevant information including the components and the amount of variance they each explain.


```python
pca.components_.T
```




    array([[-0.89551871,  0.44502386],
           [ 0.44502386,  0.89551871]])




```python
pca.explained_variance_ratio_
```




    array([0.96291648, 0.03708352])




```python
pca.mean_
```




    array([ 3.25514434, -3.35372667])




```python
fig = plt.figure(figsize=(8, 5))
plt.scatter(X_2d[:, 0], X_2d[:, 1], color='blue', alpha=0.2)
plt.xlabel('PC 1', fontsize=16)
plt.ylabel('PC 2', fontsize=16)
plt.title('Rotated 2D data', fontsize=18)
plt.axis('equal')
plt.show()
```


![png](homl_ch08_Dimensionality-reduction_files/homl_ch08_Dimensionality-reduction_14_0.png)



```python
arrowprops = {
    'arrowstyle': '->',
    'linewidth': 2,
    'shrinkA': 0, 
    'shrinkB': 0
}

def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    ax.annotate('', v1, v0, arrowprops=arrowprops)

# plot data
fig = plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], X[:, 1], color='blue', alpha=0.2)

for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    draw_vector(pca.mean_, pca.mean_ + v)

plt.xlabel('$x_1$', fontsize=16)
plt.ylabel('$x_2$', fontsize=16)
plt.title('2D data with PCs', fontsize=18)
plt.axis('equal')
plt.show()
```


![png](homl_ch08_Dimensionality-reduction_files/homl_ch08_Dimensionality-reduction_15_0.png)


### Choosing the right number of dimensions


```python

```


```python

```
