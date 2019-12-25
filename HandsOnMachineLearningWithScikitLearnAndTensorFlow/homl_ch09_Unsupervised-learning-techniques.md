# Chapter 9. Unsupervised Learning Techniques




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
%load_ext ipycache
```

    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/IPython/config.py:13: ShimWarning: The `IPython.config` package has been deprecated since IPython 4.0. You should import from traitlets.config instead.
      "You should import from traitlets.config instead.", ShimWarning)
    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/ipycache.py:17: UserWarning: IPython.utils.traitlets has moved to a top-level traitlets package.
      from IPython.utils.traitlets import Unicode


This chapter covers the following unsupervised learning techniques: clustering, anomaly detection, and density estimation.

## Clustering

*Clustering* is the task of identifying similar instances and assigning them to groups.

The author discusses some useful tasks for clustering:

1. At the beginning of a data analysis, the data can be clustered and then each cluster can be analyzed separately.
2. Once clusters have been assigned, the affinity values for each cluster can be used for dimensionality reduction.
3. Anomalies can be detected as those that don't fit into a cluster.
4. Label propagation from labeled data to new unlabeled data for semi-supervised learning.

### K-means

Below is an example of K-means clustering on artificial data made of blobs.


```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Artifical data.
X, y = make_blobs(n_samples=500, centers=5, random_state=0)

# K-means clustering.
k = 5  # The number of clusters to find.
kmeans = KMeans(n_clusters=k)
kmeans.fit(X)

# Make classifications from K-means model.
y_pred = kmeans.predict(X)


# Plotting.
fig = plt.figure(figsize=(12, 6))

# Plot the original clusters.
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='Set1')
plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.title('Original', fontsize=16)

# Plot the K-means classifications.
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='Set2')
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], c='k', s=70, marker='x')

plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.title(f'K-Means prediction ($k={k}$)', fontsize=16)
plt.show()
```


![png](homl_ch09_Unsupervised-learning-techniques_files/homl_ch09_Unsupervised-learning-techniques_5_0.png)



```python
# Show the centers of the clusters.
kmeans.cluster_centers_
```




    array([[-1.78783991,  2.76785611],
           [ 9.30286933, -2.23802673],
           [-1.33625465,  7.73822965],
           [ 1.87544954,  0.76337636],
           [ 0.87407478,  4.4332834 ]])




```python
# Create Voronoi cells by predicting over a mesh.
h = .02  # step size of the mesh grid.
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
x0, x1 = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X_mesh = np.c_[x0.ravel(), x1.ravel()]
y_pred = kmeans.predict(X_mesh).reshape(x0.shape)

# Plot Voronoi cells.
fig = plt.figure(figsize=(8, 5))

plt.contourf(x0, x1, y_pred, cmap='Set2', alpha=0.2)

plt.scatter(X[:, 0], X[:, 1], c='#bdbdbd', s=10)

plt.scatter(centroids[:, 0], centroids[:, 1], c='k', s=200, marker='.')

plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.title(f'K-Means Voronoi cells', fontsize=16)
plt.show()
```


![png](homl_ch09_Unsupervised-learning-techniques_files/homl_ch09_Unsupervised-learning-techniques_7_0.png)


The performance of K-means clustering degrades when the distributions of the clusters are different.
*Hard-voting* is when each instance is assigned to a single cluster.
Alternatively, *soft-voting* is when a score is assigned to each instance per centroid.
The score can be the distance to the centroid, or a similarity score (affinity) from the Gaussian radial basis function (RBF).
These data can be obtained by using the `transform()` method of a `KMeans` object.
This can act as a dimensionality reduction technique for higher-dimensional data.


```python
kmeans.transform(X[0:5, :])
```




    array([[ 5.65395395,  6.78719721,  9.32888063,  1.65547831,  5.42241143],
           [ 2.29711611, 11.50756245,  3.59157417,  4.11349809,  1.00516713],
           [ 5.24356985, 10.50404117,  4.31832326,  5.04967736,  2.11385281],
           [ 0.72872131, 12.22263029,  4.29719996,  4.32525956,  2.58848823],
           [ 1.71946847, 10.68298997,  5.01631429,  2.87918881,  1.81565044]])



Centroid initialization plays a role in determining the final outcome.
The original method for initialization was to randomly distribute the centroids.
Now, Scikit-Learn's `KMeans` class uses an improved method called *K-Means++* that finds initial locations that are well-distributed in relation to the density of the data.

In addition, the class has an `n_init` property that determines how many times the algorithm is run (the default is 10).
After a number of rounds, the best version is kept.
The quality of the iteration is the *inertia* of the model, that is the MSE of each instance and the nearest centroid.

Scikit-Learn also offers the `MiniBatchKMeans` class that uses a variation of the K-means algorithm to fit the model in batches.
This makes training with a large data set faster.

### Finding the optimal number of clusters

We cannot use the inertia as the metric for the number of clusters because it will always decrease as $k$ increases.
This is shown below.


```python
ks = np.arange(2, 11)
intertias = []

for k in ks:
    km = KMeans(n_clusters=k)
    km.fit(X)
    intertias.append(km.inertia_)

plt.plot(ks, intertias, 'k--o')

plt.annotate('elbow', xy=(5, intertias[5]+200),
             xycoords='data', xytext=(0.6, 0.5), 
             textcoords='axes fraction', 
             arrowprops=dict(width=0.1, headwidth=8, facecolor='black', shrink=0.05),
             horizontalalignment='right', verticalalignment='top',
             fontsize=14)

plt.xlabel('$k$')
plt.ylabel('inertia')
plt.show()
```


![png](homl_ch09_Unsupervised-learning-techniques_files/homl_ch09_Unsupervised-learning-techniques_11_0.png)


The elbow point can be a good indication for the number of clusters, though it is coarse, manual, and imprecise.

Alternatively, the *silhouette score*, the mean of the *silhouette coefficients* for each point, can be a good measure.
An instance's silhouette coefficient is shown below where $a$ is the means distance to the other instances in the same cluster and $b$ is the mean distance to the instances of the next nearest cluster (using the minimum value of all the clusters for $b$).
The value can range from -1 to 1 where the greater the number the more likely the point was assigned correctly.

$$
\frac{b-a}{max(a, b)}
$$

Thankfully, Scikit-Learn offers the `silhouette_score()` function to compute this.


```python
from sklearn.metrics import silhouette_score

ks = np.arange(2, 11)
sil_scores = []

for k in ks:
    km = KMeans(n_clusters=k)
    km.fit(X)
    sil_scores.append(silhouette_score(X, km.labels_))

plt.plot(ks, sil_scores, 'k--o')
plt.xlabel('$k$')
plt.ylabel('silhouette score')
plt.show()
```


![png](homl_ch09_Unsupervised-learning-techniques_files/homl_ch09_Unsupervised-learning-techniques_13_0.png)


Below are *silhouette diagrams* for multiple K-means models with various $k$ values.
The plot is effectively a horizontal bar plot of each instance's silhouette coefficient, ordered by cluster and rank.
The plot segments into a shape per cluster where the height indicates the number of instances in the cluster and the width reflects on well the instances fit in the clusters.
The red-dotted line indicates the silhouette score for the model.


```python
from sklearn.metrics import silhouette_samples
from yellowbrick.cluster import SilhouetteVisualizer

fig = plt.figure(figsize=(12, 18))

for i, k in enumerate(np.arange(2, 8)):
    plt.subplot(3, 2, i+1)
    km = KMeans(n_clusters=k)
    viz = SilhouetteVisualizer(km, colors='yellowbrick')
    viz.fit(X)
    viz.finalize()

plt.show()
```


![png](homl_ch09_Unsupervised-learning-techniques_files/homl_ch09_Unsupervised-learning-techniques_15_0.png)


Below are some of the limits of K-means:

* can reach suboptimal solutions
* the number of clusters must be declared
* K-means is not great at clustering non-spherical clusters of unequal densities

### Using clustering for image segmentation

Below is an example of using K-means to segment an image by color.


```python
from matplotlib.image import imread
import os

ladybug = imread(os.path.join('assets', 'homl', 'images', 'ladybug.png'))
# ladybug = ladybug[100:300, 300:500, :]
ladybug.shape
```




    (533, 800, 3)




```python
def plot_img(img): 
    plt.imshow(img)
    plt.grid(None)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)

plot_img(ladybug)
```


![png](homl_ch09_Unsupervised-learning-techniques_files/homl_ch09_Unsupervised-learning-techniques_18_0.png)



```python
X = ladybug.reshape(-1, 3)
X[:5, :]
```




    array([[0.09803922, 0.11372549, 0.00784314],
           [0.09411765, 0.10980392, 0.00392157],
           [0.09411765, 0.11372549, 0.        ],
           [0.10196079, 0.11372549, 0.        ],
           [0.09803922, 0.11372549, 0.00784314]], dtype=float32)




```python
kmeans = KMeans(n_clusters=8).fit(X)

segmented_img = kmeans.cluster_centers_[kmeans.labels_]
segmented_img = segmented_img.reshape(ladybug.shape)

plot_img(segmented_img)
```


![png](homl_ch09_Unsupervised-learning-techniques_files/homl_ch09_Unsupervised-learning-techniques_20_0.png)



```python
%%cache -d caches ch09_ladybug_silhouetteplot.pkl

ks = np.arange(2, 15)
sil_scores = []

for k in ks:
    km = KMeans(n_clusters=k).fit(X)
    # Sample the data for silhouette score (too slow, otherwise).
    sample_idx = np.random.randint(0, len(X), 5000)
    X_sample = X[sample_idx, :]
    label_sample = km.labels_[sample_idx]
    sil_scores.append(silhouette_score(X_sample, label_sample))

plt.plot(ks, sil_scores, 'k--o')
plt.xlabel('$k$')
plt.ylabel('silhouette score')
plt.title('Ladybug image silhouette scores')
plt.show()
```

    [Skipped the cell's code and loaded variables  from file '/Users/admin/Developer/Python/100DaysOfPython/HandsOnMachineLearningWithScikitLearnAndTensorFlow/caches/ch09_ladybug_silhouetteplot.pkl'.]



![png](homl_ch09_Unsupervised-learning-techniques_files/homl_ch09_Unsupervised-learning-techniques_21_1.png)



```python
%%cache -d caches ch09_ladybug_decomposed.pkl

# Subjective judgement from silhouette scores
best_k = 6

kmeans = KMeans(n_clusters=best_k).fit(X)

fig = plt.figure(figsize=(9, 15))
o3 = np.ones(3)

for i in range(best_k):
    plt.subplot(4, 2, i+1)
    partial_img = [x if l == i else o3 for x, l in zip(X, kmeans.labels_)]
    partial_img = np.concatenate(partial_img).reshape(X.shape)
    plot_img(partial_img.reshape(ladybug.shape))
    plt.title(f'Ladybug segment {i}')

plt.show()
```

    [Skipped the cell's code and loaded variables  from file '/Users/admin/Developer/Python/100DaysOfPython/HandsOnMachineLearningWithScikitLearnAndTensorFlow/caches/ch09_ladybug_decomposed.pkl'.]



![png](homl_ch09_Unsupervised-learning-techniques_files/homl_ch09_Unsupervised-learning-techniques_22_1.png)


### Using clustering for preprocessing

K-means clustering can be used for dimensionality reduction as a preprocessing step for a supervised learning algorithm.
Below is an example using the digits data.


```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

X_digits, y_digits = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits)
```

First, just a logistic regression with the raw data.


```python
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(solver='lbfgs',
                             max_iter=5000,
                             multi_class='auto')
log_reg.fit(X_train, y_train)

# Logistic regression accuracy on test data.
log_reg.score(X_test, y_test)
```




    0.9666666666666667



This can be compared to using a pipeline with K-means as a preprocessing step.

There is a large drop in performance using the default number of clusters, 8.


```python
from sklearn.pipeline import Pipeline

log_reg_pipeline = Pipeline([
    ('kmeans', KMeans()),
    ('log_reg', LogisticRegression(solver='lbfgs',
                                   max_iter=5000,
                                   multi_class='auto'))
])

log_reg_pipeline.fit(X_train, y_train)

log_reg_pipeline.score(X_test, y_test)
```

    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:947: ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.
      "of iterations.", ConvergenceWarning)





    0.9177777777777778



Grid search can be used to find the best value for $k$.


```python
%%cache -d caches ch09_digitslogreg_coarsegridsearch.pkl grid_log_reg, param_grid

from sklearn.model_selection import GridSearchCV

param_grid = {'kmeans__n_clusters': np.arange(10, 220, 20)}
grid_log_reg = GridSearchCV(log_reg_pipeline,
                            param_grid,
                            cv=3,
                            n_jobs=-1)
grid_log_reg.fit(X_train, y_train)
```

    [Skipped the cell's code and loaded variables grid_log_reg, param_grid from file '/Users/admin/Developer/Python/100DaysOfPython/HandsOnMachineLearningWithScikitLearnAndTensorFlow/caches/ch09_digitslogreg_coarsegridsearch.pkl'.]


    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:947: ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.
      "of iterations.", ConvergenceWarning)



```python
grid_log_reg.best_params_
```




    {'kmeans__n_clusters': 170}




```python
%%cache -d caches ch09_digitslogreg_finegridsearch.pkl grid_log_reg, param_grid

param_grid = {'kmeans__n_clusters': np.arange(180, 220, 1)}

grid_log_reg = GridSearchCV(log_reg_pipeline,
                            param_grid,
                            cv=3)
grid_log_reg.fit(X_train, y_train)
```

    [Skipped the cell's code and loaded variables grid_log_reg, param_grid from file '/Users/admin/Developer/Python/100DaysOfPython/HandsOnMachineLearningWithScikitLearnAndTensorFlow/caches/ch09_digitslogreg_finegridsearch.pkl'.]


    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:947: ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.
      "of iterations.", ConvergenceWarning)
    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:947: ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.
      "of iterations.", ConvergenceWarning)
    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:947: ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.
      "of iterations.", ConvergenceWarning)
    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:947: ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.
      "of iterations.", ConvergenceWarning)
    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:947: ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.
      "of iterations.", ConvergenceWarning)
    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:947: ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.
      "of iterations.", ConvergenceWarning)
    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:947: ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.
      "of iterations.", ConvergenceWarning)
    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:947: ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.
      "of iterations.", ConvergenceWarning)
    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:947: ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.
      "of iterations.", ConvergenceWarning)
    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:947: ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.
      "of iterations.", ConvergenceWarning)
    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:947: ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.
      "of iterations.", ConvergenceWarning)
    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:947: ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.
      "of iterations.", ConvergenceWarning)
    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:947: ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.
      "of iterations.", ConvergenceWarning)
    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:947: ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.
      "of iterations.", ConvergenceWarning)
    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:947: ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.
      "of iterations.", ConvergenceWarning)
    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:947: ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.
      "of iterations.", ConvergenceWarning)
    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:947: ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.
      "of iterations.", ConvergenceWarning)
    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:947: ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.
      "of iterations.", ConvergenceWarning)
    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:947: ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.
      "of iterations.", ConvergenceWarning)
    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:947: ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.
      "of iterations.", ConvergenceWarning)
    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:947: ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.
      "of iterations.", ConvergenceWarning)
    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:947: ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.
      "of iterations.", ConvergenceWarning)
    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:947: ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.
      "of iterations.", ConvergenceWarning)
    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:947: ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.
      "of iterations.", ConvergenceWarning)
    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:947: ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.
      "of iterations.", ConvergenceWarning)
    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:947: ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.
      "of iterations.", ConvergenceWarning)
    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:947: ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.
      "of iterations.", ConvergenceWarning)
    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:947: ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.
      "of iterations.", ConvergenceWarning)
    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:947: ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.
      "of iterations.", ConvergenceWarning)
    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:947: ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.
      "of iterations.", ConvergenceWarning)
    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:947: ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.
      "of iterations.", ConvergenceWarning)
    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:947: ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.
      "of iterations.", ConvergenceWarning)
    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:947: ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.
      "of iterations.", ConvergenceWarning)
    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:947: ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.
      "of iterations.", ConvergenceWarning)
    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:947: ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.
      "of iterations.", ConvergenceWarning)
    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:947: ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.
      "of iterations.", ConvergenceWarning)



```python
grid_log_reg.best_params_
```




    {'kmeans__n_clusters': 181}




```python
grid_log_reg.best_score_
```




    0.9673348181143281




```python
grid_log_reg.score(X_test, y_test)
```




    0.9911111111111112



The best value for $k$ was 181 and the accuracy achieved with the best model was 99.1 %.

### Using clustering for semi-supervised learning

Often, the training data will only be partial labeled.
We can use clustering to propagate labels to instances without them.

This is mocked below in several steps.
The main goal is to extract "representative images" from the clusters, label them manually, and then propagate the label to the rest of the cluster.

The first example below shows how using clustering to find the 50 representative images is better than some random sample (here, using the first 50 instances).


```python
n_labeled = 50
log_reg = LogisticRegression(solver='lbfgs',
                             max_iter=5000,
                             multi_class='auto')
log_reg.fit(X_train[:n_labeled], y_train[:n_labeled])
log_reg.score(X_test, y_test)
```




    0.8288888888888889




```python
k = n_labeled
kmeans = KMeans(n_clusters=k)
X_digits_dist = kmeans.fit_transform(X_train)

# Get the representative images.
representative_idx = np.argmin(X_digits_dist, axis=0)
X_representative = X_train[representative_idx]

# Plot the representative images for manual labeling.
fig = plt.figure(figsize=(12, 3))
for i, x in enumerate(X_representative):
    plt.subplot(5, 10, i+1)
    plot_img(x.reshape((8, 8)))
```


![png](homl_ch09_Unsupervised-learning-techniques_files/homl_ch09_Unsupervised-learning-techniques_38_0.png)



```python
# Normally, we wouldn't know this and have to manually assign values using
# the grid of images below.
y_representative = y_train[representative_idx]
```

Using the same number of instances for training , just this time from the clusters, greatly improved the success of the logistic regression on the test data.


```python
log_reg = LogisticRegression(solver='lbfgs',
                             max_iter=5000,
                             multi_class='auto')
log_reg.fit(X_representative, y_representative)
log_reg.score(X_test, y_test)
```




    0.9155555555555556



We can also propagate the manual labels to all of the data instances in the respective clusters.
This gives an appreciable boost to the testing accuracy.


```python
y_train_propagated = np.empty(len(X_train), dtype=np.int32)
for i in range(k):
    y_train_propagated[kmeans.labels_ == i] = y_representative[i]


log_reg = LogisticRegression(solver='lbfgs',
                             max_iter=5000,
                             multi_class='auto')
log_reg.fit(X_train, y_train_propagated)
log_reg.score(X_test, y_test)
```




    0.9311111111111111



One last improvement can come from limiting the label propagation to only those nearest the centroid.


```python
percentile_closest = 20

X_cluster_dist = X_digits_dist[np.arange(len(X_train)), kmeans.labels_]

for i in range(k):
    in_cluster = (kmeans.labels_ == i)
    
    cluster_dist = X_cluster_dist[in_cluster]
    cutoff_dist = np.percentile(cluster_dist, percentile_closest)
    above_cutoff = (X_cluster_dist > cutoff_dist)
    
    X_cluster_dist[in_cluster & above_cutoff] = -1


partially_prop_idx = (X_cluster_dist != -1)
X_train_partially_prop = X_train[partially_prop_idx]
y_train_partially_prop = y_train_propagated[partially_prop_idx]

log_reg = LogisticRegression(solver='lbfgs',
                             max_iter=5000,
                             multi_class='auto')
log_reg.fit(X_train_partially_prop, y_train_partially_prop)
log_reg.score(X_test, y_test)
```




    0.9377777777777778



It doesn't seem to have helped too much in this instance, but it can often lead to improved testing accuracy in other circumstances.
The propagation was quite successful, as shown by how many were accurately identified below.


```python
np.mean(y_train_partially_prop == y_train[partially_prop_idx])
```




    1.0



## Density-Based Spatial Clustering of Applications with Noise (DBSCAN)

DBSCAN defines clusters as continuous regions of high density.
Below is the general algorithm:

1. For each instance, the algorithm counts how many instances are located within $\epsilon$ of it. This is the instance's *$\epsilon$-neighborhood*
2. If an instance has `min_samples` instances in its neighborhood, it is a *core instance*.
3. All instances, including other core instances, in the neighborhood of a core instance belong to the same cluster. This can create chain of core instances.
4. Any instance that is not itself a core instance nor is in the neighborhood of a core instance is an anomaly.

Below is an example of using DBSCAN on the artificial moons data.


```python
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from matplotlib import cm

# Moons data.
X, y = make_moons(n_samples=1000, noise=0.05, random_state=0)

# DBSCAN with unoptimzed hyperparameters.
dbscan = DBSCAN(eps=0.05, min_samples=5)
dbscan.fit(X)
```




    DBSCAN(algorithm='auto', eps=0.05, leaf_size=30, metric='euclidean',
           metric_params=None, min_samples=5, n_jobs=None, p=None)




```python
def plot_clustered_moon(x, c, title=None):
    plt.scatter(x[:, 0], x[:, 1], c=c, cmap='Set1')
    plt.xlabel('$x_1$', fontsize=12)
    plt.ylabel('$x_2$', fontsize=12)
    plt.title(title, fontsize=14)


plot_clustered_moon(X, dbscan.labels_, 'DBSCAN (unoptimized)')
plt.show()
```


![png](homl_ch09_Unsupervised-learning-techniques_files/homl_ch09_Unsupervised-learning-techniques_50_0.png)


The random red dots above were detected to be anomalies by DBSCAN.
They are labeled as -1 and therefore are easily removed, as shown below.


```python
clustered_X = X[dbscan.labels_ != -1, :]
plot_clustered_moon(clustered_X, 
                    dbscan.labels_[dbscan.labels_ != -1], 
                    'DBSCAN (anomalies removed)')
plt.show()
```


![png](homl_ch09_Unsupervised-learning-techniques_files/homl_ch09_Unsupervised-learning-techniques_52_0.png)


Annoyingly, DBSCAN has found 11 clusters where we would want it to find 2.
By changing the $\epsilon$ to 0.2, though, 2 clusters are found.


```python
# DBSCAN with a higher epsilon.
dbscan = DBSCAN(eps=0.20, min_samples=5)
dbscan.fit(X)

# Plotting.
plot_clustered_moon(X, dbscan.labels_, 'DBSCAN clustering')
plt.show()
```


![png](homl_ch09_Unsupervised-learning-techniques_files/homl_ch09_Unsupervised-learning-techniques_54_0.png)


The `DBSCAN` class from Scikit-Learn does not have a `predict()` method.
Instead, the user is intended to use another classifier based on the clustering by DBSCAN.
This is shown below with KNN and random forest models.


```python
from sklearn.neighbors import KNeighborsClassifier

# Get labeled training data from DBSCAN.
X_dbscan = dbscan.components_
y_dbscan = dbscan.labels_[dbscan.core_sample_indices_]

# Train a KNN classifier
knn = KNeighborsClassifier(n_neighbors=50)
knn.fit(X_dbscan, y_dbscan)

# Get classifications from KNN.
knn_predicted_y = knn.predict(X)

# Plot with the color from the KNN predictions.
plot_clustered_moon(X, knn_predicted_y, 
                     'DBSCAN clustering with KNN classification')
plt.show()
```


![png](homl_ch09_Unsupervised-learning-techniques_files/homl_ch09_Unsupervised-learning-techniques_56_0.png)



```python
from sklearn.ensemble import GradientBoostingClassifier

# Train a gradient boosting random forest classifier
rf = GradientBoostingClassifier(n_estimators=50)
rf.fit(X_dbscan, y_dbscan)

# Get classifications from random forest.
rf_predicted_y = rf.predict(X)

# Plot with the color from the random forest's predictions.
plt.scatter(X[:, 0], X[:, 1], c=rf_predicted_y, cmap='Set1')
plt.xlabel('$x_1$', fontsize=12)
plt.ylabel('$x_2$', fontsize=12)
plt.title(f'DBSCAN clustering with Gradient Boosting Random Forest classification', fontsize=14)
plt.show()
```


![png](homl_ch09_Unsupervised-learning-techniques_files/homl_ch09_Unsupervised-learning-techniques_57_0.png)


This was just a brief introduction to DBSCAN.
It is efficient, flexible, and robust, though proper tuning of the hyperparameters is essential.
Another interesting related clustering algorithm is *Hierarchical DBSCAN*, implemented in the [scikit-lean-contrib/hdbscan](https://github.com/scikit-learn-contrib/hdbscan) project.

## Other clustering algorithms

Below is a list of additional common clustering algorithms. 
I tried to provide a brief explanation and example for each.

### Agglomerative clustering

The instances are clustered from the bottom up, each new cluster being treated as a single instance to be compared to the remaining instances.


```python
from sklearn.cluster import AgglomerativeClustering

fig = plt.figure(figsize=(10, 5))
for i, k in enumerate((2, 10)):
    agg_clst = AgglomerativeClustering(n_clusters=k)
    agg_clst.fit(X, y)
    plt.subplot(1, 2, i+1)
    plot_clustered_moon(X, agg_clst.labels_, title=f'Agglomerative Clustering ($k={k}$)')
```


![png](homl_ch09_Unsupervised-learning-techniques_files/homl_ch09_Unsupervised-learning-techniques_59_0.png)


### Balanced  Iterative Reducing and Clustering using Hierarchies (BIRCH)

BIRCH was specifically designed for large data sets with not too many features (<20).
It holds less information in memory than K-Means and therefore can be more efficient for predicting the cluster for new instances.


```python
from sklearn.cluster import Birch

birch = Birch(threshold=0.5, 
              branching_factor=50, 
              n_clusters=4)
birch.fit(X)
plot_clustered_moon(X, birch.labels_, 'BIRCH')
```


![png](homl_ch09_Unsupervised-learning-techniques_files/homl_ch09_Unsupervised-learning-techniques_61_0.png)


### Mean-shift

The mean-shift algorithm assigns circle around each instance with a predefined radius (the `bandwidth` in Scikit-Learn).
The circle is then shifted to have a center at the mean of all of the circles it encloses.
This is completed iteratively until the circles stop moving.
All of the instances with circles that ended up at the same place are assigned to a cluster.
This clustering method has many of the benefits of DBSCAN - can recognize different shapes, has few hyperparameters (just 1), can identify the number of clusters - but it is a far slower algorithm (exponential) and has the tendency to chop up clusters at regions of differential density.


```python
from sklearn.cluster import MeanShift

fig = plt.figure(figsize=(10, 20))
for i, b in enumerate(np.arange(0.1, 0.9, 0.1)):
    plt.subplot(4, 2, i+1)
    ms = MeanShift(bandwidth=b)
    ms.fit(X)
    plot_clustered_moon(X, ms.labels_, 
                        f'Mean-Shift (bandwidth={np.round(b, 1)})')

plt.show()
```


![png](homl_ch09_Unsupervised-learning-techniques_files/homl_ch09_Unsupervised-learning-techniques_63_0.png)


The `MeanShift` class in Scikit-Learn has the parameter `cluster_all` to determine if anomalies should be unassigned to a cluster:

> If true, then all points are clustered, even those orphans that are not within any kernel. Orphans are assigned to the nearest kernel. If false, then orphans are given cluster label -1.


```python
# Mean-shift with `cluster_all=False` so not all points must be assigned.
ms = MeanShift(bandwidth = 0.65, cluster_all=False)
ms.fit(X)

# Plot with outliers small, black points.
plot_clustered_moon(X[ms.labels_ != -1, :], 
                    ms.labels_[ms.labels_ != -1],
                   'Mean-Shift with outliers')
plt.scatter(X[ms.labels_ == -1, 0], X[ms.labels_ == -1, 1], c='k', marker='.', label='outliers')
plt.legend(loc='best', fontsize=12)
plt.show()
```


![png](homl_ch09_Unsupervised-learning-techniques_files/homl_ch09_Unsupervised-learning-techniques_65_0.png)


### Affinity propagation

Affinity propagation can be thought of as voting: each instance votes for similar instances to be their representative until convergence where the representatives and their voters form a cluster.
It can detect any number of clusters with varying size and shape, though has exponential complexity.


```python
from sklearn.cluster import AffinityPropagation

affprop = AffinityPropagation(damping=0.6)
affprop.fit(X)
plot_clustered_moon(X, affprop.labels_, 'Affinity Propagation')
```


![png](homl_ch09_Unsupervised-learning-techniques_files/homl_ch09_Unsupervised-learning-techniques_67_0.png)


### Spectral clustering

Spectral clustering embeds the similarity matrix of the data in a lower dimension (i.e. it reduces the dimensionality) then uses another clustering algorithm in the lower-dimensional space (Scikit-Lean uses K-Means).
It can capture complex structures, but does not scale well for large data sets.


```python
from sklearn.cluster import SpectralClustering

sc = SpectralClustering(n_clusters=2)
sc.fit(X)
plot_clustered_moon(X, sc.labels_, 'Spectral Clustering (default params.)')
```


![png](homl_ch09_Unsupervised-learning-techniques_files/homl_ch09_Unsupervised-learning-techniques_69_0.png)



```python
sc = SpectralClustering(n_clusters=2, 
                        affinity='nearest_neighbors', 
                        n_neighbors=10)
sc.fit(X)
plot_clustered_moon(X, sc.labels_, 'Spectral Clustering (affinity assignment by NN)')
```

    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/sklearn/manifold/spectral_embedding_.py:235: UserWarning: Graph is not fully connected, spectral embedding may not work as expected.
      warnings.warn("Graph is not fully connected, spectral embedding"



![png](homl_ch09_Unsupervised-learning-techniques_files/homl_ch09_Unsupervised-learning-techniques_70_1.png)


## Gaussian Mixtures

A *Gaussian mixture* is "a probabilistic model that assumes that the instances were generated from a mixture of several Gaussian distributions whose parameters are unknown," (*HOML*, pg. 260).
They are assumed to be generated by the following process:

1. For each instance, a cluster is picked randomly from $k$ clusters. The probability of choosing the $j^{th}$ cluster is the cluster's weight $\phi^{(j)}$. The index of the cluster chosen for the $i^{th}$ instance is denoted as $z^{(i)}$.
2. If $z^{(i)} = j$, the location $\textbf{x}^{(i)}$ of the instance is sampled from the Gaussian distribution with mean $\mu^{(j)}$ and covariance matrix $\mathbf{\Sigma}^{(j)}$. This is denoted as: $\textbf{x}^{(i)} \sim \mathcal{N}(\mu^{(j)}, \mathbf{\Sigma}^{(j)})$.

Therefore, the only known values are the locations $\textbf{x}^{(i)}$ of each instance. 
The remaining variables, the weights for the clusters and the mean and covariance matrices of the Gaussians, are *latent*.
The `GaussianMixture` class from Scikit-Learn can estimate these parameters.

The `GaussianMixture` class is demonstrated below on some "blobs" data.


```python
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=1000, n_features=2, centers = 3, random_state=0)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='Set1', s=30, alpha=0.7)
plt.xlabel('$x_1$', fontsize=12)
plt.ylabel('$x_2$', fontsize=12)
plt.show()
```


![png](homl_ch09_Unsupervised-learning-techniques_files/homl_ch09_Unsupervised-learning-techniques_72_0.png)



```python
from sklearn.mixture import GaussianMixture

gm = GaussianMixture(n_components=3, n_init=10)
gm.fit(X)
```




    GaussianMixture(covariance_type='full', init_params='kmeans', max_iter=100,
                    means_init=None, n_components=3, n_init=10,
                    precisions_init=None, random_state=None, reg_covar=1e-06,
                    tol=0.001, verbose=0, verbose_interval=10, warm_start=False,
                    weights_init=None)



The means the 3 clusters.


```python
gm.means_
```




    array([[ 0.90032899,  4.21134015],
           [ 2.13303741,  0.81616235],
           [-1.52637074,  3.03228352]])



The covariance matrices of the 3 clusters.


```python
gm.covariances_
```




    array([[[0.94327485, 0.05809688],
            [0.05809688, 1.00935793]],
    
           [[0.93452405, 0.02019267],
            [0.02019267, 0.81981003]],
    
           [[0.96775285, 0.06885643],
            [0.06885643, 1.02552756]]])



The weights for the clusters were the same, as expected from the blobs.


```python
gm.weights_
```




    array([0.32933545, 0.32827093, 0.34239362])



The `GaussianMixture` class uses the *Expectation Maximization* (EM) algorithm to estimate the latent variables.
It is simillar to K-means, though it uses the likelihood that an instance belongs to each cluster to adjust the parameters at each step.
Just like K-means, EM can converge to a suboptimal solution.
Therefore, running it several times and selecting the best result is recommended (`n_init=10`).

We can see if the algorithm converged and how many iterations it took.


```python
gm.converged_
```




    True




```python
gm.n_iter_
```




    4



The `GaussianMixture` class has both `predict()` and `predict_proba()` methods for hard or soft classifications.
The first plot below shows the hard predictions from the Gaussian mixture.


```python
y_pred = gm.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='Set1', s=30, alpha=0.5)
plt.scatter(gm.means_[:, 0], gm.means_[:, 1], marker='x', s=100, c='k')
plt.xlabel('$x_1$', fontsize=12)
plt.ylabel('$x_2$', fontsize=12)
plt.show()
```


![png](homl_ch09_Unsupervised-learning-techniques_files/homl_ch09_Unsupervised-learning-techniques_84_0.png)


The following plot shows the hard classifications made by the model, but the transparency is controlled by the certainty of the prediction.


```python
probs = gm.predict_proba(X)
probs = probs[np.arange(0, len(X), 1), y_pred]
probs = probs.ravel()

cmap = plt.cm.get_cmap('Set1')
rgba_colors = cmap(y_pred)
rgba_colors[:, 3] = probs * 0.7

plt.scatter(X[:, 0], X[:, 1], c=rgba_colors, s=30)
plt.scatter(gm.means_[:, 0], gm.means_[:, 1], marker='x', s=100, c='k')
plt.xlabel('$x_1$', fontsize=12)
plt.ylabel('$x_2$', fontsize=12)
plt.suptitle('Gaussian Mixture of blobs', fontsize=14)
plt.title('transparency associated to the certainty of the classification', fontsize=12)
plt.show()
```


![png](homl_ch09_Unsupervised-learning-techniques_files/homl_ch09_Unsupervised-learning-techniques_86_0.png)


A Gaussian mixture model is a *generative model* meaning new instances can be sampled from it.


```python
X_new, cls = gm.sample(5000)

plt.scatter(X_new[:, 0], X_new[:, 1], c=cls, cmap='Set1', s=30, alpha=0.5)
plt.scatter(gm.means_[:, 0], gm.means_[:, 1], marker='x', s=100, c='k')
plt.xlabel('$x_1$', fontsize=12)
plt.ylabel('$x_2$', fontsize=12)
plt.title('New samples from the Gaussian Mixture', fontsize=14)
plt.show()
```


![png](homl_ch09_Unsupervised-learning-techniques_files/homl_ch09_Unsupervised-learning-techniques_88_0.png)



```python

```
