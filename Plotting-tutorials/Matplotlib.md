# Matplotlib

This notebook is for working through the various [tutorials](https://matplotlib.org/tutorials/index.html) offered by on the `matplotlib` website.


```python
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.ticker import FuncFormatter
import pathlib

np.random.seed(0)

%matplotlib inline

assets_path = pathlib.Path('assets', 'Matplotlib-tutorial')
save_path = pathlib.Path('output', 'Matplotlib-tutorial')
```

## Usage guide ([link](https://matplotlib.org/tutorials/introductory/usage.html#sphx-glr-tutorials-introductory-usage-py))

### General concepts

Matplotlib is organized as a hierarchy.
The top is the matplotlib "state-machine environment" which is in the `matplotlib.pyplot` module.
At this level, simple functions are used to add standard plot elements to the current axes in the current figure.

The next level is the first level of the OO interface.
Here, the user can create figure and axes objects.

### Parts  of a figure

#### Figure

This is the **whole** figure.
It keeps track of the child axes, some special *artists* (eg. titles, figure legends) and the *canvas* (this is what actually "draws" the figure but is rarely explicitly addressed by a user).
A figure can have any number of `Axes`, but should have at least one.

The easiest way to make a new figure  is with pyplot.


```python
fig = plt.figure()
fig.suptitle('No  axes in this figure')
fig, ax_lst = plt.subplots(2, 2)
```


    <Figure size 432x288 with 0 Axes>



![png](Matplotlib_files/Matplotlib_3_1.png)


#### Axes

This is the region where the data is visualized and is the primary point of entry for the OO interface.
A given `Axes` object can only be in one `Figure`.
The Axes contains two (or three for 3D plots) `Axis` objects which take care of the data limits.
Each `Axes` has a title, set via `set_title()`, and x- and y-labels, set via `set_xlabel()` and `set_ylabel()`.

#### Axis

These are the number-line-like objects.
They take care of setting the graph limits and generating ticks and tick-labels.

#### Artist

Basically, everything that can be seen on the figure is an artist.
When the figure is rendered, the artists are drawn to the canvas.
Most artists are tied to an Axes.

###  Types of inputs to plotting functions

All plotting functions expect a `np.array` or `np.ma.masked_array` as input.

### Coding styles

There are multiple coding styles for using matplotlib, two of which are officially supported.
They are all valid, but it is important to not mix them up in a single figure.

Here is what the "pyplot" style looks like.


```python
x = np.arange(0, 10, 0.1)
y = np.sin(x)
fig, ax = plt.subplots()
ax.plot(x, y)
plt.show()
```


![png](Matplotlib_files/Matplotlib_5_0.png)


This method is good for creating plots using a function because it makes creating complicated subplots as easy as passing the desired axes.


```python
def my_plotter(ax, data1, data2, param_dict):
    """
    A helper function to make a plot.
    Parameters
    ----------
    ax : Axes
        The axes to draw to.

    data1 : array
        The `x` data.

    data2 : array
        The `y` data.

    param_dict : dict
        Dictionary of kwargs to pass to ax.plot.

    Returns
    -------
    out : list
        A list of artists added to the axes.
    """
    out = ax.plot(data1, data2, **param_dict)
    return out


x1, x2 = np.random.randn(2, 100)

fig, ax = plt.subplots(1, 1)
my_plotter(ax, x1, x2, {'marker': 'x'})
plt.show()
```


![png](Matplotlib_files/Matplotlib_7_0.png)



```python
x1, x2, x3, x4 = np.random.randn(4, 100)

fig, (ax1, ax2) = plt.subplots(1, 2)
my_plotter(ax1, x1, x2, {'marker': 'x'})
my_plotter(ax2, x3, x4, {'marker': 'o'})
plt.show()
```


![png](Matplotlib_files/Matplotlib_8_0.png)


## Pyplot tutorial ([link](https://matplotlib.org/tutorials/introductory/pyplot.html#sphx-glr-tutorials-introductory-pyplot-py))

### Intro to pyplot

`matplotlib.pyplot` is a collection of command style functions that each changes the figure.

Generating a plot is very easy.


```python
plt.plot([1, 2, 3, 4])
plt.ylabel('some numbers')
plt.show()
```


![png](Matplotlib_files/Matplotlib_10_0.png)



```python
plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
plt.show()
```


![png](Matplotlib_files/Matplotlib_11_0.png)


#### Formatting the style of your plot

There is an optional third positional argument to `plot()` that accepts a string dictating the style of the line and marker.
The defualt is `'b-'` for a solid blue line.
Here is how to use `ro` to creating red circles.
The `axis()` function sets the x- and y-axis limits.


```python
plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro')
plt.axis([0, 6, 0, 20])
plt.show()
```


![png](Matplotlib_files/Matplotlib_13_0.png)


It is also possible to specify multiple data to plot at once.


```python
t = np.arange(0., 5., 0.2)
plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
plt.show()
```


![png](Matplotlib_files/Matplotlib_15_0.png)


### Plotting with keyword strings

There are times where to data structure allows specification of particular plotting variables using strings.
A common example is a `pandas.DataFrame`.


```python
data = pd.DataFrame(
    {
        'a': np.arange(50),
        'c': np.random.randint(0, 50, 50),
        'd': np.abs(np.random.randn(50)) * 100
    }
)
data['b'] = data.a + 10 * np.random.randn(50)
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>c</th>
      <th>d</th>
      <th>b</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>4</td>
      <td>82.346757</td>
      <td>2.122954</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>17</td>
      <td>127.974266</td>
      <td>1.774748</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>3</td>
      <td>19.533102</td>
      <td>-2.338056</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>37</td>
      <td>78.674108</td>
      <td>1.692074</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>30</td>
      <td>78.160973</td>
      <td>3.637900</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.scatter('a', 'b', c='c', s='d', data=data)
plt.xlabel('entry a')
plt.ylabel('entry b')
plt.show()
```


![png](Matplotlib_files/Matplotlib_18_0.png)


### Plotting with categorical variables

Many plotting functions in matplotlib take categorical variables.


```python
names = ['group_a', 'group_b', 'group_c']
values = [1, 10, 100]

plt.figure(figsize=(9, 3))

plt.subplot(131)
plt.bar(names, values)
plt.subplot(132)
plt.scatter(names, values)
plt.subplot(133)
plt.plot(names, values)
plt.suptitle('Categorical Plotting')
plt.show()
```


![png](Matplotlib_files/Matplotlib_20_0.png)


### Controlling line properties

Lines have many attributes that can be set: (e.g. linewidth, dash style, antialiased).
Further, there are multiple ways to set these attributes.
One is with keyword arguments.


```python
plt.plot(x, y, linewidth=2.0)
plt.show()
```


![png](Matplotlib_files/Matplotlib_22_0.png)


Another is with setter methods on the line objects.
The `plt.plot()` function returns a list of `Line2D` objects.
The example below is of only one line, so it is indexed to extract the `Line2D` object from the list.


```python
line = plt.plot(x, y, '-')[0]
line.set_antialiased(False)
plt.show()
```


![png](Matplotlib_files/Matplotlib_24_0.png)



```python
line = plt.plot(x, y,
                color='k',
                linestyle='--',
                label='example line',
                dash_capstyle='butt')
```


![png](Matplotlib_files/Matplotlib_25_0.png)


### Working with multiple figures and axes

Pyplot has the concept of the *current figure* and the *current axes*.
All plotting commands apply to the current axes.
The function `gca()` returns the current `Axes` object and `gcf()` returns the current `Figure` object.
Normally, you need not worry about this because it happens automatically  behind the scenes.

Below is an example of a figure with two subplots.


```python
def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)

t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)

plt.figure()

plt.subplot(211)
plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')

plt.subplot(212)
plt.plot(t2, np.cos(2*np.pi*t2), 'r--')

plt.show()
```


![png](Matplotlib_files/Matplotlib_27_0.png)


Note that the memory required for a figure is not fully released until `plt.close()` is called.

### Working with text

The `text()` command can add arbitrary text to a plot.
It returns a `Text` object that can be customized further (see the [Text properties and layout](https://matplotlib.org/tutorials/text/text_props.html) documentation).


```python
mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10000)
n, bins, patches = plt.hist(x, 50, density=1, facecolor='g', alpha=0.75)

plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title('Histogram of IQ')
plt.text(60, 0.025, r'$\mu=100,\ \sigma=15$')
plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.show()
plt.close()
```


![png](Matplotlib_files/Matplotlib_30_0.png)


#### Annotating text

The `annotate()` command adds a tet and arrow to a plot.
There are [Basic](https://matplotlib.org/tutorials/text/annotations.html#annotations-tutorial) and [Advanced Annotation](https://matplotlib.org/tutorials/text/annotations.html#plotting-guide-annotation) tutorials available.


```python
t = np.arange(0.0, 5.0, 0.01)
s = np.cos(2*np.pi*t)
line, = plt.plot(t, s, lw=2)
plt.annotate('local max.', 
             xy=(2, 1), 
             xytext=(3, 1.5), 
             arrowprops={'facecolor': 'k', 'shrink': 0.05})
plt.ylim(-2, 2)
plt.show()
plt.close()
```


![png](Matplotlib_files/Matplotlib_32_0.png)


### Logarithmic and other nonlinear axes

To change the scale of the x-axis to logarithmic, use `plt.xscale('log')`.
Some common options are `'linear'` (default), `'log'`, `'symlog'` (symmetric log), and `'logit''`.


```python
from matplotlib.ticker import NullFormatter  # useful for `logit` scale

# make up some data in the interval (0, 1)
y = np.random.normal(loc=0.5, scale=0.4, size=1000)
y = y[(y > 0) & (y < 1)]
y.sort()
x = np.arange(len(y))

# plot with various axes scales
plt.figure(figsize=(10, 10))

# linear
plt.subplot(221)
plt.plot(x, y)
plt.yscale('linear')
plt.title('linear')
plt.grid(True)


# log
plt.subplot(222)
plt.plot(x, y)
plt.yscale('log')
plt.title('log')
plt.grid(True)


# symmetric log
plt.subplot(223)
plt.plot(x, y - y.mean())
plt.yscale('symlog', linthreshy=0.01)
plt.title('symlog')
plt.grid(True)

# logit
plt.subplot(224)
plt.plot(x, y)
plt.yscale('logit')
plt.title('logit')
plt.grid(True)
# Format the minor tick labels of the y-axis into empty strings with
# `NullFormatter`, to avoid cumbering the axis with too many labels.
plt.gca().yaxis.set_minor_formatter(NullFormatter())
# Adjust the subplot layout, because the logit one may take more space
# than usual, due to y-tick labels like "1 - 10^{-3}"
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)

plt.show()
plt.close()
```


![png](Matplotlib_files/Matplotlib_34_0.png)


## Sample plots in Matplotlib ([link](https://matplotlib.org/tutorials/introductory/sample_plots.html#sphx-glr-tutorials-introductory-sample-plots-py))

This is not a tutorial, but instead there is a gallery of example plots with their source code.
I decided to just try plotting a few of them.

[**Contouring and pseudocolor**](https://matplotlib.org/gallery/images_contours_and_fields/pcolormesh_levels.html)


```python
import matplotlib
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

dx, dy = 0.05, 0.05
y, x = np.mgrid[slice(1, 5 + dy, dy),
                slice(1, 5 + dx, dx)]
```


```python
y
```




    array([[1.  , 1.  , 1.  , ..., 1.  , 1.  , 1.  ],
           [1.05, 1.05, 1.05, ..., 1.05, 1.05, 1.05],
           [1.1 , 1.1 , 1.1 , ..., 1.1 , 1.1 , 1.1 ],
           ...,
           [4.9 , 4.9 , 4.9 , ..., 4.9 , 4.9 , 4.9 ],
           [4.95, 4.95, 4.95, ..., 4.95, 4.95, 4.95],
           [5.  , 5.  , 5.  , ..., 5.  , 5.  , 5.  ]])




```python
x
```




    array([[1.  , 1.05, 1.1 , ..., 4.9 , 4.95, 5.  ],
           [1.  , 1.05, 1.1 , ..., 4.9 , 4.95, 5.  ],
           [1.  , 1.05, 1.1 , ..., 4.9 , 4.95, 5.  ],
           ...,
           [1.  , 1.05, 1.1 , ..., 4.9 , 4.95, 5.  ],
           [1.  , 1.05, 1.1 , ..., 4.9 , 4.95, 5.  ],
           [1.  , 1.05, 1.1 , ..., 4.9 , 4.95, 5.  ]])




```python
z = np.sin(x)**10 + np.cos(10 + y*x) * np.cos(x)
z
```




    array([[ 0.18037951,  0.26823017,  0.36335424, ...,  0.70887839,
             0.58107739,  0.44192559],
           [ 0.20738012,  0.2942647 ,  0.38808212, ...,  0.68002396,
             0.54664565,  0.40298815],
           [ 0.23430727,  0.3201529 ,  0.41259223, ...,  0.66058997,
             0.5247303 ,  0.37987008],
           ...,
           [-0.19534556, -0.17962168, -0.11478605, ...,  0.67850552,
             0.52732772,  0.37422725],
           [-0.21439958, -0.19297694, -0.12193009, ...,  0.65971059,
             0.51699498,  0.37899662],
           [-0.23247283, -0.20513588, -0.12774952, ...,  0.65154933,
             0.52098567,  0.40107702]])




```python
z = z[:-1, :-1]
levels = MaxNLocator(nbins=15).tick_values(z.min(), z.max())
levels
```




    array([-1.05, -0.9 , -0.75, -0.6 , -0.45, -0.3 , -0.15,  0.  ,  0.15,
            0.3 ,  0.45,  0.6 ,  0.75,  0.9 ,  1.05,  1.2 ])




```python
cmap = plt.get_cmap('PiYG')
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

fig = plt.figure(figsize=(10, 10))
fig.tight_layout()

ax0 = plt.subplot(211)
im = ax0.pcolormesh(x, y, z, cmap=cmap, norm=norm)
fig.colorbar(im, ax=ax0)
ax0.set_title('pcolormesh with levels')

ax1 = plt.subplot(212)
cf = ax1.contourf(x[:-1, :-1] + dx/2.0,
                  y[:-1, :-1] + dy/2.0,
                  z, levels=levels, cmap=cmap)
fig.colorbar(cf, ax=ax1)
ax1.set_title('contourf with levels')

plt.show()
plt.close()
```


![png](Matplotlib_files/Matplotlib_41_0.png)


[**3D surface (color map)**](https://matplotlib.org/gallery/mplot3d/surface3d.html)


```python
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

fig = plt.figure(figsize=(8, 6))
ax = fig.gca(projection='3d')

X = np.arange(-5, 5, 0.2)
Y = np.arange(-5, 5, 0.2)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0)

fig.colorbar(surf, shrink=0.5, aspect=10)

plt.show()
plt.close()
```


![png](Matplotlib_files/Matplotlib_43_0.png)


[**Demo of the histogram (hist) function with a few features**](https://matplotlib.org/gallery/statistics/histogram_features.html)


```python
from sklearn.mixture import GaussianMixture

mu = 100
sigma = 15
x = mu + sigma * np.random.randn(10000)

num_bins = 50

fig = plt.figure(figsize=(8, 5))
ax = plt.subplot(111)

n, bins, patches = ax.hist(x, num_bins, density=1)

# Add a "best-fit" line.
gmm = GaussianMixture(n_components=1)
gmm.fit(x.reshape(-1, 1))
x_new = np.arange(40, 160, 1).reshape(-1, 1)
y_pred = np.exp(gmm.score_samples(x_new))

ax.plot(x_new, y_pred, 'r--', linewidth=2)
ax.axis([40, 170, 0, 0.030])

plt.show()
plt.close()
```


![png](Matplotlib_files/Matplotlib_45_0.png)


## Image tutorial ([link](https://matplotlib.org/tutorials/introductory/images.html#sphx-glr-tutorials-introductory-images-py))

### Importing image data into Numpy arrays

Natively, Matplotlib only supports PNG images.
Under the hood, Matplotlib relies on the *Pillow* library for loading images.
The image used below was downloaded from the tutorial website.
It is an 8-bit image with one value per pixel representing a grayscale.
Each value of the array represents the shade of a pixel.

![](assets/Matplotlib-tutorial/stinkbug.png)


```python
img = mpimg.imread(assets_path.joinpath('stinkbug.png').as_posix())
img
```




    array([[0.40784314, 0.40784314, 0.40784314, ..., 0.42745098, 0.42745098,
            0.42745098],
           [0.4117647 , 0.4117647 , 0.4117647 , ..., 0.42745098, 0.42745098,
            0.42745098],
           [0.41960785, 0.41568628, 0.41568628, ..., 0.43137255, 0.43137255,
            0.43137255],
           ...,
           [0.4392157 , 0.43529412, 0.43137255, ..., 0.45490196, 0.4509804 ,
            0.4509804 ],
           [0.44313726, 0.44313726, 0.4392157 , ..., 0.4509804 , 0.44705883,
            0.44705883],
           [0.44313726, 0.4509804 , 0.4509804 , ..., 0.44705883, 0.44705883,
            0.44313726]], dtype=float32)




```python
img.shape
```




    (375, 500)



### Plotting Numpy arrays as images

The Numpy array can be siplayed as an image using `imshow()`.
The default colormap is *viridis*


```python
imgplot = plt.imshow(img)
```


![png](Matplotlib_files/Matplotlib_50_0.png)



```python
imgplot = plt.imshow(img, cmap='hot')
```


![png](Matplotlib_files/Matplotlib_51_0.png)



```python
imgplot = plt.imshow(img)
imgplot.set_cmap('nipy_spectral')
```


![png](Matplotlib_files/Matplotlib_52_0.png)



```python
imgplot = plt.imshow(img)
plt.colorbar()
plt.show()
```


![png](Matplotlib_files/Matplotlib_53_0.png)


It is possible to change to limits of the colormap by "clipping" them to a range.


```python
imgplotg = plt.imshow(img, clim=(0.1, 0.6))
plt.colorbar()
plt.show()
```


![png](Matplotlib_files/Matplotlib_55_0.png)


## The lifecycle of a plot ([link](https://matplotlib.org/tutorials/introductory/lifecycle.html#sphx-glr-tutorials-introductory-lifecycle-py))

This tutorial demonstrated the process of creating a full-featured figure.

### Our data


```python
data = {'Barton LLC': 109438.50,
        'Frami, Hills and Schmidt': 103569.59,
        'Fritsch, Russel and Anderson': 112214.71,
        'Jerde-Hilpert': 112591.43,
        'Keeling LLC': 100934.30,
        'Koepp Ltd': 103660.54,
        'Kulas Inc': 137351.96,
        'Trantow-Barrows': 123381.38,
        'White-Trantow': 135841.99,
        'Will LLC': 104437.60}
group_data = list(data.values())
group_names = list(data.keys())
group_mean = np.mean(group_data)
data
```




    {'Barton LLC': 109438.5,
     'Frami, Hills and Schmidt': 103569.59,
     'Fritsch, Russel and Anderson': 112214.71,
     'Jerde-Hilpert': 112591.43,
     'Keeling LLC': 100934.3,
     'Koepp Ltd': 103660.54,
     'Kulas Inc': 137351.96,
     'Trantow-Barrows': 123381.38,
     'White-Trantow': 135841.99,
     'Will LLC': 104437.6}



### Getting started


```python
fig, ax = plt.subplots()
```


![png](Matplotlib_files/Matplotlib_59_0.png)



```python
fig, ax = plt.subplots()
ax.barh(group_names, group_data)
```




    <BarContainer object of 10 artists>




![png](Matplotlib_files/Matplotlib_60_1.png)


### Controlling the style

The following are the available plot styles.


```python
print(plt.style.available)
```

    ['seaborn-dark', 'seaborn-darkgrid', 'seaborn-ticks', 'fivethirtyeight', 'seaborn-whitegrid', 'classic', '_classic_test', 'fast', 'seaborn-talk', 'seaborn-dark-palette', 'seaborn-bright', 'seaborn-pastel', 'grayscale', 'seaborn-notebook', 'ggplot', 'seaborn-colorblind', 'seaborn-muted', 'seaborn', 'Solarize_Light2', 'seaborn-paper', 'bmh', 'tableau-colorblind10', 'seaborn-white', 'dark_background', 'seaborn-poster', 'seaborn-deep']


They are easy to implement.


```python
plt.style.use('fivethirtyeight')
```


```python
fig, ax = plt.subplots()
ax.barh(group_names, group_data)
```




    <BarContainer object of 10 artists>




![png](Matplotlib_files/Matplotlib_65_1.png)


### Customizing the plot

The `setp()` method can set the properties of one or a list of Matplotlib objects at once.
Below, we extract the x-axis labels and rotate them and set their horizontal alignment to the right.


```python
fig, ax = plt.subplots()
ax.barh(group_names, group_data)
x_labels = ax.get_xticklabels()
plt.setp(x_labels, rotation=45, horizontalalignment='right')
```




    [None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None]




![png](Matplotlib_files/Matplotlib_67_1.png)


We will use the OO interface of Matplotlib to add properties using the `set()` method of the `Axes` object `ax`.


```python
fig, ax = plt.subplots()
ax.barh(group_names, group_data)
labels = ax.get_xticklabels()
plt.setp(labels, rotation=45, horizontalalignment='right')
ax.set(xlim=[-1000, 140000],
       xlabel='Total Revenue',
       ylabel='Company',
       title='Company Revenue')
```




    [Text(0, 0.5, 'Company'),
     (-1000, 140000),
     Text(0.5, 0, 'Total Revenue'),
     Text(0.5, 1.0, 'Company Revenue')]




![png](Matplotlib_files/Matplotlib_69_1.png)


The size of the figure can be set when the `Figure` and `Axes` objects were created.


```python
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(group_names, group_data)
labels = ax.get_xticklabels()
plt.setp(labels, rotation=45, horizontalalignment='right')
ax.set(xlim=[-1000, 140000],
       xlabel='Total Revenue',
       ylabel='Company',
       title='Company Revenue')
```




    [Text(0, 0.5, 'Company'),
     (-1000, 140000),
     Text(0.5, 0, 'Total Revenue'),
     Text(0.5, 1.0, 'Company Revenue')]




![png](Matplotlib_files/Matplotlib_71_1.png)


Custom formating guidelines for labels can be created using a `FuncFormatter` object initialized with a function that returns the desired label string for a current value and position.


```python
def currency_formatter(x, pos):
    """The two args are the value and position."""
    if x >= 1e6:
        return '${:1.1f}M'.format(x*1e-6)
    else :
        return '${:1.0f}K'.format(x*1e-3)


formatter = FuncFormatter(currency_formatter)

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(group_names, group_data)
labels = ax.get_xticklabels()
plt.setp(labels, rotation=45, horizontalalignment='right')
ax.set(xlim=[-1000, 140000],
       xlabel='Total Revenue',
       ylabel='Company',
       title='Company Revenue')
ax.xaxis.set_major_formatter(formatter)
```


![png](Matplotlib_files/Matplotlib_73_0.png)


### Combining multiple visualizations

Multiple plot elements can be drawn on the same instance of `Axes`.


```python
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(group_names, group_data)
labels = ax.get_xticklabels()
plt.setp(labels, rotation=45, horizontalalignment='right')

# Add a vertical line.
ax.axvline(group_mean, ls='--', color='r')

# Annotate new companies.
for group in [3, 5, 8]:
    ax.text(145000, group, 'new company', 
            fontsize=10, verticalalignment='center')

# Move the title up so the plot doesn't get too cramped.
ax.title.set(y=1.05)

ax.set(xlim=[-1000, 140000],
       xlabel='Total Revenue',
       ylabel='Company',
       title='Company Revenue')
ax.xaxis.set_major_formatter(formatter)
ax.set_xticks([0, 25e3, 50e3, 75e3, 100e3, 125e3])

plt.show()
```


![png](Matplotlib_files/Matplotlib_75_0.png)


### Saving our plot

A list of all file types that Matplotlib can save to is shown below.


```python
fig.canvas.get_supported_filetypes()
```




    {'ps': 'Postscript',
     'eps': 'Encapsulated Postscript',
     'pdf': 'Portable Document Format',
     'pgf': 'PGF code for LaTeX',
     'png': 'Portable Network Graphics',
     'raw': 'Raw RGBA bitmap',
     'rgba': 'Raw RGBA bitmap',
     'svg': 'Scalable Vector Graphics',
     'svgz': 'Scalable Vector Graphics',
     'jpg': 'Joint Photographic Experts Group',
     'jpeg': 'Joint Photographic Experts Group',
     'tif': 'Tagged Image File Format',
     'tiff': 'Tagged Image File Format'}




```python
fig.savefig(
    save_path.joinpath('the-lifecycle-of-a-plot_final-img.jpeg').as_posix(),
    transparent=False, dpi=80, bbox_inches='tight'
)
```

Below is the saved image rendered in a Markdown cell.

![](output/Matplotlib-tutorial/the-lifecycle-of-a-plot_final-img.jpeg)

## Customizing Matplotlib with style sheets and rcParams ([link](https://matplotlib.org/tutorials/introductory/customizing.html#sphx-glr-tutorials-introductory-customizing-py))

Below is a list of all available styles in Matplotlib.


```python
sorted(plt.style.available)
```




    ['Solarize_Light2',
     '_classic_test',
     'bmh',
     'classic',
     'dark_background',
     'fast',
     'fivethirtyeight',
     'ggplot',
     'grayscale',
     'seaborn',
     'seaborn-bright',
     'seaborn-colorblind',
     'seaborn-dark',
     'seaborn-dark-palette',
     'seaborn-darkgrid',
     'seaborn-deep',
     'seaborn-muted',
     'seaborn-notebook',
     'seaborn-paper',
     'seaborn-pastel',
     'seaborn-poster',
     'seaborn-talk',
     'seaborn-ticks',
     'seaborn-white',
     'seaborn-whitegrid',
     'tableau-colorblind10']



Set the styling using `plt.style.use('style-name')`


```python
plt.style.use('ggplot')
```


```python
x = np.arange(0, 10, 0.1)
y = np.sin(x)
fig, ax = plt.subplots()
ax.plot(x, y)
plt.show()
```


![png](Matplotlib_files/Matplotlib_84_0.png)



```python
plt.style.use('seaborn-paper')
```


```python
x = np.arange(0, 10, 0.1)
y = np.sin(x)
fig, ax = plt.subplots()
ax.plot(x, y)
plt.show()
```


![png](Matplotlib_files/Matplotlib_86_0.png)



```python
plt.style.use('seaborn-white')
```


```python
x = np.arange(0, 10, 0.1)
y = np.sin(x)
fig, ax = plt.subplots()
ax.plot(x, y)
plt.show()
```


![png](Matplotlib_files/Matplotlib_88_0.png)


### Defining you own style

A style sheet can be added to mpl_configdir/stylelib and then loaded with `plt.style.use('your-own-stylesheet.mplstyle')` or the sheet can be saved to another directory and the path or URL can be passed to `plt.style.use('path/to/style-sheet.mplstyle')`.

### Composing styles

Style sheets are designed to be composed together.
The styles listed last will overwrite properties set by those before them.


```python
plt.style.use(['dark_background', 'seaborn-notebook'])
```


```python
x = np.arange(0, 10, 0.1)
y = np.sin(x)
fig, ax = plt.subplots()
ax.plot(x, y)
plt.show()
```


![png](Matplotlib_files/Matplotlib_91_0.png)


### Matplotlib rcParams

#### Dynamic rc settings

The default rc settings can be changed within a script or interactive session.
The settings are stored in a dictionary-like variable `matplotlib.rcParams`.
It can be modified directly.


```python
plt.style.use('classic')

mpl.rcParams['lines.linewidth'] = 10

x = np.arange(0, 10, 0.1)
y = np.sin(x)
fig, ax = plt.subplots()
ax.plot(x, y)
plt.show()
```


![png](Matplotlib_files/Matplotlib_93_0.png)


Also, multiple settings of a plot object can be adjusted at once using `matplotlib.rc()`.


```python
mpl.rc('lines', linewidth=4, color='g')

x = np.arange(0, 10, 0.1)
y = np.sin(x)
fig, ax = plt.subplots()
ax.plot(x, y)
plt.show()
```


![png](Matplotlib_files/Matplotlib_95_0.png)


The `matplotlib.rcdefaults()` restores the standard matplotlib default settings.


```python
mpl.rcdefaults()
```


```python
x = np.arange(0, 10, 0.1)
y = np.sin(x)
fig, ax = plt.subplots()
ax.plot(x, y)
plt.show()
```


![png](Matplotlib_files/Matplotlib_98_0.png)


### The "matplotlibrc" file

Matplotlib uses *matplotlibrc* configuration files to customize almost every property of a plot.
Matplotlib looks for a *matploblibrc* file in four locations in the following order:

1. current working directory
2. at `$MATPLOTLIBRC` if it is a file, else `$MATPLOTLIBRC/matplotlibrc`
3. `.matplotlib/matplotlibrc`
4. `INSTALL/matplotlib/mpl-data/matplotlibrc` where install is the installation location of Python

Only the first *matplotlibrc* file is used.
The location of the file currently used can be found using the `matplotlib_fname()` function.


```python
mpl.matplotlib_fname()
```




    '/opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/matplotlib/mpl-data/matplotlibrc'



**A sample *matplotlibrc* file is avaiable at the following link: [sample *matplotlibrc* file](https://matplotlib.org/tutorials/introductory/customizing.html#matplotlibrc-sample)**

## Legend guide ([link](https://matplotlib.org/tutorials/intermediate/legend_guide.html#sphx-glr-tutorials-intermediate-legend-guide-py))

The documentation for `legend()` is available [here](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html#matplotlib.pyplot.legend).
Below are some common terms specifcially defined:

* **legend entry**: A legend is composed of one or more legend entries. An entry is composed of one key and one label.
* **legend key**: A colored and/or patterned marker to the left of each legend label.
* **legend label**: The text describing the handle that the key represents.
* **legend handle**: The original object used to generate the appropriate entry in the legend.

### Controlling the legend entries

Calling `legend()` with no arguments automatically fetches the legend handles and their associated labels.
It is equivalent to the following.


```python
fig, ax = plt.subplots()

ax.plot([1, 2, 3], 'r', label='line 1')
ax.plot([3, 2, 1], 'b', label='line 2')

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)

plt.show()
```


![png](Matplotlib_files/Matplotlib_103_0.png)


The method returns a list of handles/artists and their labels.


```python
handles
```




    [<matplotlib.lines.Line2D at 0x1a1c260890>,
     <matplotlib.lines.Line2D at 0x1a1c1c9b90>]




```python
labels
```




    ['line 1', 'line 2']



Alternatively, for full control of what the legend includes, it is common to pass the appropriate handles directly to `legend()`. (The order *does* matter.)


```python
fig, ax = plt.subplots()

lineup, = ax.plot([1, 2, 3], 'r', label='line 1')
linedown, = ax.plot([3, 2, 1], 'b', label='line 2')

ax.legend(handles=[linedown, lineup])

plt.show()
```


![png](Matplotlib_files/Matplotlib_108_0.png)


### Legend location

The location of the legend can be set using the `loc` keyword argument.
A complete list of avaialble locations is provided in the documentation for `legend()`.

The `bbox_to_anchor` keyword gives more manual control over the legend placement.


```python
fig, ax = plt.subplots()

lineup, = ax.plot([1, 2, 3], 'r', label='line 1')
linedown, = ax.plot([3, 2, 1], 'b', label='line 2')

ax.legend(bbox_to_anchor=(0.2, 1.2))

plt.show()
```


![png](Matplotlib_files/Matplotlib_110_0.png)


Below are two more examples of adjusting the legend.


```python
fig, ax = plt.subplots()

lineup, = ax.plot([1, 2, 3], 'r', label='line 1')
linedown, = ax.plot([3, 2, 1], 'b', label='line 2')

ax.legend(bbox_to_anchor=(0.0, 1.02, 1.0, 0.0),
          loc='lower left',
          ncol=2,
          mode='expand',
          borderaxespad=0.0)

plt.show()
```


![png](Matplotlib_files/Matplotlib_112_0.png)



```python
fig, ax = plt.subplots()

lineup, = ax.plot([1, 2, 3], 'r', label='line 1')
linedown, = ax.plot([3, 2, 1], 'b', label='line 2')

ax.legend(bbox_to_anchor=(1.02, 1),
          loc='upper left',
          borderaxespad=0.0)

plt.show()
```


![png](Matplotlib_files/Matplotlib_113_0.png)


### Multiple legends on the same axes

Each call to `legend()` resets the current legend, thus multiple legends *cannot* be added by calling `legend()` multiple times.
Instead, separate legend instances must be manually added.


```python
fig, ax = plt.subplots()

lineup, = ax.plot([1, 2, 3], 'r', label='line 1')
linedown, = ax.plot([3, 2, 1], 'b', label='line 2')

first_legend = plt.legend(handles=[lineup], loc='upper right')
ax.add_artist(first_legend)

plt.legend(handles=[linedown], loc='lower right')

plt.show()
```


![png](Matplotlib_files/Matplotlib_115_0.png)


### Legend handlers

The simplest way to modify legend handles is by provide a dictionary to `handler_map` argument in `legend()`.
The dictionary maps each plotted object to a subclass of `HandlerBase`.
Below is an example of customizing the number of points for the key of one of the lines.


```python
fig, ax = plt.subplots()

lineup, = ax.plot([1, 2, 3], 'r-o', label='line 1')
linedown, = ax.plot([3, 2, 1], 'b-x', label='line 2')

ax.legend(
    loc='center right',
    handler_map={lineup: mpl.legend_handler.HandlerLine2D(numpoints=4)}
)

plt.show()
```


![png](Matplotlib_files/Matplotlib_117_0.png)


Also, the mapping for the dictionary can be by type of plot object.


```python
fig, ax = plt.subplots()

lineup, = ax.plot([1, 2, 3], 'r-o', label='line 1')
linedown, = ax.plot([3, 2, 1], 'b-x', label='line 2')

ax.legend(
    loc='center right',
    handler_map={type(lineup): mpl.legend_handler.HandlerLine2D(numpoints=4)}
)

plt.show()
```


![png](Matplotlib_files/Matplotlib_119_0.png)


The following example demonstrates combining two legend keys on top of one another.


```python
z = np.random.randn(10)

fig, ax = plt.subplots()

red_dot,  = ax.plot(z, 'ro', markersize=15)
white_cross, = ax.plot(z[:5], 'w+', markersize=14, markeredgewidth=3)

plt.legend(loc='best',
           handles=[red_dot, (red_dot, white_cross)],
           labels=['Attr A', 'Attr A & B'])

plt.show()
```


![png](Matplotlib_files/Matplotlib_121_0.png)


It is also possible to assign several legend keys to the same entry.


```python
fig, ax = plt.subplots()

lineup, = ax.plot([1, 2, 3], 'r-o', label='line 1')
linedown, = ax.plot([3, 2, 1], 'b-x', label='line 2')

ax.legend(handles=[(lineup, linedown)],
          labels=['two keys'],
          numpoints=1,
          handler_map={tuple: mpl.legend_handler.HandlerTuple(ndivide=None)})

plt.show()
```


![png](Matplotlib_files/Matplotlib_123_0.png)



```python

```
