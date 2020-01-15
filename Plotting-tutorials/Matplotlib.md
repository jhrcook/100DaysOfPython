# Matplotlib

This notebook is for working through the various [tutorials](https://matplotlib.org/tutorials/index.html) offered by on the `matplotlib` website.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(0)

%matplotlib inline
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



```python

```
