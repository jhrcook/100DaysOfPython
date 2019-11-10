# Chapter 9. Plotting and Visualization

## 9.1 A brief matplotlib API primer


```python
%matplotlib inline
```


```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.random.seed(0)
```


```python
data  = np.arange(10)
data
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
plt.plot(data)
```




    [<matplotlib.lines.Line2D at 0x12108bad0>]




![svg](pfda_ch9_files/pfda_ch9_4_1.svg)


### Figures and subplots
