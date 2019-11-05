# Chapter 7. Data Cleaning and Preparation

In this chapter, the author reviewed tools for missing data, duplicate data, string manipulation, and a few other other common practices used for data preparation.

## 7.1 Handling missing data

Missing data is ignored by default when computing summary and descriptive statistics on a DataFrame or Series.
For numeric data, the floating-point value `NaN` ("not a number") is used.


```python
import pandas as pd
import numpy as np

np.random.seed(0)
```


```python
string_data = pd.Series(['aardvark', 'artichoke', np.nan, 'avocado'])
string_data
```




    0     aardvark
    1    artichoke
    2          NaN
    3      avocado
    dtype: object




```python
string_data.isnull()
```




    0    False
    1    False
    2     True
    3    False
    dtype: bool



### Filtering out missing data

The `dropna()` method removes `NaN` values from a Series.


```python
from numpy import nan as NA

data = pd.Series([1, NA, 3.5, NA, 7])
data
```




    0    1.0
    1    NaN
    2    3.5
    3    NaN
    4    7.0
    dtype: float64




```python
data.dropna()
```




    0    1.0
    2    3.5
    4    7.0
    dtype: float64



On a DataFrame, `dropna()` removes rows with an `NaN` values.


```python
data = pd.DataFrame([[1., 6.5, 3.],
                     [1., NA, NA],
                     [NA, NA, NA],
                     [NA, 6.5, 3.]])
data
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>6.5</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>6.5</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
cleaned = data.dropna()
cleaned
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>6.5</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>



However, passing `how='all'` will remove only rows with *all* `NaN`.


```python
data.dropna(how='all')
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>6.5</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>6.5</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>



Or only columns made up of all `NaN` can be dropped by declaring the axis.


```python
data[4] = NA
data
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>6.5</td>
      <td>3.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>6.5</td>
      <td>3.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.dropna(axis=1, how='all')
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>6.5</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>6.5</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>



You can also specify the maximum number of missing data values allowed for an individual row.


```python
df = pd.DataFrame(np.random.randn(7, 3))
df.iloc[:4, 1] = NA
df.iloc[:2, 2] = NA
df
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.764052</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.240893</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.950088</td>
      <td>NaN</td>
      <td>-0.103219</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.410599</td>
      <td>NaN</td>
      <td>1.454274</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.761038</td>
      <td>0.121675</td>
      <td>0.443863</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.333674</td>
      <td>1.494079</td>
      <td>-0.205158</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.313068</td>
      <td>-0.854096</td>
      <td>-2.552990</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.dropna()
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>0.761038</td>
      <td>0.121675</td>
      <td>0.443863</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.333674</td>
      <td>1.494079</td>
      <td>-0.205158</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.313068</td>
      <td>-0.854096</td>
      <td>-2.552990</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.dropna(thresh=2)
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>0.950088</td>
      <td>NaN</td>
      <td>-0.103219</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.410599</td>
      <td>NaN</td>
      <td>1.454274</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.761038</td>
      <td>0.121675</td>
      <td>0.443863</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.333674</td>
      <td>1.494079</td>
      <td>-0.205158</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.313068</td>
      <td>-0.854096</td>
      <td>-2.552990</td>
    </tr>
  </tbody>
</table>
</div>



### Filling in missing data

`fillna()` can be used in most cases to fill in missing data.


```python
df.fillna(0)
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.764052</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.240893</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.950088</td>
      <td>0.000000</td>
      <td>-0.103219</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.410599</td>
      <td>0.000000</td>
      <td>1.454274</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.761038</td>
      <td>0.121675</td>
      <td>0.443863</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.333674</td>
      <td>1.494079</td>
      <td>-0.205158</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.313068</td>
      <td>-0.854096</td>
      <td>-2.552990</td>
    </tr>
  </tbody>
</table>
</div>



A dictionary can be used to fill specific columns with specific values.


```python
df.fillna({1: 0.5, 2: 0})
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.764052</td>
      <td>0.500000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.240893</td>
      <td>0.500000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.950088</td>
      <td>0.500000</td>
      <td>-0.103219</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.410599</td>
      <td>0.500000</td>
      <td>1.454274</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.761038</td>
      <td>0.121675</td>
      <td>0.443863</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.333674</td>
      <td>1.494079</td>
      <td>-0.205158</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.313068</td>
      <td>-0.854096</td>
      <td>-2.552990</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.fillna(0, inplace=True)
df
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.764052</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.240893</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.950088</td>
      <td>0.000000</td>
      <td>-0.103219</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.410599</td>
      <td>0.000000</td>
      <td>1.454274</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.761038</td>
      <td>0.121675</td>
      <td>0.443863</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.333674</td>
      <td>1.494079</td>
      <td>-0.205158</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.313068</td>
      <td>-0.854096</td>
      <td>-2.552990</td>
    </tr>
  </tbody>
</table>
</div>



The same interpolation methods available for deciding which rows or columns to drop can be used for filling in data.


```python
df = pd.DataFrame(np.random.rand(6, 3))
df.iloc[2:, 1] = NA
df.iloc[4:, 2] = NA
df
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.264556</td>
      <td>0.774234</td>
      <td>0.456150</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.568434</td>
      <td>0.018790</td>
      <td>0.617635</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.612096</td>
      <td>NaN</td>
      <td>0.943748</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.681820</td>
      <td>NaN</td>
      <td>0.437032</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.697631</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.670638</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.fillna(method='ffill')  # 'ffill' = 'forward fill'
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.264556</td>
      <td>0.774234</td>
      <td>0.456150</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.568434</td>
      <td>0.018790</td>
      <td>0.617635</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.612096</td>
      <td>0.018790</td>
      <td>0.943748</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.681820</td>
      <td>0.018790</td>
      <td>0.437032</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.697631</td>
      <td>0.018790</td>
      <td>0.437032</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.670638</td>
      <td>0.018790</td>
      <td>0.437032</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.fillna(method='ffill', limit=2)
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.264556</td>
      <td>0.774234</td>
      <td>0.456150</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.568434</td>
      <td>0.018790</td>
      <td>0.617635</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.612096</td>
      <td>0.018790</td>
      <td>0.943748</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.681820</td>
      <td>0.018790</td>
      <td>0.437032</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.697631</td>
      <td>NaN</td>
      <td>0.437032</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.670638</td>
      <td>NaN</td>
      <td>0.437032</td>
    </tr>
  </tbody>
</table>
</div>



## 7.2 Data transformation


```python

```
