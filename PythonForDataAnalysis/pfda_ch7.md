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

### Removing duplicates

Duplicate rows can be found using the `duplicated()` method on a DataFrame.


```python
data = pd.DataFrame({
    'k1': ['one', 'two'] * 3 + ['two'],
    'k2': [1, 1, 2, 3, 3, 4, 4]
})
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
      <th>k1</th>
      <th>k2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>one</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>two</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>one</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>two</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>one</td>
      <td>3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>two</td>
      <td>4</td>
    </tr>
    <tr>
      <th>6</th>
      <td>two</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.duplicated()
```




    0    False
    1    False
    2    False
    3    False
    4    False
    5    False
    6     True
    dtype: bool




```python
data.drop_duplicates()
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
      <th>k1</th>
      <th>k2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>one</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>two</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>one</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>two</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>one</td>
      <td>3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>two</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



It is also possible to specify the column by which to find duplicates.


```python
data['v1'] = range(7)
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
      <th>k1</th>
      <th>k2</th>
      <th>v1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>one</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>two</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>one</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>two</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>one</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>two</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>two</td>
      <td>4</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.drop_duplicates(['k1'])
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
      <th>k1</th>
      <th>k2</th>
      <th>v1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>one</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>two</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



By default, the first unique row is kept, though this can be changed to keeping the last one.


```python
data.drop_duplicates(['k1', 'k2'], keep='last')
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
      <th>k1</th>
      <th>k2</th>
      <th>v1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>one</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>two</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>one</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>two</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>one</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>6</th>
      <td>two</td>
      <td>4</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>



### Transforming data using a function or mapping

Often want to perform a transformation on an array, Series of column of a DataFrame.


```python
data = pd.DataFrame({
    'food': ['bacon', 'pulled pork', 'bacon',
             'Pastrami', 'corned beef', 'Bacon',
             'pastrami', 'honey ham', 'nova lox'],
    'ounces': [4, 3, 12, 6, 7.5, 8, 3, 5, 6]
})
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
      <th>food</th>
      <th>ounces</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>bacon</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>pulled pork</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>bacon</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Pastrami</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>corned beef</td>
      <td>7.5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Bacon</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>pastrami</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>honey ham</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>nova lox</td>
      <td>6.0</td>
    </tr>
  </tbody>
</table>
</div>



We can add a column to indicate the source animal for each food.
The `map` method on a Series accepts function or dictionary to apply to each element.


```python
meat_to_animal = {
    'bacon': 'pig',
    'pulled pork': 'pig',
    'pastrami': 'cow',
    'corned beef': 'cow',
    'honey ham': 'pig',
    'nova lox': 'salmon'
}

data['animal'] = data['food'].str.lower().map(meat_to_animal)
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
      <th>food</th>
      <th>ounces</th>
      <th>animal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>bacon</td>
      <td>4.0</td>
      <td>pig</td>
    </tr>
    <tr>
      <th>1</th>
      <td>pulled pork</td>
      <td>3.0</td>
      <td>pig</td>
    </tr>
    <tr>
      <th>2</th>
      <td>bacon</td>
      <td>12.0</td>
      <td>pig</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Pastrami</td>
      <td>6.0</td>
      <td>cow</td>
    </tr>
    <tr>
      <th>4</th>
      <td>corned beef</td>
      <td>7.5</td>
      <td>cow</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Bacon</td>
      <td>8.0</td>
      <td>pig</td>
    </tr>
    <tr>
      <th>6</th>
      <td>pastrami</td>
      <td>3.0</td>
      <td>cow</td>
    </tr>
    <tr>
      <th>7</th>
      <td>honey ham</td>
      <td>5.0</td>
      <td>pig</td>
    </tr>
    <tr>
      <th>8</th>
      <td>nova lox</td>
      <td>6.0</td>
      <td>salmon</td>
    </tr>
  </tbody>
</table>
</div>



Alternatively, we could have passed a function.


```python
data['food'].map(lambda x: meat_to_animal[x.lower()])
```




    0       pig
    1       pig
    2       pig
    3       cow
    4       cow
    5       pig
    6       cow
    7       pig
    8    salmon
    Name: food, dtype: object



### Replacing values

The `replace(to_replace=x, value=y)` method replaces `x` with `y` in a Series.


```python
data = pd.Series([1.0, -999.0, 2.0, -999.0, -1000.0, 3.0])
data.replace(-999, np.nan)
```




    0       1.0
    1       NaN
    2       2.0
    3       NaN
    4   -1000.0
    5       3.0
    dtype: float64




```python
data.replace([-999, -1000], np.nan)
```




    0    1.0
    1    NaN
    2    2.0
    3    NaN
    4    NaN
    5    3.0
    dtype: float64




```python
data.replace([-999, -1000], [np.nan, 0])
```




    0    1.0
    1    NaN
    2    2.0
    3    NaN
    4    0.0
    5    3.0
    dtype: float64




```python
data.replace({-999: np.nan, -1000: 0})
```




    0    1.0
    1    NaN
    2    2.0
    3    NaN
    4    0.0
    5    3.0
    dtype: float64



### Renaming axis indexes

Axis labels can be transformed by a function or mapping, too.
This can be accomplished in-place.


```python
data = pd.DataFrame(np.arange(12).reshape((3, 4)),
                    index=['Ohio', 'Colorado', 'New York'],
                    columns=['one', 'two', 'three', 'four'])
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
      <th>one</th>
      <th>two</th>
      <th>three</th>
      <th>four</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ohio</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Colorado</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
transform = lambda x: x[:4].upper()
data.index.map(transform)
```




    Index(['OHIO', 'COLO', 'NEW '], dtype='object')




```python
data.index = data.index.map(transform)
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
      <th>one</th>
      <th>two</th>
      <th>three</th>
      <th>four</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>OHIO</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>COLO</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>NEW</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>



The index and column names can be transformed by passing functions to the `rename()` method.


```python
data.rename(index = str.title, columns = str.upper)
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
      <th>ONE</th>
      <th>TWO</th>
      <th>THREE</th>
      <th>FOUR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ohio</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Colo</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>New</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>



One can rename a subset of the indices and column names using a dictionary mapping.


```python
data.rename(index={'OHIO': 'INDIANA'},
            columns={'three': 'peekaboo'})
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
      <th>one</th>
      <th>two</th>
      <th>peekaboo</th>
      <th>four</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>INDIANA</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>COLO</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>NEW</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>



### Discretization and binning

As an example, we want to bin the following age data into 18-25, 26-35, 36-60, 61 and older.


```python
ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
bins = [18, 25, 35, 60, 100]
cats = pd.cut(ages, bins)
cats
```




    [(18, 25], (18, 25], (18, 25], (25, 35], (18, 25], ..., (25, 35], (60, 100], (35, 60], (35, 60], (25, 35]]
    Length: 12
    Categories (4, interval[int64]): [(18, 25] < (25, 35] < (35, 60] < (60, 100]]



The `cut()` function returns a `Categorical` object.


```python
cats.codes
```




    array([0, 0, 0, 1, 0, 0, 2, 1, 3, 2, 2, 1], dtype=int8)




```python
cats.categories
```




    IntervalIndex([(18, 25], (25, 35], (35, 60], (60, 100]],
                  closed='right',
                  dtype='interval[int64]')




```python
pd.value_counts(cats)
```




    (18, 25]     5
    (35, 60]     3
    (25, 35]     3
    (60, 100]    1
    dtype: int64



We can dictate a few paramters for `cut()` such as `right`for setting the left of the boundary to be inclusive and the right to be exclusive (denoted in standard methematical notation).


```python
pd.cut(ages, [18, 26, 36, 61, 100], right=False)
```




    [[18, 26), [18, 26), [18, 26), [26, 36), [18, 26), ..., [26, 36), [61, 100), [36, 61), [36, 61), [26, 36)]
    Length: 12
    Categories (4, interval[int64]): [[18, 26) < [26, 36) < [36, 61) < [61, 100)]



The names of te bins can be specified.


```python
group_names = ['Youth', 'YoungAdult', 'MiddleAged', 'Senior']
pd.cut(ages, bins, labels=group_names)
```




    [Youth, Youth, Youth, YoungAdult, Youth, ..., YoungAdult, Senior, MiddleAged, MiddleAged, YoungAdult]
    Length: 12
    Categories (4, object): [Youth < YoungAdult < MiddleAged < Senior]



Instead of specifying the bins, an integer can be passed for the number of equispaced bins to use.


```python
data = np.random.rand(20)
pd.cut(data, 4, precision=2)
```




    [(0.096, 0.32], (0.32, 0.54], (0.54, 0.77], (0.32, 0.54], (0.77, 0.99], ..., (0.096, 0.32], (0.096, 0.32], (0.32, 0.54], (0.77, 0.99], (0.096, 0.32]]
    Length: 20
    Categories (4, interval[float64]): [(0.096, 0.32] < (0.32, 0.54] < (0.54, 0.77] < (0.77, 0.99]]



The `qcut()` function binds the data based on sample quantiles.
This results in roughly equal-sized bins.


```python
data = np.random.randn(1000)
cats = pd.qcut(data, 4)  # cut into quartiles
cats
```




    [(0.605, 2.759], (-0.0516, 0.605], (-0.0516, 0.605], (-0.0516, 0.605], (-0.0516, 0.605], ..., (-0.0516, 0.605], (-3.0469999999999997, -0.698], (-3.0469999999999997, -0.698], (-3.0469999999999997, -0.698], (-0.698, -0.0516]]
    Length: 1000
    Categories (4, interval[float64]): [(-3.0469999999999997, -0.698] < (-0.698, -0.0516] < (-0.0516, 0.605] < (0.605, 2.759]]




```python
pd.value_counts(cats)
```




    (0.605, 2.759]                   250
    (-0.0516, 0.605]                 250
    (-0.698, -0.0516]                250
    (-3.0469999999999997, -0.698]    250
    dtype: int64



The specific quantiles to use can also be passed to `qcut()`.


```python
pd.qcut(data, [0, 0.1, 0.5, 0.9, 1.0])
```




    [(-0.0516, 1.182], (-0.0516, 1.182], (-0.0516, 1.182], (-0.0516, 1.182], (-0.0516, 1.182], ..., (-0.0516, 1.182], (-1.28, -0.0516], (-1.28, -0.0516], (-1.28, -0.0516], (-1.28, -0.0516]]
    Length: 1000
    Categories (4, interval[float64]): [(-3.0469999999999997, -1.28] < (-1.28, -0.0516] < (-0.0516, 1.182] < (1.182, 2.759]]



### Detecting and filtering outliers

Here is an example for detecting and filtering outliers in a data set.


```python
data = pd.DataFrame(np.random.randn(1000, 4))
data.describe()
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
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.014493</td>
      <td>-0.005655</td>
      <td>-0.002238</td>
      <td>-0.041619</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.971579</td>
      <td>0.999478</td>
      <td>0.991132</td>
      <td>0.990014</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-3.007437</td>
      <td>-3.116857</td>
      <td>-3.392300</td>
      <td>-3.740101</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.654814</td>
      <td>-0.671151</td>
      <td>-0.637831</td>
      <td>-0.763313</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-0.010606</td>
      <td>0.003705</td>
      <td>-0.063378</td>
      <td>0.011749</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.659244</td>
      <td>0.693056</td>
      <td>0.646954</td>
      <td>0.637535</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.979976</td>
      <td>3.801660</td>
      <td>3.427539</td>
      <td>2.929096</td>
    </tr>
  </tbody>
</table>
</div>



Lets identify values greater than 3 in magnitude.


```python
col = data[2]
col[np.abs(col) > 3]
```




    598    3.427539
    656   -3.392300
    Name: 2, dtype: float64



The rows with data greater than 3 in magnitide can be selected using the `any()` method.


```python
data[(np.abs(data) > 3).any(1)]
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
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.049093</td>
      <td>3.170975</td>
      <td>0.189500</td>
      <td>-1.348413</td>
    </tr>
    <tr>
      <th>241</th>
      <td>0.422819</td>
      <td>-3.116857</td>
      <td>0.644452</td>
      <td>-1.913743</td>
    </tr>
    <tr>
      <th>507</th>
      <td>-0.347585</td>
      <td>3.306574</td>
      <td>-1.510200</td>
      <td>0.203540</td>
    </tr>
    <tr>
      <th>516</th>
      <td>-0.126856</td>
      <td>3.801660</td>
      <td>2.315171</td>
      <td>0.139827</td>
    </tr>
    <tr>
      <th>598</th>
      <td>-0.752582</td>
      <td>0.045113</td>
      <td>3.427539</td>
      <td>0.604682</td>
    </tr>
    <tr>
      <th>602</th>
      <td>-0.553965</td>
      <td>-3.006499</td>
      <td>-0.047166</td>
      <td>0.831878</td>
    </tr>
    <tr>
      <th>656</th>
      <td>-0.479297</td>
      <td>-1.345508</td>
      <td>-3.392300</td>
      <td>0.155794</td>
    </tr>
    <tr>
      <th>674</th>
      <td>1.061435</td>
      <td>-0.435034</td>
      <td>0.657682</td>
      <td>-3.740101</td>
    </tr>
    <tr>
      <th>854</th>
      <td>-3.007437</td>
      <td>-2.330467</td>
      <td>-0.567803</td>
      <td>2.667322</td>
    </tr>
  </tbody>
</table>
</div>



Values can be set using this boolean indexing.


```python
data[np.abs(data) > 3] = np.sign(data) * 3
data.describe()
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
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.014500</td>
      <td>-0.006811</td>
      <td>-0.002273</td>
      <td>-0.040879</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.971556</td>
      <td>0.994856</td>
      <td>0.988474</td>
      <td>0.987520</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-3.000000</td>
      <td>-3.000000</td>
      <td>-3.000000</td>
      <td>-3.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.654814</td>
      <td>-0.671151</td>
      <td>-0.637831</td>
      <td>-0.763313</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-0.010606</td>
      <td>0.003705</td>
      <td>-0.063378</td>
      <td>0.011749</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.659244</td>
      <td>0.693056</td>
      <td>0.646954</td>
      <td>0.637535</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.979976</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>2.929096</td>
    </tr>
  </tbody>
</table>
</div>



### Permutation and random sampling

Permuting a Series or rows of a DataFrame is easily accomplished using `numpy.random.permutation()`.
Providing this function with the length of the axis to permute returns a new indexing to use for reordering.


```python
df = pd.DataFrame(np.arange(5 * 4).reshape((5, 4)))
sampler = np.random.permutation(5)
sampler
```




    array([0, 3, 2, 4, 1])




```python
df.take(sampler)
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
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16</td>
      <td>17</td>
      <td>18</td>
      <td>19</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>



A sample of a DataFrame can be taken with or without replacement.


```python
df.sample(n=3)
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
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.sample(n=6, replace=True)
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
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



### Computing indicator or dummy variables

This is often a useful tool for statistical analysis or machine learning.


```python
df = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'],
                   'data1': range(6)})
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
      <th>key</th>
      <th>data1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>b</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>a</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>c</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>a</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>b</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.get_dummies(df['key'])
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
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Somtimes if it useful to include a prefix in the column names.


```python
dummies = pd.get_dummies(df['key'], prefix='key')
df_with_dummy = df[['data1']].join(dummies)
df_with_dummy
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
      <th>data1</th>
      <th>key_a</th>
      <th>key_b</th>
      <th>key_c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Creating a dummy matrix is more complicated if a row of a DataFrame belongs to multiple categories.
For an example, we used the MovieLens 1M data set.


```python
mnames = ['movie_id', 'title', 'genres']
movies = pd.read_table('assets/datasets/movielens/movies.dat', sep='::',
                       header=None, names = mnames)
movies
```

    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/ipykernel_launcher.py:3: ParserWarning: Falling back to the 'python' engine because the 'c' engine does not support regex separators (separators > 1 char and different from '\s+' are interpreted as regex); you can avoid this warning by specifying engine='python'.
      This is separate from the ipykernel package so we can avoid doing imports until





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
      <th>movie_id</th>
      <th>title</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Animation|Children's|Comedy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children's|Fantasy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men (1995)</td>
      <td>Comedy|Romance</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale (1995)</td>
      <td>Comedy|Drama</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II (1995)</td>
      <td>Comedy</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3878</th>
      <td>3948</td>
      <td>Meet the Parents (2000)</td>
      <td>Comedy</td>
    </tr>
    <tr>
      <th>3879</th>
      <td>3949</td>
      <td>Requiem for a Dream (2000)</td>
      <td>Drama</td>
    </tr>
    <tr>
      <th>3880</th>
      <td>3950</td>
      <td>Tigerland (2000)</td>
      <td>Drama</td>
    </tr>
    <tr>
      <th>3881</th>
      <td>3951</td>
      <td>Two Family House (2000)</td>
      <td>Drama</td>
    </tr>
    <tr>
      <th>3882</th>
      <td>3952</td>
      <td>Contender, The (2000)</td>
      <td>Drama|Thriller</td>
    </tr>
  </tbody>
</table>
<p>3883 rows × 3 columns</p>
</div>



The goal of the next set of operations is to add an indicator variable for genre.


```python
all_genres = []
for x in movies.genres:
    all_genres.extend(x.split('|'))

genres = pd.unique(all_genres)
genres
```




    array(['Animation', "Children's", 'Comedy', 'Adventure', 'Fantasy',
           'Romance', 'Drama', 'Action', 'Crime', 'Thriller', 'Horror',
           'Sci-Fi', 'Documentary', 'War', 'Musical', 'Mystery', 'Film-Noir',
           'Western'], dtype=object)




```python
zero_matrix = np.zeros((len(movies), len(genres)))
dummies = pd.DataFrame(zero_matrix, columns=genres)
gen = movies.genres[0]
gen.split('|')
```




    ['Animation', "Children's", 'Comedy']




```python
dummies.columns.get_indexer(gen.split('|'))
```




    array([0, 1, 2])




```python
for i, gen in enumerate(movies.genres):
    indices = dummies.columns.get_indexer(gen.split('|'))
    dummies.iloc[i, indices] = 1

dummies
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
      <th>Animation</th>
      <th>Children's</th>
      <th>Comedy</th>
      <th>Adventure</th>
      <th>Fantasy</th>
      <th>Romance</th>
      <th>Drama</th>
      <th>Action</th>
      <th>Crime</th>
      <th>Thriller</th>
      <th>Horror</th>
      <th>Sci-Fi</th>
      <th>Documentary</th>
      <th>War</th>
      <th>Musical</th>
      <th>Mystery</th>
      <th>Film-Noir</th>
      <th>Western</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3878</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3879</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3880</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3881</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3882</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>3883 rows × 18 columns</p>
</div>




```python
movies_windic = movies.join(dummies.add_prefix('Genre_'))
movies_windic.iloc[0]
```




    movie_id                                       1
    title                           Toy Story (1995)
    genres               Animation|Children's|Comedy
    Genre_Animation                                1
    Genre_Children's                               1
    Genre_Comedy                                   1
    Genre_Adventure                                0
    Genre_Fantasy                                  0
    Genre_Romance                                  0
    Genre_Drama                                    0
    Genre_Action                                   0
    Genre_Crime                                    0
    Genre_Thriller                                 0
    Genre_Horror                                   0
    Genre_Sci-Fi                                   0
    Genre_Documentary                              0
    Genre_War                                      0
    Genre_Musical                                  0
    Genre_Mystery                                  0
    Genre_Film-Noir                                0
    Genre_Western                                  0
    Name: 0, dtype: object



## 7.3 String manipulation



```python

```
