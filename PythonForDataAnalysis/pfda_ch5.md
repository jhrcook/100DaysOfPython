# Chapter 5. Getting Starting with pandas

Designed to for conducting vectorized functions with tabular data.

## Introduction to pandas data structures

The two most common data structures from pandas are *Series* and *DataFrame*.

### Series

A 1-D array like object containing a sequence of values and an associated array of data labels (called the *index*).

The simplest Series is from from an array.


```python
import pandas as pd
import numpy as np

obj = pd.Series([4, 7, -5, 3])
obj
```




    0    4
    1    7
    2   -5
    3    3
    dtype: int64



The index is shown to the left of each data point.
The values and indices can be extracted, specifically.


```python
obj.values
```




    array([ 4,  7, -5,  3])




```python
obj.index
```




    RangeIndex(start=0, stop=4, step=1)



The index can be specified.


```python
obj2 = pd.Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
obj2
```




    d    4
    b    7
    a   -5
    c    3
    dtype: int64




```python
obj2.index
```




    Index(['d', 'b', 'a', 'c'], dtype='object')



The index can be used to select single or multiple values.


```python
obj2['a']
```




    -5




```python
obj2['d'] = 6
```


```python
obj2[['c', 'a', 'd']]
```




    c    3
    a   -5
    d    6
    dtype: int64



Boolean indices, multiplication, and mathematical operations can also be used just like with NumPy.


```python
obj2[obj2 > 0]
```




    d    6
    b    7
    c    3
    dtype: int64




```python
obj2 * 2
```




    d    12
    b    14
    a   -10
    c     6
    dtype: int64




```python
np.exp(obj2)
```




    d     403.428793
    b    1096.633158
    a       0.006738
    c      20.085537
    dtype: float64



A Series can be thought of as a fixed-length, ordered dictionary.
It can often be used in a simillar fashion as a dictionary.
A Series can be created from a dictionary.


```python
'b' in obj2
```




    True




```python
sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
obj3  = pd.Series(sdata)
obj3
```




    Ohio      35000
    Texas     71000
    Oregon    16000
    Utah       5000
    dtype: int64



By default, the resulting Series is ordered by the index.
This can be overriden when the Series is created.
Note that indices without values get assigned `NaN` and only values from the dictionary with included indices are retained.


```python
states = ['California', 'Ohio', 'Oregon', 'Texas']
obj4 = pd.Series(sdata, index=states)
obj4
```




    California        NaN
    Ohio          35000.0
    Oregon        16000.0
    Texas         71000.0
    dtype: float64




```python
pd.isnull(obj4)
```




    California     True
    Ohio          False
    Oregon        False
    Texas         False
    dtype: bool




```python
obj4.isnull()
```




    California     True
    Ohio          False
    Oregon        False
    Texas         False
    dtype: bool



Series can be joined using the `+` operator that automatically joins by index.


```python
obj3 + obj4
```




    California         NaN
    Ohio           70000.0
    Oregon         32000.0
    Texas         142000.0
    Utah               NaN
    dtype: float64



Bot the Series object itself and its index have a `name` attribute.


```python
obj4.name = 'population'
obj4.index.name = 'state'
obj4
```




    state
    California        NaN
    Ohio          35000.0
    Oregon        16000.0
    Texas         71000.0
    Name: population, dtype: float64



### DataFrame

A rectangular table of data with an *ordered* collection of columns.
It has both a row and column index.

A DataFrame can be constructed from a dictionary of NumPy arrays.


```python
data = {
    'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
    'year': [2000, 2001, 2002, 2001, 2002, 2003],
    'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]
}
frame = pd.DataFrame(data)
frame
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
      <th>state</th>
      <th>year</th>
      <th>pop</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ohio</td>
      <td>2000</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ohio</td>
      <td>2001</td>
      <td>1.7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ohio</td>
      <td>2002</td>
      <td>3.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Nevada</td>
      <td>2001</td>
      <td>2.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Nevada</td>
      <td>2002</td>
      <td>2.9</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Nevada</td>
      <td>2003</td>
      <td>3.2</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame.head()
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
      <th>state</th>
      <th>year</th>
      <th>pop</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ohio</td>
      <td>2000</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ohio</td>
      <td>2001</td>
      <td>1.7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ohio</td>
      <td>2002</td>
      <td>3.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Nevada</td>
      <td>2001</td>
      <td>2.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Nevada</td>
      <td>2002</td>
      <td>2.9</td>
    </tr>
  </tbody>
</table>
</div>



The column order can be declared during creation.


```python
pd.DataFrame(data, columns=['year', 'state', 'pop'])
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
      <th>year</th>
      <th>state</th>
      <th>pop</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2000</td>
      <td>Ohio</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2001</td>
      <td>Ohio</td>
      <td>1.7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2002</td>
      <td>Ohio</td>
      <td>3.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2001</td>
      <td>Nevada</td>
      <td>2.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2002</td>
      <td>Nevada</td>
      <td>2.9</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2003</td>
      <td>Nevada</td>
      <td>3.2</td>
    </tr>
  </tbody>
</table>
</div>



This is also a way to declare an empty column.


```python
frame2 = pd.DataFrame(
    data, 
    columns=['year', 'state', 'pop', 'debt'],
    index=['one', 'two', 'three', 'four', 'five', 'six']
)
frame2
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
      <th>year</th>
      <th>state</th>
      <th>pop</th>
      <th>debt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>2000</td>
      <td>Ohio</td>
      <td>1.5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>two</th>
      <td>2001</td>
      <td>Ohio</td>
      <td>1.7</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>three</th>
      <td>2002</td>
      <td>Ohio</td>
      <td>3.6</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>four</th>
      <td>2001</td>
      <td>Nevada</td>
      <td>2.4</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>five</th>
      <td>2002</td>
      <td>Nevada</td>
      <td>2.9</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>six</th>
      <td>2003</td>
      <td>Nevada</td>
      <td>3.2</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame2.columns
```




    Index(['year', 'state', 'pop', 'debt'], dtype='object')




```python
frame2['state']
```




    one        Ohio
    two        Ohio
    three      Ohio
    four     Nevada
    five     Nevada
    six      Nevada
    Name: state, dtype: object




```python
frame2.year
```




    one      2000
    two      2001
    three    2002
    four     2001
    five     2002
    six      2003
    Name: year, dtype: int64




```python
type(frame2.year)
```




    pandas.core.series.Series




```python
type(frame2)
```




    pandas.core.frame.DataFrame



A DataFrame be indexed by column using either a list or dictionary-like syntax.
The rows can be subset using the `loc` method and passing the row index.


```python
frame2['state']
```




    one        Ohio
    two        Ohio
    three      Ohio
    four     Nevada
    five     Nevada
    six      Nevada
    Name: state, dtype: object




```python
frame2.year
```




    one      2000
    two      2001
    three    2002
    four     2001
    five     2002
    six      2003
    Name: year, dtype: int64




```python
frame2.loc['three']
```




    year     2002
    state    Ohio
    pop       3.6
    debt      NaN
    Name: three, dtype: object



Columns can be modified by assignment.


```python
frame2['debt'] = 16.5
frame2
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
      <th>year</th>
      <th>state</th>
      <th>pop</th>
      <th>debt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>2000</td>
      <td>Ohio</td>
      <td>1.5</td>
      <td>16.5</td>
    </tr>
    <tr>
      <th>two</th>
      <td>2001</td>
      <td>Ohio</td>
      <td>1.7</td>
      <td>16.5</td>
    </tr>
    <tr>
      <th>three</th>
      <td>2002</td>
      <td>Ohio</td>
      <td>3.6</td>
      <td>16.5</td>
    </tr>
    <tr>
      <th>four</th>
      <td>2001</td>
      <td>Nevada</td>
      <td>2.4</td>
      <td>16.5</td>
    </tr>
    <tr>
      <th>five</th>
      <td>2002</td>
      <td>Nevada</td>
      <td>2.9</td>
      <td>16.5</td>
    </tr>
    <tr>
      <th>six</th>
      <td>2003</td>
      <td>Nevada</td>
      <td>3.2</td>
      <td>16.5</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame2['debt'] = np.arange(6.0)
frame2
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
      <th>year</th>
      <th>state</th>
      <th>pop</th>
      <th>debt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>2000</td>
      <td>Ohio</td>
      <td>1.5</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>two</th>
      <td>2001</td>
      <td>Ohio</td>
      <td>1.7</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>three</th>
      <td>2002</td>
      <td>Ohio</td>
      <td>3.6</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>four</th>
      <td>2001</td>
      <td>Nevada</td>
      <td>2.4</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>five</th>
      <td>2002</td>
      <td>Nevada</td>
      <td>2.9</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>six</th>
      <td>2003</td>
      <td>Nevada</td>
      <td>3.2</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
val = pd.Series([-1.2, -1.5, -1.7], index=['two', 'four', 'five'])
frame2['debt'] = val
frame2
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
      <th>year</th>
      <th>state</th>
      <th>pop</th>
      <th>debt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>2000</td>
      <td>Ohio</td>
      <td>1.5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>two</th>
      <td>2001</td>
      <td>Ohio</td>
      <td>1.7</td>
      <td>-1.2</td>
    </tr>
    <tr>
      <th>three</th>
      <td>2002</td>
      <td>Ohio</td>
      <td>3.6</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>four</th>
      <td>2001</td>
      <td>Nevada</td>
      <td>2.4</td>
      <td>-1.5</td>
    </tr>
    <tr>
      <th>five</th>
      <td>2002</td>
      <td>Nevada</td>
      <td>2.9</td>
      <td>-1.7</td>
    </tr>
    <tr>
      <th>six</th>
      <td>2003</td>
      <td>Nevada</td>
      <td>3.2</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Columns can be easily created and deleted.


```python
frame2['eastern'] = frame2.state == 'Ohio'
frame2
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
      <th>year</th>
      <th>state</th>
      <th>pop</th>
      <th>debt</th>
      <th>eastern</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>2000</td>
      <td>Ohio</td>
      <td>1.5</td>
      <td>NaN</td>
      <td>True</td>
    </tr>
    <tr>
      <th>two</th>
      <td>2001</td>
      <td>Ohio</td>
      <td>1.7</td>
      <td>-1.2</td>
      <td>True</td>
    </tr>
    <tr>
      <th>three</th>
      <td>2002</td>
      <td>Ohio</td>
      <td>3.6</td>
      <td>NaN</td>
      <td>True</td>
    </tr>
    <tr>
      <th>four</th>
      <td>2001</td>
      <td>Nevada</td>
      <td>2.4</td>
      <td>-1.5</td>
      <td>False</td>
    </tr>
    <tr>
      <th>five</th>
      <td>2002</td>
      <td>Nevada</td>
      <td>2.9</td>
      <td>-1.7</td>
      <td>False</td>
    </tr>
    <tr>
      <th>six</th>
      <td>2003</td>
      <td>Nevada</td>
      <td>3.2</td>
      <td>NaN</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
del frame2['eastern']
```


```python
frame2.columns
```




    Index(['year', 'state', 'pop', 'debt'], dtype='object')



DataFrames can also be created from nested dictionaries.
The first level becomes the column index and the second the row index.


```python
 pop = {
    'Nevada': {2001: 2.4, 2002: 2.9},
    'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}
}
frame3 = pd.DataFrame(pop)
frame3
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
      <th>Nevada</th>
      <th>Ohio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2001</th>
      <td>2.4</td>
      <td>1.7</td>
    </tr>
    <tr>
      <th>2002</th>
      <td>2.9</td>
      <td>3.6</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>NaN</td>
      <td>1.5</td>
    </tr>
  </tbody>
</table>
</div>



A DataFrame can be transposed.


```python
frame3.T
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
      <th>2001</th>
      <th>2002</th>
      <th>2000</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Nevada</th>
      <td>2.4</td>
      <td>2.9</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Ohio</th>
      <td>1.7</td>
      <td>3.6</td>
      <td>1.5</td>
    </tr>
  </tbody>
</table>
</div>



A DataFrame can be created from a dictionary of Series.


```python
pdata = {
    'Ohio': frame3['Ohio'][:-1],
    'Nevada': frame3['Nevada'][:2]
}
pd.DataFrame(pdata)
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
      <th>Ohio</th>
      <th>Nevada</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2001</th>
      <td>1.7</td>
      <td>2.4</td>
    </tr>
    <tr>
      <th>2002</th>
      <td>3.6</td>
      <td>2.9</td>
    </tr>
  </tbody>
</table>
</div>



A DataFrame's `index` and `columns` have their own `name` attribute.


```python
frame3.index.name = 'year'
frame3.columns.name = 'state'
frame3
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
      <th>state</th>
      <th>Nevada</th>
      <th>Ohio</th>
    </tr>
    <tr>
      <th>year</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2001</th>
      <td>2.4</td>
      <td>1.7</td>
    </tr>
    <tr>
      <th>2002</th>
      <td>2.9</td>
      <td>3.6</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>NaN</td>
      <td>1.5</td>
    </tr>
  </tbody>
</table>
</div>



The `values` attribute returns the data as a 2-D ndarray.


```python
frame3.values
```




    array([[2.4, 1.7],
           [2.9, 3.6],
           [nan, 1.5]])




```python
frame2.values
```




    array([[2000, 'Ohio', 1.5, nan],
           [2001, 'Ohio', 1.7, -1.2],
           [2002, 'Ohio', 3.6, nan],
           [2001, 'Nevada', 2.4, -1.5],
           [2002, 'Nevada', 2.9, -1.7],
           [2003, 'Nevada', 3.2, nan]], dtype=object)



### Index Objects

These hold the axis labels and other metadata.
They are immutable.


```python
obj = pd.Series(range(3), index=['a', 'b', 'c'])
index = obj.index
index
```




    Index(['a', 'b', 'c'], dtype='object')




```python
index[1:]
```




    Index(['b', 'c'], dtype='object')



Index objects can be shared amongst data stuctures.


```python
labels = pd.Index(np.arange(3))
labels
```




    Int64Index([0, 1, 2], dtype='int64')




```python
obj2 = pd.Series([1.5, -2.5, 0], index=labels)
obj2
```




    0    1.5
    1   -2.5
    2    0.0
    dtype: float64




```python
obj2.index is labels
```




    True




```python
frame3.columns
```




    Index(['Nevada', 'Ohio'], dtype='object', name='state')




```python
'Ohio' in frame3.columns
```




    True




```python
2003 in frame3.index
```




    False



## 5.2 Eddential functionality

This section discusses the fundamental interations with Series and DataFrames.

### Reindexing

This creates a *new object* with the data conformed to a new index.


```python
obj = pd.Series([4.3, 7.2, -5.3, 3.6], index = ['d', 'b', 'a', 'c'])
obj
```




    d    4.3
    b    7.2
    a   -5.3
    c    3.6
    dtype: float64




```python
obj2 = obj.reindex(['a', 'b', 'c', 'd', 'e'])
obj2
```




    a   -5.3
    b    7.2
    c    3.6
    d    4.3
    e    NaN
    dtype: float64



There is a `method` option to describe how to handle missing data.
Here is an example with `'ffill'` which is a "forward-fill."


```python
obj3 = pd.Series(['blue', 'purple', 'yellow'], index=[0, 2, 4])
obj3
```




    0      blue
    2    purple
    4    yellow
    dtype: object




```python
obj3.reindex(range(6), method='ffill')
```




    0      blue
    1      blue
    2    purple
    3    purple
    4    yellow
    5    yellow
    dtype: object



With DataFrame, 'reindex' can alter either the row index, columns, or both.


```python
frame = pd.DataFrame(
    np.arange(9).reshape((3, 3)),
    index=['a', 'c', 'd'],
    columns=['Ohio', 'Texas', 'California']
)
frame
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
      <th>Ohio</th>
      <th>Texas</th>
      <th>California</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>c</th>
      <td>3</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>d</th>
      <td>6</td>
      <td>7</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame2 = frame.reindex(['a', 'b', 'c', 'd'])
frame2
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
      <th>Ohio</th>
      <th>Texas</th>
      <th>California</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>b</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>c</th>
      <td>3.0</td>
      <td>4.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>d</th>
      <td>6.0</td>
      <td>7.0</td>
      <td>8.0</td>
    </tr>
  </tbody>
</table>
</div>



Columns can be reindexed with the `columns` keyward.


```python
states = ['Texas', 'Utah', 'California']
frame.reindex(columns=states)
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
      <th>Texas</th>
      <th>Utah</th>
      <th>California</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1</td>
      <td>NaN</td>
      <td>2</td>
    </tr>
    <tr>
      <th>c</th>
      <td>4</td>
      <td>NaN</td>
      <td>5</td>
    </tr>
    <tr>
      <th>d</th>
      <td>7</td>
      <td>NaN</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>



Reindexing can be done more succienctly by label-indexing with 'loc'.


```python
frame.loc[['a', 'b', 'c', 'd'], states]
```

    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/pandas/core/indexing.py:1418: FutureWarning: 
    Passing list-likes to .loc or [] with any missing label will raise
    KeyError in the future, you can use .reindex() as an alternative.
    
    See the documentation here:
    https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#deprecate-loc-reindex-listlike
      return self._getitem_tuple(key)





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
      <th>Texas</th>
      <th>Utah</th>
      <th>California</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>b</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>c</th>
      <td>4.0</td>
      <td>NaN</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>d</th>
      <td>7.0</td>
      <td>NaN</td>
      <td>8.0</td>
    </tr>
  </tbody>
</table>
</div>



### Dropping entries from an axis

The `drop` method returns a *new object* with the indicated values deleted from an axis.


```python
obj = pd.Series(np.arange(5.0), index=['a', 'b', 'c', 'd', 'e'])
obj
```




    a    0.0
    b    1.0
    c    2.0
    d    3.0
    e    4.0
    dtype: float64




```python
new_obj = obj.drop('c')
new_obj
```




    a    0.0
    b    1.0
    d    3.0
    e    4.0
    dtype: float64




```python
obj.drop(['d', 'c'])
```




    a    0.0
    b    1.0
    e    4.0
    dtype: float64



With DataFrame, index values can be deleted from either axis.


```python
data = pd.DataFrame(
    np.arange(16).reshape((4, 4)),
    index=['Ohio', 'Colorado', 'Utah', 'New York'],
    columns=['one', 'two', 'three', 'four']
)
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
      <th>Utah</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.drop(['Colorado', 'Ohio'])
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
      <th>Utah</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.drop('two', axis=1)
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
      <th>three</th>
      <th>four</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ohio</th>
      <td>0</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Colorado</th>
      <td>4</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>Utah</th>
      <td>8</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>12</td>
      <td>14</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.drop(['two', 'four'], axis='columns')
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
      <th>three</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ohio</th>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Colorado</th>
      <td>4</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Utah</th>
      <td>8</td>
      <td>10</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>12</td>
      <td>14</td>
    </tr>
  </tbody>
</table>
</div>



There is an option to make the change in-place.


```python
obj.drop('c', inplace=True)
obj
```




    a    0.0
    b    1.0
    d    3.0
    e    4.0
    dtype: float64



### Indexing, selection, and filtering

For Series, can use either position, boolean, or index values for indexing.


```python
obj = pd.Series(np.arange(4.0), index=['a', 'b', 'c', 'd'])
obj
```




    a    0.0
    b    1.0
    c    2.0
    d    3.0
    dtype: float64




```python
obj['b']
```




    1.0




```python
obj[1]
```




    1.0




```python
obj[2:4]
```




    c    2.0
    d    3.0
    dtype: float64




```python
obj[['b', 'a', 'd']]
```




    b    1.0
    a    0.0
    d    3.0
    dtype: float64




```python
obj[[1, 3]]
```




    b    1.0
    d    3.0
    dtype: float64




```python
obj[obj < 2]
```




    a    0.0
    b    1.0
    dtype: float64




```python
obj['b':'c']
```




    b    1.0
    c    2.0
    dtype: float64




```python
obj['b':'c'] = 5
obj
```




    a    0.0
    b    5.0
    c    5.0
    d    3.0
    dtype: float64



Indexing a DataFrame can retrieve multiple columns.


```python
data = pd.DataFrame(
    np.arange(16).reshape((4, 4)),
    index=['Ohio', 'Colorado', 'Utah', 'New York'],
    columns=['one', 'two', 'three', 'four']
)
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
      <th>Utah</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
data['two']
```




    Ohio         1
    Colorado     5
    Utah         9
    New York    13
    Name: two, dtype: int64




```python
data[['three', 'one']]
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
      <th>three</th>
      <th>one</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ohio</th>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Colorado</th>
      <td>6</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Utah</th>
      <td>10</td>
      <td>8</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>14</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>




```python
data[:2]
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
  </tbody>
</table>
</div>




```python
data < 5
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
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>Colorado</th>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>Utah</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
data[data['three'] > 5]
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
      <th>Colorado</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>Utah</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
data[data < 5] = 0
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Colorado</th>
      <td>0</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>Utah</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>



`loc` and `iloc` are methods specifically for label-indexing on the rows of a DataFrame.


```python
data.loc['Colorado', ['two', 'three']]
```




    two      5
    three    6
    Name: Colorado, dtype: int64




```python
data.iloc[2, [3, 0, 1]]
```




    four    11
    one      8
    two      9
    Name: Utah, dtype: int64




```python
data.iloc[2]
```




    one       8
    two       9
    three    10
    four     11
    Name: Utah, dtype: int64




```python
data.iloc[[1, 2], [3, 0, 1]]
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
      <th>four</th>
      <th>one</th>
      <th>two</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Colorado</th>
      <td>7</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Utah</th>
      <td>11</td>
      <td>8</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.loc[:'Utah', 'two']
```




    Ohio        0
    Colorado    5
    Utah        9
    Name: two, dtype: int64




```python
data.iloc[:, :3][data.three > 5]
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Colorado</th>
      <td>0</td>
      <td>5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Utah</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
    </tr>
  </tbody>
</table>
</div>



### Arithmetic and data alignment

When adding objects together, if any index pairs are not the same, the respective index in the result will be the union of the index pairs.


```python
s1 = pd.Series([7.3, -2.5, 3.4, 1.5], index=['a', 'c', 'd', 'e'])
s2 = pd.Series([-2.1, 3.6, -1.5, 4, 3.1], index=['a', 'c', 'e', 'f', 'g'])
s1
```




    a    7.3
    c   -2.5
    d    3.4
    e    1.5
    dtype: float64




```python
s2
```




    a   -2.1
    c    3.6
    e   -1.5
    f    4.0
    g    3.1
    dtype: float64



`NaN` are returned for when there is only one value being added together.


```python
s1 + s2
```




    a    5.2
    c    1.1
    d    NaN
    e    0.0
    f    NaN
    g    NaN
    dtype: float64



For DataFrame, alignment is performed on both rows and columns.


```python
df1 = pd.DataFrame(np.arange(9.).reshape((3, 3)), 
                   columns=list('bcd'), 
                   index=['Ohio', 'Texas', 'Colorado'])
df2 = pd.DataFrame(np.arange(12.).reshape((4, 3)), 
                   columns=list('bde'),
                   index=['Utah', 'Ohio', 'Texas', 'Oregon'])
df1
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
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ohio</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>3.0</td>
      <td>4.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>Colorado</th>
      <td>6.0</td>
      <td>7.0</td>
      <td>8.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2
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
      <th>b</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Utah</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>Ohio</th>
      <td>3.0</td>
      <td>4.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>6.0</td>
      <td>7.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>Oregon</th>
      <td>9.0</td>
      <td>10.0</td>
      <td>11.0</td>
    </tr>
  </tbody>
</table>
</div>



When added together, values are only returned for positions in both DataFrames.


```python
df1 + df2
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
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Colorado</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Ohio</th>
      <td>3.0</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Oregon</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>9.0</td>
      <td>NaN</td>
      <td>12.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Utah</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



There is an option to fill missing data with a specific value to be used for the operation.


```python
df1 = pd.DataFrame(np.arange(12.).reshape((3, 4)), columns=list('abcd'))
df2 = pd.DataFrame(np.arange(20.).reshape((4, 5)), columns=list('abcde'))
df1
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
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8.0</td>
      <td>9.0</td>
      <td>10.0</td>
      <td>11.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2.loc[1, 'b'] = np.nan
df2
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
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.0</td>
      <td>NaN</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10.0</td>
      <td>11.0</td>
      <td>12.0</td>
      <td>13.0</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15.0</td>
      <td>16.0</td>
      <td>17.0</td>
      <td>18.0</td>
      <td>19.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df1 + df2
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
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9.0</td>
      <td>NaN</td>
      <td>13.0</td>
      <td>15.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>20.0</td>
      <td>22.0</td>
      <td>24.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df1.add(df2, fill_value=0)
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
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9.0</td>
      <td>5.0</td>
      <td>13.0</td>
      <td>15.0</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>20.0</td>
      <td>22.0</td>
      <td>24.0</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15.0</td>
      <td>16.0</td>
      <td>17.0</td>
      <td>18.0</td>
      <td>19.0</td>
    </tr>
  </tbody>
</table>
</div>



There are defined behaviours for arithmetic between DataFrame and Series.
It is slightly different than for 1D and multidimensional ndarrys.


```python
arr = np.arange(12.0).reshape((3, 4))
arr
```




    array([[ 0.,  1.,  2.,  3.],
           [ 4.,  5.,  6.,  7.],
           [ 8.,  9., 10., 11.]])




```python
arr[0]
```




    array([0., 1., 2., 3.])



The subtration below results in one operation per row (an example of broadcasting).


```python
arr - arr[0]
```




    array([[0., 0., 0., 0.],
           [4., 4., 4., 4.],
           [8., 8., 8., 8.]])



A simillar mechanism is used for operations between a Series and DataFrame.


```python
frame = pd.DataFrame(np.arange(12.).reshape((4, 3)),
                     columns=list('bde'),
                     index=['Utah', 'Ohio', 'Texas', 'Oregon'])
frame
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
      <th>b</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Utah</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>Ohio</th>
      <td>3.0</td>
      <td>4.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>6.0</td>
      <td>7.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>Oregon</th>
      <td>9.0</td>
      <td>10.0</td>
      <td>11.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
series = frame.iloc[0]
series
```




    b    0.0
    d    1.0
    e    2.0
    Name: Utah, dtype: float64




```python
frame - series
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
      <th>b</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Utah</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Ohio</th>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>6.0</td>
      <td>6.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>Oregon</th>
      <td>9.0</td>
      <td>9.0</td>
      <td>9.0</td>
    </tr>
  </tbody>
</table>
</div>



IF an index is not found in either the DataFrame columns or Series index, the objects are reindexed to form the union.


```python
series2 = pd.Series(range(3), index=['b', 'e', 'f'])
series2
```




    b    0
    e    1
    f    2
    dtype: int64




```python
frame + series2
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
      <th>b</th>
      <th>d</th>
      <th>e</th>
      <th>f</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Utah</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Ohio</th>
      <td>3.0</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>6.0</td>
      <td>NaN</td>
      <td>9.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Oregon</th>
      <td>9.0</td>
      <td>NaN</td>
      <td>12.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
series3 = frame['d']
series3
```




    Utah       1.0
    Ohio       4.0
    Texas      7.0
    Oregon    10.0
    Name: d, dtype: float64




```python
frame.sub(series3)
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
      <th>Ohio</th>
      <th>Oregon</th>
      <th>Texas</th>
      <th>Utah</th>
      <th>b</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Utah</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Ohio</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Oregon</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame.sub(series3, axis='index')
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
      <th>b</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Utah</th>
      <td>-1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Ohio</th>
      <td>-1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>-1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Oregon</th>
      <td>-1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



### Function application and mapping

DataFrames work well with ufuncs, too.


```python
frame = pd.DataFrame(np.random.randn(4, 3), 
                     columns=list('bde'),
                     index=['Utah', 'Ohio', 'Texas', 'Oregon'])
frame
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
      <th>b</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Utah</th>
      <td>-1.167574</td>
      <td>-1.923661</td>
      <td>-0.856915</td>
    </tr>
    <tr>
      <th>Ohio</th>
      <td>-0.712398</td>
      <td>0.835224</td>
      <td>1.783416</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>0.283403</td>
      <td>1.123811</td>
      <td>-1.542719</td>
    </tr>
    <tr>
      <th>Oregon</th>
      <td>-1.049275</td>
      <td>-0.444459</td>
      <td>-0.123116</td>
    </tr>
  </tbody>
</table>
</div>




```python
np.abs(frame)
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
      <th>b</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Utah</th>
      <td>1.167574</td>
      <td>1.923661</td>
      <td>0.856915</td>
    </tr>
    <tr>
      <th>Ohio</th>
      <td>0.712398</td>
      <td>0.835224</td>
      <td>1.783416</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>0.283403</td>
      <td>1.123811</td>
      <td>1.542719</td>
    </tr>
    <tr>
      <th>Oregon</th>
      <td>1.049275</td>
      <td>0.444459</td>
      <td>0.123116</td>
    </tr>
  </tbody>
</table>
</div>



Use the `apply` method to apply a function to the 1D arrays from columns or rows.


```python
f = lambda x: x.max() - x.min()

frame.apply(f)
```




    b    1.450977
    d    3.047472
    e    3.326136
    dtype: float64



To operate of the rows, pass the value `axis=1`.


```python
frame.apply(f, axis=1)
```




    Utah      1.066746
    Ohio      2.495815
    Texas     2.666531
    Oregon    0.926159
    dtype: float64




```python
def f(x):
    return pd.Series([x.min(), x.max()], index=['min', 'max'])

frame.apply(f)
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
      <th>b</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>min</th>
      <td>-1.167574</td>
      <td>-1.923661</td>
      <td>-1.542719</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.283403</td>
      <td>1.123811</td>
      <td>1.783416</td>
    </tr>
  </tbody>
</table>
</div>



### Sorting and ranking

Use the `sort_index()` method to sort a Series or DataFrame lexicographically.


```python
obj = pd.Series(range(4), index=['d', 'a', 'b', 'c'])
obj.sort_index()
```




    a    1
    b    2
    c    3
    d    0
    dtype: int64




```python
frame = pd.DataFrame(np.arange(8).reshape((2, 4)),
                     index=['three', 'one'],
                     columns=['d', 'a', 'b', 'c'])
frame.sort_index()
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
      <th>d</th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>three</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame.sort_index(axis=1)
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
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>three</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>one</th>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame.sort_index(axis=1, ascending=False)
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
      <th>d</th>
      <th>c</th>
      <th>b</th>
      <th>a</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>three</th>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>one</th>
      <td>4</td>
      <td>7</td>
      <td>6</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



Use the `sort_values` method to sort a Series by its values.


```python
obj = pd.Series([4, 7, -3, -2])
obj.sort_values()
```




    2   -3
    3   -2
    0    4
    1    7
    dtype: int64




```python
obj = pd.Series([4, np.nan, 7, np.nan, -3, 2])
obj.sort_values()
```




    4   -3.0
    5    2.0
    0    4.0
    2    7.0
    1    NaN
    3    NaN
    dtype: float64



For a DataFrame, a column can be specified to use for sorting.


```python
frame = pd.DataFrame({
    'b': [4, 7, -3, 2],
    'a': [0, 1, 0, 1]
})
frame.sort_values(by='b')
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
      <th>b</th>
      <th>a</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>-3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame.sort_values(by=['a', 'b'])
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
      <th>b</th>
      <th>a</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>-3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Ranking assigns ranks from 1 through the number of valid data points (rows).
There are a few different ways to handle ties and they can be declared using the `method` argument.


```python
obj = pd.Series([7, -5, 7, 4, 3, 0, 4])
obj.rank()
```




    0    6.5
    1    1.0
    2    6.5
    3    4.5
    4    3.0
    5    2.0
    6    4.5
    dtype: float64




```python
obj.rank(method='first')
```




    0    6.0
    1    1.0
    2    7.0
    3    4.0
    4    3.0
    5    2.0
    6    5.0
    dtype: float64




```python
obj.rank(pct=True)
```




    0    0.928571
    1    0.142857
    2    0.928571
    3    0.642857
    4    0.428571
    5    0.285714
    6    0.642857
    dtype: float64




```python
obj.rank(ascending=False, method='max')
```




    0    2.0
    1    7.0
    2    2.0
    3    4.0
    4    5.0
    5    6.0
    6    4.0
    dtype: float64



DataFrames can be ranked over the columns are rows.


```python
frame = pd.DataFrame({
    'b': [4.3, 7, -3, 2],
    'a': [0, 1, 0, 1],
    'c': [-2, 5, 8, -2.5]
}).sort_index(axis=1)
frame
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
      <td>4.3</td>
      <td>-2.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>7.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>-3.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>2.0</td>
      <td>-2.5</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame.rank()
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
      <td>1.5</td>
      <td>3.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3.5</td>
      <td>4.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.5</td>
      <td>1.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.5</td>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame.rank(axis=1)
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
      <td>2.0</td>
      <td>3.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>3.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>1.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.0</td>
      <td>3.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



### Axis indexes with duplicate labels

Many pandas functions require the labels to be unique, but it is not mandatory for a Series or DataFrame.
HEre is a small example Series with non-unique labels.


```python
obj = pd.Series(range(5), index=['a', 'a', 'b', 'b', 'c'])
obj
```




    a    0
    a    1
    b    2
    b    3
    c    4
    dtype: int64




```python
obj.index.is_unique
```




    False




```python
obj.a
```




    a    0
    a    1
    dtype: int64




```python
obj.c
```




    4



Here is an example with DataFrame.


```python
df = pd.DataFrame(np.random.randn(4, 3), index=['a', 'b', 'b', 'b'])
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
      <th>a</th>
      <td>0.315786</td>
      <td>-0.868693</td>
      <td>-1.045036</td>
    </tr>
    <tr>
      <th>b</th>
      <td>1.144670</td>
      <td>0.902213</td>
      <td>-2.583696</td>
    </tr>
    <tr>
      <th>b</th>
      <td>-0.499635</td>
      <td>0.883807</td>
      <td>-1.347935</td>
    </tr>
    <tr>
      <th>b</th>
      <td>-1.053965</td>
      <td>-0.075102</td>
      <td>0.066315</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.loc['b']
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
      <th>b</th>
      <td>1.144670</td>
      <td>0.902213</td>
      <td>-2.583696</td>
    </tr>
    <tr>
      <th>b</th>
      <td>-0.499635</td>
      <td>0.883807</td>
      <td>-1.347935</td>
    </tr>
    <tr>
      <th>b</th>
      <td>-1.053965</td>
      <td>-0.075102</td>
      <td>0.066315</td>
    </tr>
  </tbody>
</table>
</div>





## 5.3 Summarizing and computing descriptive statistics

Pandas has many methods for computing reductions and summarising the data in a DataFrame.
Importantly, they naturally handle missing data.


```python
df = pd.DataFrame([[1.4, np.nan],
                   [7.1, -4.5],
                   [np.nan, np.nan],
                   [0.75, -1.3]],
                  index=['a', 'b', 'c', 'd'],
                  columns=['one', 'two'])
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
      <th>one</th>
      <th>two</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1.40</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>b</th>
      <td>7.10</td>
      <td>-4.5</td>
    </tr>
    <tr>
      <th>c</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>d</th>
      <td>0.75</td>
      <td>-1.3</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.sum()
```




    one    9.25
    two   -5.80
    dtype: float64




```python
df.sum(axis=1)
```




    a    1.40
    b    2.60
    c    0.00
    d   -0.55
    dtype: float64




```python
df.sum(axis=1, skipna=False)
```




    a     NaN
    b    2.60
    c     NaN
    d   -0.55
    dtype: float64



There are some methods for getting the index value of key data.


```python
df.idxmax()
```




    one    b
    two    d
    dtype: object




```python
df.cumsum()
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1.40</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>b</th>
      <td>8.50</td>
      <td>-4.5</td>
    </tr>
    <tr>
      <th>c</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>d</th>
      <td>9.25</td>
      <td>-5.8</td>
    </tr>
  </tbody>
</table>
</div>



The `describe` method is useful to getting a high-level overview of the data in a DataFrame.


```python
df.describe()
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.083333</td>
      <td>-2.900000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.493685</td>
      <td>2.262742</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.750000</td>
      <td>-4.500000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.075000</td>
      <td>-3.700000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.400000</td>
      <td>-2.900000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.250000</td>
      <td>-2.100000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.100000</td>
      <td>-1.300000</td>
    </tr>
  </tbody>
</table>
</div>



For non-numeric data.


```python
obj = pd.Series(['a', 'a', 'b', 'c'] * 4)
obj
```




    0     a
    1     a
    2     b
    3     c
    4     a
    5     a
    6     b
    7     c
    8     a
    9     a
    10    b
    11    c
    12    a
    13    a
    14    b
    15    c
    dtype: object




```python
obj.describe
```




    <bound method NDFrame.describe of 0     a
    1     a
    2     b
    3     c
    4     a
    5     a
    6     b
    7     c
    8     a
    9     a
    10    b
    11    c
    12    a
    13    a
    14    b
    15    c
    dtype: object>



Below are examples a some of the more common summary statistics.


```python
df = pd.DataFrame(np.random.randn(100).reshape((20, 5)))
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
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.045266</td>
      <td>1.282355</td>
      <td>2.090783</td>
      <td>-0.252437</td>
      <td>1.040161</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.150544</td>
      <td>-0.147333</td>
      <td>0.320805</td>
      <td>1.567862</td>
      <td>-0.174218</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.166097</td>
      <td>-1.042703</td>
      <td>-0.079506</td>
      <td>-0.118705</td>
      <td>0.123881</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.043958</td>
      <td>2.006066</td>
      <td>0.471954</td>
      <td>0.686379</td>
      <td>0.540834</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.244186</td>
      <td>-0.874540</td>
      <td>-1.410670</td>
      <td>-1.593762</td>
      <td>0.318683</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.071923</td>
      <td>0.148731</td>
      <td>-0.456628</td>
      <td>-1.155017</td>
      <td>0.624898</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.083861</td>
      <td>0.218961</td>
      <td>2.650821</td>
      <td>-0.113464</td>
      <td>-0.422801</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-0.289215</td>
      <td>3.006028</td>
      <td>-2.028504</td>
      <td>0.290864</td>
      <td>0.984765</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.866779</td>
      <td>1.039549</td>
      <td>0.772484</td>
      <td>-1.578464</td>
      <td>-0.247977</td>
    </tr>
    <tr>
      <th>9</th>
      <td>-0.079468</td>
      <td>0.327789</td>
      <td>0.740373</td>
      <td>0.268327</td>
      <td>-1.121380</td>
    </tr>
    <tr>
      <th>10</th>
      <td>-0.728884</td>
      <td>-1.385715</td>
      <td>1.322314</td>
      <td>-0.502122</td>
      <td>0.456615</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.460693</td>
      <td>2.520307</td>
      <td>-0.929717</td>
      <td>1.136553</td>
      <td>1.818280</td>
    </tr>
    <tr>
      <th>12</th>
      <td>-0.226714</td>
      <td>-0.790623</td>
      <td>0.940863</td>
      <td>-0.279578</td>
      <td>0.408899</td>
    </tr>
    <tr>
      <th>13</th>
      <td>-2.252070</td>
      <td>-0.513878</td>
      <td>-1.115919</td>
      <td>-1.057158</td>
      <td>-0.112186</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2.193616</td>
      <td>1.330908</td>
      <td>0.217283</td>
      <td>0.489868</td>
      <td>-0.011269</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1.823363</td>
      <td>1.266734</td>
      <td>-1.286281</td>
      <td>0.673735</td>
      <td>-0.398431</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.780628</td>
      <td>1.615514</td>
      <td>0.525163</td>
      <td>-0.062664</td>
      <td>0.088914</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2.037299</td>
      <td>1.412365</td>
      <td>-0.284433</td>
      <td>0.963503</td>
      <td>0.373894</td>
    </tr>
    <tr>
      <th>18</th>
      <td>-0.214364</td>
      <td>2.878076</td>
      <td>-0.450742</td>
      <td>-0.346410</td>
      <td>0.520312</td>
    </tr>
    <tr>
      <th>19</th>
      <td>-0.610795</td>
      <td>-1.430337</td>
      <td>0.121110</td>
      <td>0.483379</td>
      <td>2.001245</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.count()  # number of non-NA values
```




    0    20
    1    20
    2    20
    3    20
    4    20
    dtype: int64




```python
df.mean()
```




    0    0.253834
    1    0.643413
    2    0.106578
    3   -0.024966
    4    0.340656
    dtype: float64




```python
df.median()
```




    0   -0.057940
    1    0.683669
    2    0.169197
    3   -0.088064
    4    0.346288
    dtype: float64




```python
df.mad()
```




    0    0.854199
    1    1.192377
    2    0.900160
    3    0.678516
    4    0.536334
    dtype: float64




```python
df.prod()
```




    0   -1.721891e-07
    1   -2.590494e-01
    2   -1.466648e-04
    3   -4.670023e-07
    4   -1.657156e-09
    dtype: float64




```python
df.std()
```




    0    1.094775
    1    1.396016
    2    1.169365
    3    0.866003
    4    0.737545
    dtype: float64




```python
df.var()
```




    0    1.198532
    1    1.948861
    2    1.367415
    3    0.749961
    4    0.543972
    dtype: float64




```python
df.cumsum()
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
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.045266</td>
      <td>1.282355</td>
      <td>2.090783</td>
      <td>-0.252437</td>
      <td>1.040161</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.894723</td>
      <td>1.135022</td>
      <td>2.411587</td>
      <td>1.315425</td>
      <td>0.865944</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.271374</td>
      <td>0.092320</td>
      <td>2.332081</td>
      <td>1.196720</td>
      <td>0.989825</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.227416</td>
      <td>2.098386</td>
      <td>2.804035</td>
      <td>1.883099</td>
      <td>1.530659</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.471602</td>
      <td>1.223846</td>
      <td>1.393365</td>
      <td>0.289337</td>
      <td>1.849342</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.399679</td>
      <td>1.372576</td>
      <td>0.936738</td>
      <td>-0.865680</td>
      <td>2.474240</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.315818</td>
      <td>1.591538</td>
      <td>3.587559</td>
      <td>-0.979144</td>
      <td>2.051439</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1.026603</td>
      <td>4.597565</td>
      <td>1.559055</td>
      <td>-0.688280</td>
      <td>3.036203</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.893382</td>
      <td>5.637114</td>
      <td>2.331539</td>
      <td>-2.266744</td>
      <td>2.788226</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1.813915</td>
      <td>5.964904</td>
      <td>3.071912</td>
      <td>-1.998416</td>
      <td>1.666846</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1.085031</td>
      <td>4.579189</td>
      <td>4.394226</td>
      <td>-2.500538</td>
      <td>2.123461</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1.545723</td>
      <td>7.099496</td>
      <td>3.464508</td>
      <td>-1.363985</td>
      <td>3.941741</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1.319009</td>
      <td>6.308873</td>
      <td>4.405372</td>
      <td>-1.643564</td>
      <td>4.350640</td>
    </tr>
    <tr>
      <th>13</th>
      <td>-0.933061</td>
      <td>5.794995</td>
      <td>3.289452</td>
      <td>-2.700722</td>
      <td>4.238454</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1.260555</td>
      <td>7.125903</td>
      <td>3.506736</td>
      <td>-2.210854</td>
      <td>4.227184</td>
    </tr>
    <tr>
      <th>15</th>
      <td>3.083918</td>
      <td>8.392638</td>
      <td>2.220455</td>
      <td>-1.537119</td>
      <td>3.828753</td>
    </tr>
    <tr>
      <th>16</th>
      <td>3.864546</td>
      <td>10.008152</td>
      <td>2.745618</td>
      <td>-1.599783</td>
      <td>3.917667</td>
    </tr>
    <tr>
      <th>17</th>
      <td>5.901845</td>
      <td>11.420517</td>
      <td>2.461186</td>
      <td>-0.636280</td>
      <td>4.291561</td>
    </tr>
    <tr>
      <th>18</th>
      <td>5.687480</td>
      <td>14.298593</td>
      <td>2.010443</td>
      <td>-0.982690</td>
      <td>4.811873</td>
    </tr>
    <tr>
      <th>19</th>
      <td>5.076686</td>
      <td>12.868256</td>
      <td>2.131554</td>
      <td>-0.499311</td>
      <td>6.813118</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.cumprod()
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
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.045266e+00</td>
      <td>1.282355</td>
      <td>2.090783</td>
      <td>-2.524371e-01</td>
      <td>1.040161e+00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.573583e-01</td>
      <td>-0.188933</td>
      <td>0.670733</td>
      <td>-3.957865e-01</td>
      <td>-1.812143e-01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.834949e-01</td>
      <td>0.197001</td>
      <td>-0.053328</td>
      <td>4.698193e-02</td>
      <td>-2.244901e-02</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8.066110e-03</td>
      <td>0.395197</td>
      <td>-0.025168</td>
      <td>3.224742e-02</td>
      <td>-1.214120e-02</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.003574e-02</td>
      <td>-0.345616</td>
      <td>0.035504</td>
      <td>-5.139471e-02</td>
      <td>-3.869194e-03</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-7.217967e-04</td>
      <td>-0.051404</td>
      <td>-0.016212</td>
      <td>5.936175e-02</td>
      <td>-2.417850e-03</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6.053054e-05</td>
      <td>-0.011255</td>
      <td>-0.042975</td>
      <td>-6.735445e-03</td>
      <td>1.022269e-03</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-1.750635e-05</td>
      <td>-0.033834</td>
      <td>0.087176</td>
      <td>-1.959101e-03</td>
      <td>1.006695e-03</td>
    </tr>
    <tr>
      <th>8</th>
      <td>-1.517415e-05</td>
      <td>-0.035172</td>
      <td>0.067342</td>
      <td>3.092369e-03</td>
      <td>-2.496376e-04</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1.205853e-06</td>
      <td>-0.011529</td>
      <td>0.049858</td>
      <td>8.297670e-04</td>
      <td>2.799386e-04</td>
    </tr>
    <tr>
      <th>10</th>
      <td>-8.789272e-07</td>
      <td>0.015976</td>
      <td>0.065928</td>
      <td>-4.166443e-04</td>
      <td>1.278243e-04</td>
    </tr>
    <tr>
      <th>11</th>
      <td>-4.049152e-07</td>
      <td>0.040264</td>
      <td>-0.061294</td>
      <td>-4.735383e-04</td>
      <td>2.324203e-04</td>
    </tr>
    <tr>
      <th>12</th>
      <td>9.180015e-08</td>
      <td>-0.031834</td>
      <td>-0.057670</td>
      <td>1.323911e-04</td>
      <td>9.503635e-05</td>
    </tr>
    <tr>
      <th>13</th>
      <td>-2.067404e-07</td>
      <td>0.016359</td>
      <td>0.064355</td>
      <td>-1.399583e-04</td>
      <td>-1.066178e-05</td>
    </tr>
    <tr>
      <th>14</th>
      <td>-4.535091e-07</td>
      <td>0.021772</td>
      <td>0.013983</td>
      <td>-6.856113e-05</td>
      <td>1.201514e-07</td>
    </tr>
    <tr>
      <th>15</th>
      <td>-8.269117e-07</td>
      <td>0.027579</td>
      <td>-0.017986</td>
      <td>-4.619204e-05</td>
      <td>-4.787200e-08</td>
    </tr>
    <tr>
      <th>16</th>
      <td>-6.455103e-07</td>
      <td>0.044555</td>
      <td>-0.009446</td>
      <td>2.894595e-06</td>
      <td>-4.256488e-09</td>
    </tr>
    <tr>
      <th>17</th>
      <td>-1.315097e-06</td>
      <td>0.062928</td>
      <td>0.002687</td>
      <td>2.788951e-06</td>
      <td>-1.591474e-09</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2.819100e-07</td>
      <td>0.181111</td>
      <td>-0.001211</td>
      <td>-9.661200e-07</td>
      <td>-8.280626e-10</td>
    </tr>
    <tr>
      <th>19</th>
      <td>-1.721891e-07</td>
      <td>-0.259049</td>
      <td>-0.000147</td>
      <td>-4.670023e-07</td>
      <td>-1.657156e-09</td>
    </tr>
  </tbody>
</table>
</div>



### Correlation and covariance

Here, the author showed an example of using pandas to compute correlation and covariance measures between stocks.
The first step was to download the data from Yahoo! Finance


```python
import pandas_datareader.data as web

stock_names = ['AAPL', 'IBM', 'MSFT', 'GOOG']
all_data = {ticker: web.get_data_yahoo(ticker) for ticker in stock_names}
all_data
```




    {'AAPL':                   High         Low        Open       Close        Volume  \
     Date                                                                       
     2014-11-03  110.300003  108.010002  108.220001  109.400002  5.228260e+07   
     2014-11-04  109.489998  107.720001  109.360001  108.599998  4.157440e+07   
     2014-11-05  109.300003  108.129997  109.099998  108.860001  3.743590e+07   
     2014-11-06  108.790001  107.800003  108.599998  108.699997  3.496850e+07   
     2014-11-07  109.320000  108.550003  108.750000  109.010002  3.369150e+07   
     ...                ...         ...         ...         ...           ...   
     2019-10-28  249.250000  246.720001  247.419998  249.050003  2.414320e+07   
     2019-10-29  249.750000  242.570007  248.970001  243.289993  3.566010e+07   
     2019-10-30  245.300003  241.210007  244.759995  243.259995  3.095060e+07   
     2019-10-31  249.169998  237.259995  247.240005  248.759995  3.476660e+09   
     2019-11-01  255.929993  249.160004  249.539993  255.820007  3.773870e+07   
     
                  Adj Close  
     Date                    
     2014-11-03  100.385109  
     2014-11-04   99.651047  
     2014-11-05   99.889618  
     2014-11-06  100.175293  
     2014-11-07  100.460983  
     ...                ...  
     2019-10-28  249.050003  
     2019-10-29  243.289993  
     2019-10-30  243.259995  
     2019-10-31  248.759995  
     2019-11-01  255.820007  
     
     [1259 rows x 6 columns],
     'IBM':                   High         Low        Open       Close       Volume  \
     Date                                                                      
     2014-11-03  164.539993  163.380005  164.250000  164.360001    4688200.0   
     2014-11-04  164.360001  162.240005  164.339996  162.649994    4246900.0   
     2014-11-05  163.539993  161.559998  163.130005  161.820007    4104700.0   
     2014-11-06  161.529999  160.050003  161.279999  161.460007    4067600.0   
     2014-11-07  162.210007  160.850006  161.419998  162.070007    3494800.0   
     ...                ...         ...         ...         ...          ...   
     2019-10-28  136.630005  135.449997  136.000000  135.970001    3225700.0   
     2019-10-29  135.570007  133.440002  135.419998  133.820007    4159600.0   
     2019-10-30  135.279999  133.199997  133.830002  135.250000    2252700.0   
     2019-10-31  135.250000  133.229996  135.110001  133.729996  341090000.0   
     2019-11-01  135.559998  134.089996  134.500000  135.529999    3088800.0   
     
                  Adj Close  
     Date                    
     2014-11-03  133.097809  
     2014-11-04  131.713089  
     2014-11-05  131.040939  
     2014-11-06  131.644287  
     2014-11-07  132.141617  
     ...                ...  
     2019-10-28  135.970001  
     2019-10-29  133.820007  
     2019-10-30  135.250000  
     2019-10-31  133.729996  
     2019-11-01  135.529999  
     
     [1259 rows x 6 columns],
     'MSFT':                   High         Low        Open       Close        Volume  \
     Date                                                                       
     2014-11-03   47.459999   46.730000   46.889999   47.439999  2.313040e+07   
     2014-11-04   47.730000   47.250000   47.299999   47.570000  2.153080e+07   
     2014-11-05   47.900002   47.259998   47.799999   47.860001  2.244960e+07   
     2014-11-06   48.860001   47.790001   47.860001   48.700001  3.303780e+07   
     2014-11-07   48.919998   48.290001   48.919998   48.680000  2.800060e+07   
     ...                ...         ...         ...         ...           ...   
     2019-10-28  145.669998  143.509995  144.399994  144.190002  3.528010e+07   
     2019-10-29  144.500000  142.649994  144.080002  142.830002  2.051970e+07   
     2019-10-30  145.000000  142.789993  143.520004  144.610001  1.847170e+07   
     2019-10-31  144.929993  142.990005  144.899994  143.369995  2.459620e+09   
     2019-11-01  144.419998  142.970001  144.259995  143.720001  3.311920e+07   
     
                  Adj Close  
     Date                    
     2014-11-03   42.466732  
     2014-11-04   42.583115  
     2014-11-05   42.842709  
     2014-11-06   43.594646  
     2014-11-07   43.576752  
     ...                ...  
     2019-10-28  144.190002  
     2019-10-29  142.830002  
     2019-10-30  144.610001  
     2019-10-31  143.369995  
     2019-11-01  143.720001  
     
     [1259 rows x 6 columns],
     'GOOG':                    High          Low         Open        Close       Volume  \
     Date                                                                          
     2014-11-03   556.372498   551.715271   553.979065   553.699829    1382200.0   
     2014-11-04   553.979065   547.796021   551.485901   552.592834    1244200.0   
     2014-11-05   555.275513   542.560425   555.275513   544.425293    2032200.0   
     2014-11-06   545.387634   539.488831   544.006409   540.555908    1333200.0   
     2014-11-07   544.714478   537.195129   544.714478   539.528748    1633700.0   
     ...                 ...          ...          ...          ...          ...   
     2019-10-28  1299.310059  1272.540039  1275.449951  1290.000000    2613200.0   
     2019-10-29  1281.589966  1257.212036  1276.229980  1262.619995    1869200.0   
     2019-10-30  1269.359985  1252.000000  1252.969971  1261.290039    1407700.0   
     2019-10-31  1267.670044  1250.843018  1261.280029  1260.109985  145470000.0   
     2019-11-01  1274.619995  1260.500000  1265.000000  1273.739990    1669400.0   
     
                   Adj Close  
     Date                     
     2014-11-03   553.699829  
     2014-11-04   552.592834  
     2014-11-05   544.425293  
     2014-11-06   540.555908  
     2014-11-07   539.528748  
     ...                 ...  
     2019-10-28  1290.000000  
     2019-10-29  1262.619995  
     2019-10-30  1261.290039  
     2019-10-31  1260.109985  
     2019-11-01  1273.739990  
     
     [1259 rows x 6 columns]}




```python
price = pd.DataFrame({ticker: data['Adj Close'] for ticker, data in all_data.items()})
price
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
      <th>AAPL</th>
      <th>IBM</th>
      <th>MSFT</th>
      <th>GOOG</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2014-11-03</th>
      <td>100.385109</td>
      <td>133.097809</td>
      <td>42.466732</td>
      <td>553.699829</td>
    </tr>
    <tr>
      <th>2014-11-04</th>
      <td>99.651047</td>
      <td>131.713089</td>
      <td>42.583115</td>
      <td>552.592834</td>
    </tr>
    <tr>
      <th>2014-11-05</th>
      <td>99.889618</td>
      <td>131.040939</td>
      <td>42.842709</td>
      <td>544.425293</td>
    </tr>
    <tr>
      <th>2014-11-06</th>
      <td>100.175293</td>
      <td>131.644287</td>
      <td>43.594646</td>
      <td>540.555908</td>
    </tr>
    <tr>
      <th>2014-11-07</th>
      <td>100.460983</td>
      <td>132.141617</td>
      <td>43.576752</td>
      <td>539.528748</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2019-10-28</th>
      <td>249.050003</td>
      <td>135.970001</td>
      <td>144.190002</td>
      <td>1290.000000</td>
    </tr>
    <tr>
      <th>2019-10-29</th>
      <td>243.289993</td>
      <td>133.820007</td>
      <td>142.830002</td>
      <td>1262.619995</td>
    </tr>
    <tr>
      <th>2019-10-30</th>
      <td>243.259995</td>
      <td>135.250000</td>
      <td>144.610001</td>
      <td>1261.290039</td>
    </tr>
    <tr>
      <th>2019-10-31</th>
      <td>248.759995</td>
      <td>133.729996</td>
      <td>143.369995</td>
      <td>1260.109985</td>
    </tr>
    <tr>
      <th>2019-11-01</th>
      <td>255.820007</td>
      <td>135.529999</td>
      <td>143.720001</td>
      <td>1273.739990</td>
    </tr>
  </tbody>
</table>
<p>1259 rows × 4 columns</p>
</div>




```python
volume = pd.DataFrame({ticker: data['Volume'] for ticker, data in all_data.items()})
volume
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
      <th>AAPL</th>
      <th>IBM</th>
      <th>MSFT</th>
      <th>GOOG</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2014-11-03</th>
      <td>5.228260e+07</td>
      <td>4688200.0</td>
      <td>2.313040e+07</td>
      <td>1382200.0</td>
    </tr>
    <tr>
      <th>2014-11-04</th>
      <td>4.157440e+07</td>
      <td>4246900.0</td>
      <td>2.153080e+07</td>
      <td>1244200.0</td>
    </tr>
    <tr>
      <th>2014-11-05</th>
      <td>3.743590e+07</td>
      <td>4104700.0</td>
      <td>2.244960e+07</td>
      <td>2032200.0</td>
    </tr>
    <tr>
      <th>2014-11-06</th>
      <td>3.496850e+07</td>
      <td>4067600.0</td>
      <td>3.303780e+07</td>
      <td>1333200.0</td>
    </tr>
    <tr>
      <th>2014-11-07</th>
      <td>3.369150e+07</td>
      <td>3494800.0</td>
      <td>2.800060e+07</td>
      <td>1633700.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2019-10-28</th>
      <td>2.414320e+07</td>
      <td>3225700.0</td>
      <td>3.528010e+07</td>
      <td>2613200.0</td>
    </tr>
    <tr>
      <th>2019-10-29</th>
      <td>3.566010e+07</td>
      <td>4159600.0</td>
      <td>2.051970e+07</td>
      <td>1869200.0</td>
    </tr>
    <tr>
      <th>2019-10-30</th>
      <td>3.095060e+07</td>
      <td>2252700.0</td>
      <td>1.847170e+07</td>
      <td>1407700.0</td>
    </tr>
    <tr>
      <th>2019-10-31</th>
      <td>3.476660e+09</td>
      <td>341090000.0</td>
      <td>2.459620e+09</td>
      <td>145470000.0</td>
    </tr>
    <tr>
      <th>2019-11-01</th>
      <td>3.773870e+07</td>
      <td>3088800.0</td>
      <td>3.311920e+07</td>
      <td>1669400.0</td>
    </tr>
  </tbody>
</table>
<p>1259 rows × 4 columns</p>
</div>



We can now compute the percent changes of the prices.


```python
returns = price.pct_change()
returns
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
      <th>AAPL</th>
      <th>IBM</th>
      <th>MSFT</th>
      <th>GOOG</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2014-11-03</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2014-11-04</th>
      <td>-0.007312</td>
      <td>-0.010404</td>
      <td>0.002741</td>
      <td>-0.001999</td>
    </tr>
    <tr>
      <th>2014-11-05</th>
      <td>0.002394</td>
      <td>-0.005103</td>
      <td>0.006096</td>
      <td>-0.014780</td>
    </tr>
    <tr>
      <th>2014-11-06</th>
      <td>0.002860</td>
      <td>0.004604</td>
      <td>0.017551</td>
      <td>-0.007107</td>
    </tr>
    <tr>
      <th>2014-11-07</th>
      <td>0.002852</td>
      <td>0.003778</td>
      <td>-0.000410</td>
      <td>-0.001900</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2019-10-28</th>
      <td>0.010017</td>
      <td>0.003913</td>
      <td>0.024586</td>
      <td>0.019658</td>
    </tr>
    <tr>
      <th>2019-10-29</th>
      <td>-0.023128</td>
      <td>-0.015812</td>
      <td>-0.009432</td>
      <td>-0.021225</td>
    </tr>
    <tr>
      <th>2019-10-30</th>
      <td>-0.000123</td>
      <td>0.010686</td>
      <td>0.012462</td>
      <td>-0.001053</td>
    </tr>
    <tr>
      <th>2019-10-31</th>
      <td>0.022610</td>
      <td>-0.011238</td>
      <td>-0.008575</td>
      <td>-0.000936</td>
    </tr>
    <tr>
      <th>2019-11-01</th>
      <td>0.028381</td>
      <td>0.013460</td>
      <td>0.002441</td>
      <td>0.010817</td>
    </tr>
  </tbody>
</table>
<p>1259 rows × 4 columns</p>
</div>



The `corr` and `cov` methods for **Series** computes the correlation and covariance of two Series, aligned by index.
The `corr` and `cov` methods for **DataFrame** computes the correlation and covariance matrices.


```python
returns.MSFT.corr(returns.IBM)
```




    0.4870311189559724




```python
returns.corr()
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
      <th>AAPL</th>
      <th>IBM</th>
      <th>MSFT</th>
      <th>GOOG</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AAPL</th>
      <td>1.000000</td>
      <td>0.405805</td>
      <td>0.570409</td>
      <td>0.521556</td>
    </tr>
    <tr>
      <th>IBM</th>
      <td>0.405805</td>
      <td>1.000000</td>
      <td>0.487031</td>
      <td>0.413659</td>
    </tr>
    <tr>
      <th>MSFT</th>
      <td>0.570409</td>
      <td>0.487031</td>
      <td>1.000000</td>
      <td>0.655724</td>
    </tr>
    <tr>
      <th>GOOG</th>
      <td>0.521556</td>
      <td>0.413659</td>
      <td>0.655724</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
returns.cov()
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
      <th>AAPL</th>
      <th>IBM</th>
      <th>MSFT</th>
      <th>GOOG</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AAPL</th>
      <td>0.000247</td>
      <td>0.000083</td>
      <td>0.000133</td>
      <td>0.000125</td>
    </tr>
    <tr>
      <th>IBM</th>
      <td>0.000083</td>
      <td>0.000170</td>
      <td>0.000094</td>
      <td>0.000082</td>
    </tr>
    <tr>
      <th>MSFT</th>
      <td>0.000133</td>
      <td>0.000094</td>
      <td>0.000219</td>
      <td>0.000147</td>
    </tr>
    <tr>
      <th>GOOG</th>
      <td>0.000125</td>
      <td>0.000082</td>
      <td>0.000147</td>
      <td>0.000231</td>
    </tr>
  </tbody>
</table>
</div>



DataFrame can still compute specific correlations using the `corrwith` method.


```python
returns.corrwith(returns.IBM)
```




    AAPL    0.405805
    IBM     1.000000
    MSFT    0.487031
    GOOG    0.413659
    dtype: float64




```python
returns.corrwith(volume)
```




    AAPL    0.016438
    IBM    -0.057802
    MSFT   -0.035033
    GOOG   -0.004697
    dtype: float64



### Unique values, value counts, and membership


```python
obj = pd.Series(['c', 'a', 'd', 'a', 'a', 'b', 'b', 'c', 'c'])
uniques = obj.unique()
uniques
```




    array(['c', 'a', 'd', 'b'], dtype=object)




```python
obj.value_counts()
```




    c    3
    a    3
    b    2
    d    1
    dtype: int64




```python
mask = obj.isin(['b', 'c'])
mask
```




    0     True
    1    False
    2    False
    3    False
    4    False
    5     True
    6     True
    7     True
    8     True
    dtype: bool




```python
to_match = pd.Series(['c', 'a', 'b', 'b', 'c', 'a'])
unique_vals = pd.Series(['c', 'b', 'a'])
pd.Index(unique_vals).get_indexer(to_match)
```




    array([0, 2, 1, 1, 0, 2])


