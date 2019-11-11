# Chapter 10. Data aggregation and group operations

## 10.1 Groupby mechanics

The grouping key can be provided in different forms:

1. a list of values the same length oas the axis being grouped
2. a column name
3. a dictionary or Series that corresponds the values on the axis being grouped and the group names
4. a function to be invoked on the axis index or individual index labels

The following are some examples using these methods.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(0)
```


```python
df = pd.DataFrame({
    'key1' : ['a', 'a', 'b', 'b', 'a'],
    'key2' : ['one', 'two', 'one', 'two', 'one'],
    'data1' : np.random.randn(5), 
    'data2' : np.random.randn(5)
})
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
      <th>key1</th>
      <th>key2</th>
      <th>data1</th>
      <th>data2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>one</td>
      <td>1.764052</td>
      <td>-0.977278</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a</td>
      <td>two</td>
      <td>0.400157</td>
      <td>0.950088</td>
    </tr>
    <tr>
      <th>2</th>
      <td>b</td>
      <td>one</td>
      <td>0.978738</td>
      <td>-0.151357</td>
    </tr>
    <tr>
      <th>3</th>
      <td>b</td>
      <td>two</td>
      <td>2.240893</td>
      <td>-0.103219</td>
    </tr>
    <tr>
      <th>4</th>
      <td>a</td>
      <td>one</td>
      <td>1.867558</td>
      <td>0.410599</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = pd.DataFrame({
    'key1' : ['a', 'a', 'b', 'b', 'a'],
    'key2' : ['one', 'two', 'one', 'two', 'one'],
    'data1' : np.random.randn(5), 
    'data2' : np.random.randn(5)
})
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
      <th>key1</th>
      <th>key2</th>
      <th>data1</th>
      <th>data2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>one</td>
      <td>0.144044</td>
      <td>0.333674</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a</td>
      <td>two</td>
      <td>1.454274</td>
      <td>1.494079</td>
    </tr>
    <tr>
      <th>2</th>
      <td>b</td>
      <td>one</td>
      <td>0.761038</td>
      <td>-0.205158</td>
    </tr>
    <tr>
      <th>3</th>
      <td>b</td>
      <td>two</td>
      <td>0.121675</td>
      <td>0.313068</td>
    </tr>
    <tr>
      <th>4</th>
      <td>a</td>
      <td>one</td>
      <td>0.443863</td>
      <td>-0.854096</td>
    </tr>
  </tbody>
</table>
</div>




```python
# `groupby()` creates a new `GroupBy` object.
grouped = df['data1'].groupby(df['key1'])
grouped
```




    <pandas.core.groupby.generic.SeriesGroupBy object at 0x1136f1e90>




```python
# Mean of the data in the 'data1' column, grouped by 'key1'.
grouped.mean()
```




    key1
    a    0.680727
    b    0.441356
    Name: data1, dtype: float64




```python
# Group by two columns.
means = df['data1'].groupby([df['key1'], df['key2']]).mean()
means
```




    key1  key2
    a     one     0.293953
          two     1.454274
    b     one     0.761038
          two     0.121675
    Name: data1, dtype: float64




```python
means.unstack()
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
      <th>key2</th>
      <th>one</th>
      <th>two</th>
    </tr>
    <tr>
      <th>key1</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>0.293953</td>
      <td>1.454274</td>
    </tr>
    <tr>
      <th>b</th>
      <td>0.761038</td>
      <td>0.121675</td>
    </tr>
  </tbody>
</table>
</div>



If the grouping information is a column of the same DataFrame, then only the grouping column name is required.


```python
df.groupby('key1').mean()
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
      <th>data2</th>
    </tr>
    <tr>
      <th>key1</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>0.680727</td>
      <td>0.324553</td>
    </tr>
    <tr>
      <th>b</th>
      <td>0.441356</td>
      <td>0.053955</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.groupby(['key1', 'key2']).mean()
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
      <th></th>
      <th>data1</th>
      <th>data2</th>
    </tr>
    <tr>
      <th>key1</th>
      <th>key2</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">a</th>
      <th>one</th>
      <td>0.293953</td>
      <td>-0.260211</td>
    </tr>
    <tr>
      <th>two</th>
      <td>1.454274</td>
      <td>1.494079</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">b</th>
      <th>one</th>
      <td>0.761038</td>
      <td>-0.205158</td>
    </tr>
    <tr>
      <th>two</th>
      <td>0.121675</td>
      <td>0.313068</td>
    </tr>
  </tbody>
</table>
</div>



A frequently useful method on a grouped DataFrame is `size()`.


```python
df.groupby(['key1', 'key2']).size()
```




    key1  key2
    a     one     2
          two     1
    b     one     1
          two     1
    dtype: int64



### Iterating over groups

The GroupBy object created by `groupby()` supports iteration over a sequence of 2-tuples containing the group name and the data.


```python
for name, group in df.groupby('key1'):
    print(f'group name: {name}')
    print(group)
    print('')

```

    group name: a
      key1 key2     data1     data2
    0    a  one  0.144044  0.333674
    1    a  two  1.454274  1.494079
    4    a  one  0.443863 -0.854096
    
    group name: b
      key1 key2     data1     data2
    2    b  one  0.761038 -0.205158
    3    b  two  0.121675  0.313068
    



```python
for name, group in df.groupby(['key1', 'key2']):
    print(f'group name: {name[0]}-{name[1]}')
    print(group)
    print('')

```

    group name: a-one
      key1 key2     data1     data2
    0    a  one  0.144044  0.333674
    4    a  one  0.443863 -0.854096
    
    group name: a-two
      key1 key2     data1     data2
    1    a  two  1.454274  1.494079
    
    group name: b-one
      key1 key2     data1     data2
    2    b  one  0.761038 -0.205158
    
    group name: b-two
      key1 key2     data1     data2
    3    b  two  0.121675  0.313068
    


By default, `groupby()` groups on `axis=0`, though the columns could also be grouped.


```python
df.dtypes
```




    key1      object
    key2      object
    data1    float64
    data2    float64
    dtype: object




```python
grouped = df.groupby(df.dtypes, axis=1)
for dtype, group in grouped:
    print(f'data type: {dtype}')
    print(group)
    print('')

```

    data type: float64
          data1     data2
    0  0.144044  0.333674
    1  1.454274  1.494079
    2  0.761038 -0.205158
    3  0.121675  0.313068
    4  0.443863 -0.854096
    
    data type: object
      key1 key2
    0    a  one
    1    a  two
    2    b  one
    3    b  two
    4    a  one
    


### Selecting a column or subset of columns

A GroupBy object can still be indexed by column name.
The next two statements are equivalent.


```python
df.groupby('key1')['data1']
```




    <pandas.core.groupby.generic.SeriesGroupBy object at 0x113875750>




```python
df['data1'].groupby(df['key1'])
```




    <pandas.core.groupby.generic.SeriesGroupBy object at 0x11387b5d0>



### Grouping with dictionaries and Series




```python
people = pd.DataFrame(np.random.randn(5, 5),
                      columns=list('abcde'),
                      index=['Joe', 'Steve', 'Wex', 'Jim', 'Travis'])
people
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
      <th>Joe</th>
      <td>-2.552990</td>
      <td>0.653619</td>
      <td>0.864436</td>
      <td>-0.742165</td>
      <td>2.269755</td>
    </tr>
    <tr>
      <th>Steve</th>
      <td>-1.454366</td>
      <td>0.045759</td>
      <td>-0.187184</td>
      <td>1.532779</td>
      <td>1.469359</td>
    </tr>
    <tr>
      <th>Wex</th>
      <td>0.154947</td>
      <td>0.378163</td>
      <td>-0.887786</td>
      <td>-1.980796</td>
      <td>-0.347912</td>
    </tr>
    <tr>
      <th>Jim</th>
      <td>0.156349</td>
      <td>1.230291</td>
      <td>1.202380</td>
      <td>-0.387327</td>
      <td>-0.302303</td>
    </tr>
    <tr>
      <th>Travis</th>
      <td>-1.048553</td>
      <td>-1.420018</td>
      <td>-1.706270</td>
      <td>1.950775</td>
      <td>-0.509652</td>
    </tr>
  </tbody>
</table>
</div>




```python
people.iloc[2, [1, 2]] = np.nan
people
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
      <th>Joe</th>
      <td>-2.552990</td>
      <td>0.653619</td>
      <td>0.864436</td>
      <td>-0.742165</td>
      <td>2.269755</td>
    </tr>
    <tr>
      <th>Steve</th>
      <td>-1.454366</td>
      <td>0.045759</td>
      <td>-0.187184</td>
      <td>1.532779</td>
      <td>1.469359</td>
    </tr>
    <tr>
      <th>Wex</th>
      <td>0.154947</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-1.980796</td>
      <td>-0.347912</td>
    </tr>
    <tr>
      <th>Jim</th>
      <td>0.156349</td>
      <td>1.230291</td>
      <td>1.202380</td>
      <td>-0.387327</td>
      <td>-0.302303</td>
    </tr>
    <tr>
      <th>Travis</th>
      <td>-1.048553</td>
      <td>-1.420018</td>
      <td>-1.706270</td>
      <td>1.950775</td>
      <td>-0.509652</td>
    </tr>
  </tbody>
</table>
</div>



If I have a group correspondence for the columns and want to sum together the columns by these groups, I can just pass the dictionary for grouping.


```python
mapping = {
    'a': 'red', 'b': 'red', 'c': 'blue',
    'd': 'blue', 'e': 'red', 'f' : 'orange'
}
```


```python
by_column = people.groupby(mapping, axis=1)
by_column.sum()
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
      <th>blue</th>
      <th>red</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Joe</th>
      <td>0.122271</td>
      <td>0.370383</td>
    </tr>
    <tr>
      <th>Steve</th>
      <td>1.345595</td>
      <td>0.060752</td>
    </tr>
    <tr>
      <th>Wex</th>
      <td>-1.980796</td>
      <td>-0.192965</td>
    </tr>
    <tr>
      <th>Jim</th>
      <td>0.815053</td>
      <td>1.084337</td>
    </tr>
    <tr>
      <th>Travis</th>
      <td>0.244505</td>
      <td>-2.978223</td>
    </tr>
  </tbody>
</table>
</div>



### Grouping with functions

A function can be used to create the mappings.
Each group key will be passed once, and the return value defines the groups.

Here is an example of grouping by the length of the first names.


```python
people.groupby(len).sum()
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
      <th>3</th>
      <td>-2.241693</td>
      <td>1.883909</td>
      <td>2.066816</td>
      <td>-3.110288</td>
      <td>1.619540</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-1.454366</td>
      <td>0.045759</td>
      <td>-0.187184</td>
      <td>1.532779</td>
      <td>1.469359</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-1.048553</td>
      <td>-1.420018</td>
      <td>-1.706270</td>
      <td>1.950775</td>
      <td>-0.509652</td>
    </tr>
  </tbody>
</table>
</div>



It is possible to use both a function and an array or dictionary for grouping at the same time.


```python
key_list = ['one', 'one', 'one', 'two', 'two']
people.groupby([len, key_list]).min()
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
      <th rowspan="2" valign="top">3</th>
      <th>one</th>
      <td>-2.552990</td>
      <td>0.653619</td>
      <td>0.864436</td>
      <td>-1.980796</td>
      <td>-0.347912</td>
    </tr>
    <tr>
      <th>two</th>
      <td>0.156349</td>
      <td>1.230291</td>
      <td>1.202380</td>
      <td>-0.387327</td>
      <td>-0.302303</td>
    </tr>
    <tr>
      <th>5</th>
      <th>one</th>
      <td>-1.454366</td>
      <td>0.045759</td>
      <td>-0.187184</td>
      <td>1.532779</td>
      <td>1.469359</td>
    </tr>
    <tr>
      <th>6</th>
      <th>two</th>
      <td>-1.048553</td>
      <td>-1.420018</td>
      <td>-1.706270</td>
      <td>1.950775</td>
      <td>-0.509652</td>
    </tr>
  </tbody>
</table>
</div>



### Grouping by index levels

For hierarchically indexed data structures, the levels of the axis can be used for grouping.


```python
columns = pd.MultiIndex.from_arrays([['US', 'US', 'US', 'JP', 'JP'],
                                     [1, 3, 5, 1, 3]],
                                    names=['cty', 'tenor'])
hier_df = pd.DataFrame(np.random.randn(4, 5), columns=columns)
hier_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th>cty</th>
      <th colspan="3" halign="left">US</th>
      <th colspan="2" halign="left">JP</th>
    </tr>
    <tr>
      <th>tenor</th>
      <th>1</th>
      <th>3</th>
      <th>5</th>
      <th>1</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.438074</td>
      <td>-1.252795</td>
      <td>0.777490</td>
      <td>-1.613898</td>
      <td>-0.212740</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.895467</td>
      <td>0.386902</td>
      <td>-0.510805</td>
      <td>-1.180632</td>
      <td>-0.028182</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.428332</td>
      <td>0.066517</td>
      <td>0.302472</td>
      <td>-0.634322</td>
      <td>-0.362741</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.672460</td>
      <td>-0.359553</td>
      <td>-0.813146</td>
      <td>-1.726283</td>
      <td>0.177426</td>
    </tr>
  </tbody>
</table>
</div>




```python
hier_df.groupby(level='cty', axis=1).count()
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
      <th>cty</th>
      <th>JP</th>
      <th>US</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



## 10.2 Data Aggregation


```python

```
