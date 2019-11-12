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




    <pandas.core.groupby.generic.SeriesGroupBy object at 0x12d2e21d0>




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




    <pandas.core.groupby.generic.SeriesGroupBy object at 0x12d2db190>




```python
df['data1'].groupby(df['key1'])
```




    <pandas.core.groupby.generic.SeriesGroupBy object at 0x12d2f02d0>



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

An aggregation method is a data transformation that turns an array into a scalar value.
When used on a GroupBy object, the transformation is applied to each group, separately.


```python
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
grouped = df.groupby('key1')
grouped['data1'].quantile()
```




    key1
    a    0.443863
    b    0.441356
    Name: data1, dtype: float64



It is common to define custom aggregation methods.
The can be applied to grouped data by passing them to the `agg()` method of a GroupBy object.


```python
def peak_to_peak(arr):
    return arr.max() - arr.min()

grouped.agg(peak_to_peak)
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
      <td>1.310230</td>
      <td>2.348175</td>
    </tr>
    <tr>
      <th>b</th>
      <td>0.639363</td>
      <td>0.518226</td>
    </tr>
  </tbody>
</table>
</div>



Other methods will also perform as expected, even though they are not sticktly aggregations.


```python
grouped.describe()
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

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="8" halign="left">data1</th>
      <th colspan="8" halign="left">data2</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>key1</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>3.0</td>
      <td>0.680727</td>
      <td>0.686479</td>
      <td>0.144044</td>
      <td>0.293953</td>
      <td>0.443863</td>
      <td>0.949068</td>
      <td>1.454274</td>
      <td>3.0</td>
      <td>0.324553</td>
      <td>1.174114</td>
      <td>-0.854096</td>
      <td>-0.260211</td>
      <td>0.333674</td>
      <td>0.913877</td>
      <td>1.494079</td>
    </tr>
    <tr>
      <th>b</th>
      <td>2.0</td>
      <td>0.441356</td>
      <td>0.452098</td>
      <td>0.121675</td>
      <td>0.281516</td>
      <td>0.441356</td>
      <td>0.601197</td>
      <td>0.761038</td>
      <td>2.0</td>
      <td>0.053955</td>
      <td>0.366441</td>
      <td>-0.205158</td>
      <td>-0.075602</td>
      <td>0.053955</td>
      <td>0.183511</td>
      <td>0.313068</td>
    </tr>
  </tbody>
</table>
</div>



### Column-wise and multiple function application

It is common to need to use a different aggregation method for each column.
This is demonstrated using an example.


```python
# Read in data on tipping at restaurants.
tips = pd.read_csv('assets/examples/tips.csv')

# Calculate the percent of the bill the tip covered.
tips['tip_pct'] = tips['tip'] / tips['total_bill']
tips.head()
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
      <th>total_bill</th>
      <th>tip</th>
      <th>smoker</th>
      <th>day</th>
      <th>time</th>
      <th>size</th>
      <th>tip_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>16.99</td>
      <td>1.01</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>2</td>
      <td>0.059447</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10.34</td>
      <td>1.66</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>3</td>
      <td>0.160542</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21.01</td>
      <td>3.50</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>3</td>
      <td>0.166587</td>
    </tr>
    <tr>
      <th>3</th>
      <td>23.68</td>
      <td>3.31</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>2</td>
      <td>0.139780</td>
    </tr>
    <tr>
      <th>4</th>
      <td>24.59</td>
      <td>3.61</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>4</td>
      <td>0.146808</td>
    </tr>
  </tbody>
</table>
</div>




```python
grouped = tips.groupby(['day', 'smoker'])
grouped_pct = grouped['tip_pct']

# The average tipping percentage per day, split into smokers and non-smokers.
grouped_pct.agg('mean')
```




    day   smoker
    Fri   No        0.151650
          Yes       0.174783
    Sat   No        0.158048
          Yes       0.147906
    Sun   No        0.160113
          Yes       0.187250
    Thur  No        0.160298
          Yes       0.163863
    Name: tip_pct, dtype: float64



The `agg` method can be passed a list of functions or function names to apply to the grouped data.


```python
grouped_pct.agg(['mean', 'std', peak_to_peak])
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
      <th>mean</th>
      <th>std</th>
      <th>peak_to_peak</th>
    </tr>
    <tr>
      <th>day</th>
      <th>smoker</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">Fri</th>
      <th>No</th>
      <td>0.151650</td>
      <td>0.028123</td>
      <td>0.067349</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>0.174783</td>
      <td>0.051293</td>
      <td>0.159925</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Sat</th>
      <th>No</th>
      <td>0.158048</td>
      <td>0.039767</td>
      <td>0.235193</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>0.147906</td>
      <td>0.061375</td>
      <td>0.290095</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Sun</th>
      <th>No</th>
      <td>0.160113</td>
      <td>0.042347</td>
      <td>0.193226</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>0.187250</td>
      <td>0.154134</td>
      <td>0.644685</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Thur</th>
      <th>No</th>
      <td>0.160298</td>
      <td>0.038774</td>
      <td>0.193350</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>0.163863</td>
      <td>0.039389</td>
      <td>0.151240</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Passing 2-tuples provides the name of the new column and the aggregation method.
grouped_pct.agg([('foo', 'mean'), ('bar', np.std)])
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
      <th>foo</th>
      <th>bar</th>
    </tr>
    <tr>
      <th>day</th>
      <th>smoker</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">Fri</th>
      <th>No</th>
      <td>0.151650</td>
      <td>0.028123</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>0.174783</td>
      <td>0.051293</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Sat</th>
      <th>No</th>
      <td>0.158048</td>
      <td>0.039767</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>0.147906</td>
      <td>0.061375</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Sun</th>
      <th>No</th>
      <td>0.160113</td>
      <td>0.042347</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>0.187250</td>
      <td>0.154134</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Thur</th>
      <th>No</th>
      <td>0.160298</td>
      <td>0.038774</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>0.163863</td>
      <td>0.039389</td>
    </tr>
  </tbody>
</table>
</div>



The above example was using a grouped Series.
With a DataFrame, a list of functions can be applied to all columns, or specific functions can be applied to specified columns.


```python
functions = ['count', 'mean', 'max']
result = grouped['tip_pct', 'total_bill'].agg(functions)
result
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

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="3" halign="left">tip_pct</th>
      <th colspan="3" halign="left">total_bill</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>max</th>
      <th>count</th>
      <th>mean</th>
      <th>max</th>
    </tr>
    <tr>
      <th>day</th>
      <th>smoker</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">Fri</th>
      <th>No</th>
      <td>4</td>
      <td>0.151650</td>
      <td>0.187735</td>
      <td>4</td>
      <td>18.420000</td>
      <td>22.75</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>15</td>
      <td>0.174783</td>
      <td>0.263480</td>
      <td>15</td>
      <td>16.813333</td>
      <td>40.17</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Sat</th>
      <th>No</th>
      <td>45</td>
      <td>0.158048</td>
      <td>0.291990</td>
      <td>45</td>
      <td>19.661778</td>
      <td>48.33</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>42</td>
      <td>0.147906</td>
      <td>0.325733</td>
      <td>42</td>
      <td>21.276667</td>
      <td>50.81</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Sun</th>
      <th>No</th>
      <td>57</td>
      <td>0.160113</td>
      <td>0.252672</td>
      <td>57</td>
      <td>20.506667</td>
      <td>48.17</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>19</td>
      <td>0.187250</td>
      <td>0.710345</td>
      <td>19</td>
      <td>24.120000</td>
      <td>45.35</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Thur</th>
      <th>No</th>
      <td>45</td>
      <td>0.160298</td>
      <td>0.266312</td>
      <td>45</td>
      <td>17.113111</td>
      <td>41.19</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>17</td>
      <td>0.163863</td>
      <td>0.241255</td>
      <td>17</td>
      <td>19.190588</td>
      <td>43.11</td>
    </tr>
  </tbody>
</table>
</div>




```python
# A mapping of column names to functions.
fxn_mapping = {
    'tip_pct': ['min', 'max', 'mean', 'std'],
    'size': 'sum'
}
grouped.agg(fxn_mapping)
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

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="4" halign="left">tip_pct</th>
      <th>size</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>min</th>
      <th>max</th>
      <th>mean</th>
      <th>std</th>
      <th>sum</th>
    </tr>
    <tr>
      <th>day</th>
      <th>smoker</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">Fri</th>
      <th>No</th>
      <td>0.120385</td>
      <td>0.187735</td>
      <td>0.151650</td>
      <td>0.028123</td>
      <td>9</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>0.103555</td>
      <td>0.263480</td>
      <td>0.174783</td>
      <td>0.051293</td>
      <td>31</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Sat</th>
      <th>No</th>
      <td>0.056797</td>
      <td>0.291990</td>
      <td>0.158048</td>
      <td>0.039767</td>
      <td>115</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>0.035638</td>
      <td>0.325733</td>
      <td>0.147906</td>
      <td>0.061375</td>
      <td>104</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Sun</th>
      <th>No</th>
      <td>0.059447</td>
      <td>0.252672</td>
      <td>0.160113</td>
      <td>0.042347</td>
      <td>167</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>0.065660</td>
      <td>0.710345</td>
      <td>0.187250</td>
      <td>0.154134</td>
      <td>49</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Thur</th>
      <th>No</th>
      <td>0.072961</td>
      <td>0.266312</td>
      <td>0.160298</td>
      <td>0.038774</td>
      <td>112</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>0.090014</td>
      <td>0.241255</td>
      <td>0.163863</td>
      <td>0.039389</td>
      <td>40</td>
    </tr>
  </tbody>
</table>
</div>



### Return aggregated data without row indexes

So far, the returned aggregations have the group key combinations for an index, often hierarchical.
This can be disabled by passing `as_index=False`.


```python
# with index
tips.groupby(['day', 'smoker']).mean()
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
      <th>total_bill</th>
      <th>tip</th>
      <th>size</th>
      <th>tip_pct</th>
    </tr>
    <tr>
      <th>day</th>
      <th>smoker</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">Fri</th>
      <th>No</th>
      <td>18.420000</td>
      <td>2.812500</td>
      <td>2.250000</td>
      <td>0.151650</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>16.813333</td>
      <td>2.714000</td>
      <td>2.066667</td>
      <td>0.174783</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Sat</th>
      <th>No</th>
      <td>19.661778</td>
      <td>3.102889</td>
      <td>2.555556</td>
      <td>0.158048</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>21.276667</td>
      <td>2.875476</td>
      <td>2.476190</td>
      <td>0.147906</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Sun</th>
      <th>No</th>
      <td>20.506667</td>
      <td>3.167895</td>
      <td>2.929825</td>
      <td>0.160113</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>24.120000</td>
      <td>3.516842</td>
      <td>2.578947</td>
      <td>0.187250</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Thur</th>
      <th>No</th>
      <td>17.113111</td>
      <td>2.673778</td>
      <td>2.488889</td>
      <td>0.160298</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>19.190588</td>
      <td>3.030000</td>
      <td>2.352941</td>
      <td>0.163863</td>
    </tr>
  </tbody>
</table>
</div>




```python
# without index
tips.groupby(['day', 'smoker'], as_index=False).mean()
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
      <th>day</th>
      <th>smoker</th>
      <th>total_bill</th>
      <th>tip</th>
      <th>size</th>
      <th>tip_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Fri</td>
      <td>No</td>
      <td>18.420000</td>
      <td>2.812500</td>
      <td>2.250000</td>
      <td>0.151650</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Fri</td>
      <td>Yes</td>
      <td>16.813333</td>
      <td>2.714000</td>
      <td>2.066667</td>
      <td>0.174783</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Sat</td>
      <td>No</td>
      <td>19.661778</td>
      <td>3.102889</td>
      <td>2.555556</td>
      <td>0.158048</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sat</td>
      <td>Yes</td>
      <td>21.276667</td>
      <td>2.875476</td>
      <td>2.476190</td>
      <td>0.147906</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Sun</td>
      <td>No</td>
      <td>20.506667</td>
      <td>3.167895</td>
      <td>2.929825</td>
      <td>0.160113</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Sun</td>
      <td>Yes</td>
      <td>24.120000</td>
      <td>3.516842</td>
      <td>2.578947</td>
      <td>0.187250</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Thur</td>
      <td>No</td>
      <td>17.113111</td>
      <td>2.673778</td>
      <td>2.488889</td>
      <td>0.160298</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Thur</td>
      <td>Yes</td>
      <td>19.190588</td>
      <td>3.030000</td>
      <td>2.352941</td>
      <td>0.163863</td>
    </tr>
  </tbody>
</table>
</div>




```python
# also without index
tips.groupby(['day', 'smoker']).mean().reset_index()
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
      <th>day</th>
      <th>smoker</th>
      <th>total_bill</th>
      <th>tip</th>
      <th>size</th>
      <th>tip_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Fri</td>
      <td>No</td>
      <td>18.420000</td>
      <td>2.812500</td>
      <td>2.250000</td>
      <td>0.151650</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Fri</td>
      <td>Yes</td>
      <td>16.813333</td>
      <td>2.714000</td>
      <td>2.066667</td>
      <td>0.174783</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Sat</td>
      <td>No</td>
      <td>19.661778</td>
      <td>3.102889</td>
      <td>2.555556</td>
      <td>0.158048</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sat</td>
      <td>Yes</td>
      <td>21.276667</td>
      <td>2.875476</td>
      <td>2.476190</td>
      <td>0.147906</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Sun</td>
      <td>No</td>
      <td>20.506667</td>
      <td>3.167895</td>
      <td>2.929825</td>
      <td>0.160113</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Sun</td>
      <td>Yes</td>
      <td>24.120000</td>
      <td>3.516842</td>
      <td>2.578947</td>
      <td>0.187250</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Thur</td>
      <td>No</td>
      <td>17.113111</td>
      <td>2.673778</td>
      <td>2.488889</td>
      <td>0.160298</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Thur</td>
      <td>Yes</td>
      <td>19.190588</td>
      <td>3.030000</td>
      <td>2.352941</td>
      <td>0.163863</td>
    </tr>
  </tbody>
</table>
</div>



## 10.3 Apply: general split-apply-combine

Apply splits the object being manipulated into pieces, invokes a provided function on each piece, and then concatenates the pieces together.
Here is an example using the tipping data.


```python
def top(df, n=5, column='tip_pct'):
    '''
    Return rows with the top `n` values in `column`.
    '''
    return df.sort_values(by=column)[-n:]

top(tips, n=6)
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
      <th>total_bill</th>
      <th>tip</th>
      <th>smoker</th>
      <th>day</th>
      <th>time</th>
      <th>size</th>
      <th>tip_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>109</th>
      <td>14.31</td>
      <td>4.00</td>
      <td>Yes</td>
      <td>Sat</td>
      <td>Dinner</td>
      <td>2</td>
      <td>0.279525</td>
    </tr>
    <tr>
      <th>183</th>
      <td>23.17</td>
      <td>6.50</td>
      <td>Yes</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>4</td>
      <td>0.280535</td>
    </tr>
    <tr>
      <th>232</th>
      <td>11.61</td>
      <td>3.39</td>
      <td>No</td>
      <td>Sat</td>
      <td>Dinner</td>
      <td>2</td>
      <td>0.291990</td>
    </tr>
    <tr>
      <th>67</th>
      <td>3.07</td>
      <td>1.00</td>
      <td>Yes</td>
      <td>Sat</td>
      <td>Dinner</td>
      <td>1</td>
      <td>0.325733</td>
    </tr>
    <tr>
      <th>178</th>
      <td>9.60</td>
      <td>4.00</td>
      <td>Yes</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>2</td>
      <td>0.416667</td>
    </tr>
    <tr>
      <th>172</th>
      <td>7.25</td>
      <td>5.15</td>
      <td>Yes</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>2</td>
      <td>0.710345</td>
    </tr>
  </tbody>
</table>
</div>




```python
tips.groupby('smoker').apply(top, n=3)
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
      <th>total_bill</th>
      <th>tip</th>
      <th>smoker</th>
      <th>day</th>
      <th>time</th>
      <th>size</th>
      <th>tip_pct</th>
    </tr>
    <tr>
      <th>smoker</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">No</th>
      <th>51</th>
      <td>10.29</td>
      <td>2.60</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>2</td>
      <td>0.252672</td>
    </tr>
    <tr>
      <th>149</th>
      <td>7.51</td>
      <td>2.00</td>
      <td>No</td>
      <td>Thur</td>
      <td>Lunch</td>
      <td>2</td>
      <td>0.266312</td>
    </tr>
    <tr>
      <th>232</th>
      <td>11.61</td>
      <td>3.39</td>
      <td>No</td>
      <td>Sat</td>
      <td>Dinner</td>
      <td>2</td>
      <td>0.291990</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">Yes</th>
      <th>67</th>
      <td>3.07</td>
      <td>1.00</td>
      <td>Yes</td>
      <td>Sat</td>
      <td>Dinner</td>
      <td>1</td>
      <td>0.325733</td>
    </tr>
    <tr>
      <th>178</th>
      <td>9.60</td>
      <td>4.00</td>
      <td>Yes</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>2</td>
      <td>0.416667</td>
    </tr>
    <tr>
      <th>172</th>
      <td>7.25</td>
      <td>5.15</td>
      <td>Yes</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>2</td>
      <td>0.710345</td>
    </tr>
  </tbody>
</table>
</div>



### Suppressing the group keys

To prevent the resulting DataFrame from having the grouping keys for row indexes, use `group_keys=False` in `groupby()`.


```python
tips.groupby('smoker', group_keys=False).apply(top, n=3)
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
      <th>total_bill</th>
      <th>tip</th>
      <th>smoker</th>
      <th>day</th>
      <th>time</th>
      <th>size</th>
      <th>tip_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>51</th>
      <td>10.29</td>
      <td>2.60</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>2</td>
      <td>0.252672</td>
    </tr>
    <tr>
      <th>149</th>
      <td>7.51</td>
      <td>2.00</td>
      <td>No</td>
      <td>Thur</td>
      <td>Lunch</td>
      <td>2</td>
      <td>0.266312</td>
    </tr>
    <tr>
      <th>232</th>
      <td>11.61</td>
      <td>3.39</td>
      <td>No</td>
      <td>Sat</td>
      <td>Dinner</td>
      <td>2</td>
      <td>0.291990</td>
    </tr>
    <tr>
      <th>67</th>
      <td>3.07</td>
      <td>1.00</td>
      <td>Yes</td>
      <td>Sat</td>
      <td>Dinner</td>
      <td>1</td>
      <td>0.325733</td>
    </tr>
    <tr>
      <th>178</th>
      <td>9.60</td>
      <td>4.00</td>
      <td>Yes</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>2</td>
      <td>0.416667</td>
    </tr>
    <tr>
      <th>172</th>
      <td>7.25</td>
      <td>5.15</td>
      <td>Yes</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>2</td>
      <td>0.710345</td>
    </tr>
  </tbody>
</table>
</div>



### Quantile and bucket analysis

pandas includes `cut()` and `qcut()` for dividing data into quantiles.
This often is useful in conjunction with `groupby()`.
The categorical object returned by `cut()` can be passed directly to `groupby()` to use to separate the data.


```python
frame = pd.DataFrame({'data1': np.random.randn(1000),
                      'data2': np.random.randn(1000)})
quartiles = pd.cut(frame.data1, 4)
quartiles.head()
```




    0    (-1.492, 0.0624]
    1    (-3.052, -1.492]
    2     (0.0624, 1.617]
    3    (-1.492, 0.0624]
    4    (-1.492, 0.0624]
    Name: data1, dtype: category
    Categories (4, interval[float64]): [(-3.052, -1.492] < (-1.492, 0.0624] < (0.0624, 1.617] < (1.617, 3.171]]




```python
def get_stats(group):
    return {
        'min': group.min(),
        'max': group.max(),
        'count': group.count(),
        'mean': group.mean()
    }

frame.data2.groupby(quartiles).apply(get_stats).unstack()
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
      <th>min</th>
      <th>max</th>
      <th>count</th>
      <th>mean</th>
    </tr>
    <tr>
      <th>data1</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>(-3.052, -1.492]</th>
      <td>-1.730276</td>
      <td>2.153120</td>
      <td>68.0</td>
      <td>0.097696</td>
    </tr>
    <tr>
      <th>(-1.492, 0.0624]</th>
      <td>-2.777359</td>
      <td>2.680571</td>
      <td>482.0</td>
      <td>0.056230</td>
    </tr>
    <tr>
      <th>(0.0624, 1.617]</th>
      <td>-3.116857</td>
      <td>2.929096</td>
      <td>398.0</td>
      <td>-0.046616</td>
    </tr>
    <tr>
      <th>(1.617, 3.171]</th>
      <td>-2.158069</td>
      <td>2.042072</td>
      <td>52.0</td>
      <td>-0.156563</td>
    </tr>
  </tbody>
</table>
</div>



Use `qcut()` to get equally-sized buckets.


```python
grouping = pd.qcut(frame.data1, 10, labels=False)
frame.data2.groupby(grouping).apply(get_stats).unstack()
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
      <th>min</th>
      <th>max</th>
      <th>count</th>
      <th>mean</th>
    </tr>
    <tr>
      <th>data1</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-2.534554</td>
      <td>2.642936</td>
      <td>100.0</td>
      <td>-0.013313</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.641703</td>
      <td>2.232016</td>
      <td>100.0</td>
      <td>0.059095</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-2.777359</td>
      <td>2.620574</td>
      <td>100.0</td>
      <td>0.193168</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.830029</td>
      <td>2.488442</td>
      <td>100.0</td>
      <td>-0.103425</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-2.123890</td>
      <td>2.526368</td>
      <td>100.0</td>
      <td>0.169418</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-2.121176</td>
      <td>2.680571</td>
      <td>100.0</td>
      <td>0.044411</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-3.116857</td>
      <td>2.929096</td>
      <td>100.0</td>
      <td>0.083797</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-2.994613</td>
      <td>2.540232</td>
      <td>100.0</td>
      <td>-0.305252</td>
    </tr>
    <tr>
      <th>8</th>
      <td>-1.980566</td>
      <td>2.190898</td>
      <td>100.0</td>
      <td>-0.058111</td>
    </tr>
    <tr>
      <th>9</th>
      <td>-2.802203</td>
      <td>2.198296</td>
      <td>100.0</td>
      <td>0.000731</td>
    </tr>
  </tbody>
</table>
</div>



## 10.4 Pivot tables and cross-tabulation

Pivot tables are made with pandas by using `groupby()` and some reshaping operations.
There is a `pivot_table()` method for DataFrames and a top-level `pd.pivot_table()` function.


```python
tips.pivot_table(index=['day', 'smoker'])
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
      <th>size</th>
      <th>tip</th>
      <th>tip_pct</th>
      <th>total_bill</th>
    </tr>
    <tr>
      <th>day</th>
      <th>smoker</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">Fri</th>
      <th>No</th>
      <td>2.250000</td>
      <td>2.812500</td>
      <td>0.151650</td>
      <td>18.420000</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>2.066667</td>
      <td>2.714000</td>
      <td>0.174783</td>
      <td>16.813333</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Sat</th>
      <th>No</th>
      <td>2.555556</td>
      <td>3.102889</td>
      <td>0.158048</td>
      <td>19.661778</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>2.476190</td>
      <td>2.875476</td>
      <td>0.147906</td>
      <td>21.276667</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Sun</th>
      <th>No</th>
      <td>2.929825</td>
      <td>3.167895</td>
      <td>0.160113</td>
      <td>20.506667</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>2.578947</td>
      <td>3.516842</td>
      <td>0.187250</td>
      <td>24.120000</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Thur</th>
      <th>No</th>
      <td>2.488889</td>
      <td>2.673778</td>
      <td>0.160298</td>
      <td>17.113111</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>2.352941</td>
      <td>3.030000</td>
      <td>0.163863</td>
      <td>19.190588</td>
    </tr>
  </tbody>
</table>
</div>



The following example only aggragates `'tip_pct'` and `'size'` and additionally groups by `'time'`.


```python
tips.pivot_table(['tip_pct', 'size'], index=['time', 'day'], columns='smoker')
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

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="2" halign="left">size</th>
      <th colspan="2" halign="left">tip_pct</th>
    </tr>
    <tr>
      <th></th>
      <th>smoker</th>
      <th>No</th>
      <th>Yes</th>
      <th>No</th>
      <th>Yes</th>
    </tr>
    <tr>
      <th>time</th>
      <th>day</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">Dinner</th>
      <th>Fri</th>
      <td>2.000000</td>
      <td>2.222222</td>
      <td>0.139622</td>
      <td>0.165347</td>
    </tr>
    <tr>
      <th>Sat</th>
      <td>2.555556</td>
      <td>2.476190</td>
      <td>0.158048</td>
      <td>0.147906</td>
    </tr>
    <tr>
      <th>Sun</th>
      <td>2.929825</td>
      <td>2.578947</td>
      <td>0.160113</td>
      <td>0.187250</td>
    </tr>
    <tr>
      <th>Thur</th>
      <td>2.000000</td>
      <td>NaN</td>
      <td>0.159744</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Lunch</th>
      <th>Fri</th>
      <td>3.000000</td>
      <td>1.833333</td>
      <td>0.187735</td>
      <td>0.188937</td>
    </tr>
    <tr>
      <th>Thur</th>
      <td>2.500000</td>
      <td>2.352941</td>
      <td>0.160311</td>
      <td>0.163863</td>
    </tr>
  </tbody>
</table>
</div>



Setting `margins=True` computes partial totals, adding an `'All'` row and column.


```python
tips.pivot_table(['tip_pct', 'size'],
                 index=['time', 'day'], 
                 columns='smoker', 
                 margins=True)
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

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="3" halign="left">size</th>
      <th colspan="3" halign="left">tip_pct</th>
    </tr>
    <tr>
      <th></th>
      <th>smoker</th>
      <th>No</th>
      <th>Yes</th>
      <th>All</th>
      <th>No</th>
      <th>Yes</th>
      <th>All</th>
    </tr>
    <tr>
      <th>time</th>
      <th>day</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">Dinner</th>
      <th>Fri</th>
      <td>2.000000</td>
      <td>2.222222</td>
      <td>2.166667</td>
      <td>0.139622</td>
      <td>0.165347</td>
      <td>0.158916</td>
    </tr>
    <tr>
      <th>Sat</th>
      <td>2.555556</td>
      <td>2.476190</td>
      <td>2.517241</td>
      <td>0.158048</td>
      <td>0.147906</td>
      <td>0.153152</td>
    </tr>
    <tr>
      <th>Sun</th>
      <td>2.929825</td>
      <td>2.578947</td>
      <td>2.842105</td>
      <td>0.160113</td>
      <td>0.187250</td>
      <td>0.166897</td>
    </tr>
    <tr>
      <th>Thur</th>
      <td>2.000000</td>
      <td>NaN</td>
      <td>2.000000</td>
      <td>0.159744</td>
      <td>NaN</td>
      <td>0.159744</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Lunch</th>
      <th>Fri</th>
      <td>3.000000</td>
      <td>1.833333</td>
      <td>2.000000</td>
      <td>0.187735</td>
      <td>0.188937</td>
      <td>0.188765</td>
    </tr>
    <tr>
      <th>Thur</th>
      <td>2.500000</td>
      <td>2.352941</td>
      <td>2.459016</td>
      <td>0.160311</td>
      <td>0.163863</td>
      <td>0.161301</td>
    </tr>
    <tr>
      <th>All</th>
      <th></th>
      <td>2.668874</td>
      <td>2.408602</td>
      <td>2.569672</td>
      <td>0.159328</td>
      <td>0.163196</td>
      <td>0.160803</td>
    </tr>
  </tbody>
</table>
</div>



The default aggregation behaviour is to take the mean.
This can be changed by passing a different function to `aggfunc`.


```python
tips.pivot_table('tip_pct', 
                 index=['time', 'smoker'], 
                 columns='day', 
                 aggfunc=len, 
                 margins=True)
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
      <th>day</th>
      <th>Fri</th>
      <th>Sat</th>
      <th>Sun</th>
      <th>Thur</th>
      <th>All</th>
    </tr>
    <tr>
      <th>time</th>
      <th>smoker</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">Dinner</th>
      <th>No</th>
      <td>3.0</td>
      <td>45.0</td>
      <td>57.0</td>
      <td>1.0</td>
      <td>106.0</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>9.0</td>
      <td>42.0</td>
      <td>19.0</td>
      <td>NaN</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Lunch</th>
      <th>No</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>45.0</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>6.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>17.0</td>
      <td>23.0</td>
    </tr>
    <tr>
      <th>All</th>
      <th></th>
      <td>19.0</td>
      <td>87.0</td>
      <td>76.0</td>
      <td>62.0</td>
      <td>244.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
tips.pivot_table('tip_pct',
                 index=['time', 'smoker'], 
                 columns='day', 
                 aggfunc=len, 
                 margins=True, 
                 fill_value=0)
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
      <th>day</th>
      <th>Fri</th>
      <th>Sat</th>
      <th>Sun</th>
      <th>Thur</th>
      <th>All</th>
    </tr>
    <tr>
      <th>time</th>
      <th>smoker</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">Dinner</th>
      <th>No</th>
      <td>3</td>
      <td>45</td>
      <td>57</td>
      <td>1</td>
      <td>106.0</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>9</td>
      <td>42</td>
      <td>19</td>
      <td>0</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Lunch</th>
      <th>No</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>44</td>
      <td>45.0</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>17</td>
      <td>23.0</td>
    </tr>
    <tr>
      <th>All</th>
      <th></th>
      <td>19</td>
      <td>87</td>
      <td>76</td>
      <td>62</td>
      <td>244.0</td>
    </tr>
  </tbody>
</table>
</div>



### Cross-tabulations (crosstab)

Cross-tabulation is a special case of a pivot table the computes group frequencies.


```python
data = pd.DataFrame({
    'Sample': list(range(1, 11)),
    'Nationality': ['USA', 'Japan', 'USA', 'Japan', 'Japan', 'Japan', 'USA', 'USA', 'Japan', 'USA'],
    'Handedness': ['Right-handed', 'Left-handed', 'Right-handed', 'Right-handed', 'Left-handed', 'Right-handed', 'Right-handed', 'Left-handed', 'Right-handed', 'Right-handed']
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
      <th>Sample</th>
      <th>Nationality</th>
      <th>Handedness</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>USA</td>
      <td>Right-handed</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Japan</td>
      <td>Left-handed</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>USA</td>
      <td>Right-handed</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Japan</td>
      <td>Right-handed</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Japan</td>
      <td>Left-handed</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>Japan</td>
      <td>Right-handed</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>USA</td>
      <td>Right-handed</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>USA</td>
      <td>Left-handed</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>Japan</td>
      <td>Right-handed</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>USA</td>
      <td>Right-handed</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.crosstab(data.Nationality, data.Handedness, margins=True)
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
      <th>Handedness</th>
      <th>Left-handed</th>
      <th>Right-handed</th>
      <th>All</th>
    </tr>
    <tr>
      <th>Nationality</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Japan</th>
      <td>2</td>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>USA</th>
      <td>1</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>All</th>
      <td>3</td>
      <td>7</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.crosstab([tips.time, tips.day], tips.smoker, margins=True)
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
      <th>smoker</th>
      <th>No</th>
      <th>Yes</th>
      <th>All</th>
    </tr>
    <tr>
      <th>time</th>
      <th>day</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">Dinner</th>
      <th>Fri</th>
      <td>3</td>
      <td>9</td>
      <td>12</td>
    </tr>
    <tr>
      <th>Sat</th>
      <td>45</td>
      <td>42</td>
      <td>87</td>
    </tr>
    <tr>
      <th>Sun</th>
      <td>57</td>
      <td>19</td>
      <td>76</td>
    </tr>
    <tr>
      <th>Thur</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Lunch</th>
      <th>Fri</th>
      <td>1</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>Thur</th>
      <td>44</td>
      <td>17</td>
      <td>61</td>
    </tr>
    <tr>
      <th>All</th>
      <th></th>
      <td>151</td>
      <td>93</td>
      <td>244</td>
    </tr>
  </tbody>
</table>
</div>


