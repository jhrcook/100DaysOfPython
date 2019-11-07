# Chapter 8. Data wrangling: join, combine, and reshape

## 8.1 Hierarchical indexing

This feature of pandas allows one Series or DataFrame to have multiple levels of indexing.


```python
import pandas as pd
import numpy as np

np.random.seed(0)

data = pd.Series(np.random.randn(9),
                 index=[['a', 'a', 'a', 'b', 'b', 'c', 'c', 'd', 'd'],
                 [1,2,3,1,3,1,2,2,3]])
data
```




    a  1    1.764052
       2    0.400157
       3    0.978738
    b  1    2.240893
       3    1.867558
    c  1   -0.977278
       2    0.950088
    d  2   -0.151357
       3   -0.103219
    dtype: float64




```python
data.index
```




    MultiIndex([('a', 1),
                ('a', 2),
                ('a', 3),
                ('b', 1),
                ('b', 3),
                ('c', 1),
                ('c', 2),
                ('d', 2),
                ('d', 3)],
               )




```python
data['b']
```




    1    2.240893
    3    1.867558
    dtype: float64




```python
data['b':'c']
```




    b  1    2.240893
       3    1.867558
    c  1   -0.977278
       2    0.950088
    dtype: float64




```python
data.loc[['b', 'd']]
```




    b  1    2.240893
       3    1.867558
    d  2   -0.151357
       3   -0.103219
    dtype: float64




```python
data.loc[:, 2]
```




    a    0.400157
    c    0.950088
    d   -0.151357
    dtype: float64



This example data can easily be converted into a DataFrame.


```python
data.unstack()
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
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1.764052</td>
      <td>0.400157</td>
      <td>0.978738</td>
    </tr>
    <tr>
      <th>b</th>
      <td>2.240893</td>
      <td>NaN</td>
      <td>1.867558</td>
    </tr>
    <tr>
      <th>c</th>
      <td>-0.977278</td>
      <td>0.950088</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>d</th>
      <td>NaN</td>
      <td>-0.151357</td>
      <td>-0.103219</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.unstack().stack()
```




    a  1    1.764052
       2    0.400157
       3    0.978738
    b  1    2.240893
       3    1.867558
    c  1   -0.977278
       2    0.950088
    d  2   -0.151357
       3   -0.103219
    dtype: float64



Either axis of a DataFrame can have multilevel indexing.


```python
frame = pd.DataFrame(np.arange(12).reshape((4, 3)),
                     index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
                     columns=[['Ohio', 'Ohio', 'Colorado'],
                     ['Green', 'Red', 'Green']])
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

    .dataframe thead tr th {
        text-align: left;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="2" halign="left">Ohio</th>
      <th>Colorado</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>Green</th>
      <th>Red</th>
      <th>Green</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">a</th>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">b</th>
      <th>1</th>
      <td>6</td>
      <td>7</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame.index.names = ['key1', 'key2']
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
      <th colspan="2" halign="left">Ohio</th>
      <th>Colorado</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>Green</th>
      <th>Red</th>
      <th>Green</th>
    </tr>
    <tr>
      <th>key1</th>
      <th>key2</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">a</th>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">b</th>
      <th>1</th>
      <td>6</td>
      <td>7</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame.columns.names = ['state', 'color']
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
      <th>state</th>
      <th colspan="2" halign="left">Ohio</th>
      <th>Colorado</th>
    </tr>
    <tr>
      <th></th>
      <th>color</th>
      <th>Green</th>
      <th>Red</th>
      <th>Green</th>
    </tr>
    <tr>
      <th>key1</th>
      <th>key2</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">a</th>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">b</th>
      <th>1</th>
      <td>6</td>
      <td>7</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame['Ohio']
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
      <th>color</th>
      <th>Green</th>
      <th>Red</th>
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
      <th>1</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">b</th>
      <th>1</th>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>



### Reordering and sorting levels

The `swaplevel()` method takes two level numbers or names and return a new object with those levels swapped.


```python
frame.swaplevel('key1', 'key2')
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
      <th>state</th>
      <th colspan="2" halign="left">Ohio</th>
      <th>Colorado</th>
    </tr>
    <tr>
      <th></th>
      <th>color</th>
      <th>Green</th>
      <th>Red</th>
      <th>Green</th>
    </tr>
    <tr>
      <th>key2</th>
      <th>key1</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <th>a</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <th>a</th>
      <td>3</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <th>b</th>
      <td>6</td>
      <td>7</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <th>b</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>



The `sort_index()` method sorts the data using only the values of a single level.


```python
frame.sort_index(level=1)
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
      <th>state</th>
      <th colspan="2" halign="left">Ohio</th>
      <th>Colorado</th>
    </tr>
    <tr>
      <th></th>
      <th>color</th>
      <th>Green</th>
      <th>Red</th>
      <th>Green</th>
    </tr>
    <tr>
      <th>key1</th>
      <th>key2</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>b</th>
      <th>1</th>
      <td>6</td>
      <td>7</td>
      <td>8</td>
    </tr>
    <tr>
      <th>a</th>
      <th>2</th>
      <td>3</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>b</th>
      <th>2</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>



It is common to use these in tandem to re-order a DataFrame after swapping levels.


```python
frame.swaplevel(0, 1).sort_index(level=0)
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
      <th>state</th>
      <th colspan="2" halign="left">Ohio</th>
      <th>Colorado</th>
    </tr>
    <tr>
      <th></th>
      <th>color</th>
      <th>Green</th>
      <th>Red</th>
      <th>Green</th>
    </tr>
    <tr>
      <th>key2</th>
      <th>key1</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">1</th>
      <th>a</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>b</th>
      <td>6</td>
      <td>7</td>
      <td>8</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">2</th>
      <th>a</th>
      <td>3</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>b</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>



### Summary statistics by level


```python
frame.sum(level=1)
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
      <th>state</th>
      <th colspan="2" halign="left">Ohio</th>
      <th>Colorado</th>
    </tr>
    <tr>
      <th>color</th>
      <th>Green</th>
      <th>Red</th>
      <th>Green</th>
    </tr>
    <tr>
      <th>key2</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>6</td>
      <td>8</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12</td>
      <td>14</td>
      <td>16</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame.sum(level='color', axis=1)
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
      <th>color</th>
      <th>Green</th>
      <th>Red</th>
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
      <th>1</th>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>4</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">b</th>
      <th>1</th>
      <td>14</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>




```python
### Indexing with a DataFrame's columns

It is common to want to use a column for a row index or, vice versa, turn a row index into a column.
```


      File "<ipython-input-42-3a8e72a591d8>", line 3
        It is common to want to use a column for a row index or, vice versa, turn a row index into a column.
                      ^
    SyntaxError: invalid syntax




```python
frame = pd.DataFrame({
    'a': range(7), 'b': range(7, 0, -1),
    'c': ['one', 'one', 'one', 'two', 'two', 'two', 'two'],
    'd':[0,1,2,0,1,2,3]
})
frame
```


```python
frame2 = frame.set_index(['c', 'd'])
frame2
```


```python
frame.set_index(['c', 'd'], drop=False)
```


```python
frame2.reset_index()
```

## 8.2 Combining and merging datasets


```python

```
