# Chapter 12. Advanced pandas

## 12.1 Categorical data

This section introduces the pandas `Categorical` type.
It can often be more performance and memory efficient than the string equivalent.

### Background and motivation


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(0)
```


```python
%matplotlib inline
```


```python
values = pd.Series(['apple', 'orange', 'apple', 'apple'] * 2)
values
```




    0     apple
    1    orange
    2     apple
    3     apple
    4     apple
    5    orange
    6     apple
    7     apple
    dtype: object




```python
pd.unique(values)
```




    array(['apple', 'orange'], dtype=object)




```python
pd.value_counts(values)
```




    apple     6
    orange    2
    dtype: int64



A good solution for sotring categorical data is to use a dimension table.


```python
values = pd.Series([0, 1, 0, 0] * 2)
dim = pd.Series(['apple', 'orange'])
values
```




    0    0
    1    1
    2    0
    3    0
    4    0
    5    1
    6    0
    7    0
    dtype: int64




```python
dim[values]
```




    0     apple
    1    orange
    0     apple
    0     apple
    0     apple
    1    orange
    0     apple
    0     apple
    dtype: object




```python
dim.take(values)
```




    0     apple
    1    orange
    0     apple
    0     apple
    0     apple
    1    orange
    0     apple
    0     apple
    dtype: object



### Categorical type in pandas

The `Categorical` type uses an integer-based encoding.


```python
fruits = ['apple', 'orange', 'apple', 'apple'] * 2
N = len(fruits)
df = pd.DataFrame({
        'fruit': fruits,
        'basket_id': np.arange(N),
        'count': np.random.randint(3, 15, size=N),
        'weight': np.random.uniform(0, 4, size=N)
    }, columns = ['basket_id', 'fruit', 'count', 'weight'])
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
      <th>basket_id</th>
      <th>fruit</th>
      <th>count</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>apple</td>
      <td>8</td>
      <td>2.583576</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>orange</td>
      <td>3</td>
      <td>1.750349</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>apple</td>
      <td>6</td>
      <td>3.567092</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>apple</td>
      <td>14</td>
      <td>3.854651</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>apple</td>
      <td>6</td>
      <td>1.533766</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>orange</td>
      <td>10</td>
      <td>3.166900</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>apple</td>
      <td>12</td>
      <td>2.115580</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>apple</td>
      <td>6</td>
      <td>2.272178</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Turn the fruit data into a Categorical
fruit_cat = df['fruit'].astype('category')
fruit_cat
```




    0     apple
    1    orange
    2     apple
    3     apple
    4     apple
    5    orange
    6     apple
    7     apple
    Name: fruit, dtype: category
    Categories (2, object): [apple, orange]




```python
c = fruit_cat.values
type(c)
```




    pandas.core.arrays.categorical.Categorical



This Categorical object has `categories` and `codes` attributes.


```python
c.categories
```




    Index(['apple', 'orange'], dtype='object')




```python
c.codes
```




    array([0, 1, 0, 0, 0, 1, 0, 0], dtype=int8)




```python
# Convert the DataFrame column into a Categorical
df['fruit'] = df['fruit'].astype('category')
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
      <th>basket_id</th>
      <th>fruit</th>
      <th>count</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>apple</td>
      <td>8</td>
      <td>2.583576</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>orange</td>
      <td>3</td>
      <td>1.750349</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>apple</td>
      <td>6</td>
      <td>3.567092</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>apple</td>
      <td>14</td>
      <td>3.854651</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>apple</td>
      <td>6</td>
      <td>1.533766</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>orange</td>
      <td>10</td>
      <td>3.166900</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>apple</td>
      <td>12</td>
      <td>2.115580</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>apple</td>
      <td>6</td>
      <td>2.272178</td>
    </tr>
  </tbody>
</table>
</div>



A Categorical object can be created from the codes and categories using the `from_codes()` method.


```python
categories = ['foo', 'bar', 'baz']
codes = [0, 1, 2, 0, 0, 1]
my_cats_2 = pd.Categorical.from_codes(codes, categories)
my_cats_2
```




    [foo, bar, baz, foo, foo, bar]
    Categories (3, object): [foo, bar, baz]



By default, no order is assumed for the data, though it can be by setting `ordered=True`.
It can also be removed with the `as_unordered()` method.


```python
ordered_cat = pd.Categorical.from_codes(codes, categories, ordered=True)
ordered_cat
```




    [foo, bar, baz, foo, foo, bar]
    Categories (3, object): [foo < bar < baz]




```python
ordered_cat.as_unordered()
```




    [foo, bar, baz, foo, foo, bar]
    Categories (3, object): [foo, bar, baz]




```python
my_cats_2.as_ordered(inplace=True)
```


```python
my_cats_2
```




    [foo, bar, baz, foo, foo, bar]
    Categories (3, object): [foo < bar < baz]




```python
my_cats_2.ordered
```




    True



### Computations with categoricals

Most uses of Categorical will behave as if the data were still an unencoded structure (such as an array of strings).
Some pandas operations, such as `groupby()`, and other functions take advantage of the coded nature of Categorical for performance enhancements.

the `qcut()` and `cut()` pandas functions return Categoricals.


```python
draws = np.random.randn(1000)
draws[:5]
```




    array([ 0.48431215,  0.57914048, -0.18158257,  1.41020463, -0.37447169])




```python
bins = pd.qcut(draws, 4)
bins
```




    [(-0.0301, 0.599], (-0.0301, 0.599], (-0.741, -0.0301], (0.599, 3.102], (-0.741, -0.0301], ..., (-0.0301, 0.599], (0.599, 3.102], (-0.741, -0.0301], (-0.741, -0.0301], (-0.0301, 0.599]]
    Length: 1000
    Categories (4, interval[float64]): [(-3.516, -0.741] < (-0.741, -0.0301] < (-0.0301, 0.599] < (0.599, 3.102]]




```python
# The same quartiles but with more helpful names.
bins = pd.qcut(draws, 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
bins
```




    [Q3, Q3, Q2, Q4, Q2, ..., Q3, Q4, Q2, Q2, Q3]
    Length: 1000
    Categories (4, object): [Q1 < Q2 < Q3 < Q4]




```python
bins = pd.Series(bins, name='quartile')
results = (
    pd.Series(draws)
        .groupby(bins)
        .agg(['count', 'min', 'max'])
        .reset_index()
)

results
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
      <th>quartile</th>
      <th>count</th>
      <th>min</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Q1</td>
      <td>250</td>
      <td>-3.514768</td>
      <td>-0.742269</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Q2</td>
      <td>250</td>
      <td>-0.740747</td>
      <td>-0.031245</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Q3</td>
      <td>250</td>
      <td>-0.028946</td>
      <td>0.596688</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Q4</td>
      <td>250</td>
      <td>0.604160</td>
      <td>3.101531</td>
    </tr>
  </tbody>
</table>
</div>




```python
results.quartile
```




    0    Q1
    1    Q2
    2    Q3
    3    Q4
    Name: quartile, dtype: category
    Categories (4, object): [Q1 < Q2 < Q3 < Q4]



Using a Categorical in a DataFrame will provide improvements in speed of computations and resource-consumption by the DataFrame.


```python
N = 1000000

draws = pd.Series(np.random.randn(N))

labels = pd.Series(['foo', 'bar', 'baz', 'qux'] * (N // 4))

categories = labels.astype('category')
```


```python
labels.memory_usage()
```




    8000128




```python
categories.memory_usage()
```




    1000320



### Categorical methods

There are a few extra conviencne methods provided for Categorical objects.
They are accessed via the `cat` attribute of a Categorical.


```python
s = pd.Series(['a', 'b', 'c', 'd'] * 2)
cat_s = s.astype('category')
cat_s
```




    0    a
    1    b
    2    c
    3    d
    4    a
    5    b
    6    c
    7    d
    dtype: category
    Categories (4, object): [a, b, c, d]




```python
cat_s.cat.codes
```




    0    0
    1    1
    2    2
    3    3
    4    0
    5    1
    6    2
    7    3
    dtype: int8




```python
cat_s.cat.categories
```




    Index(['a', 'b', 'c', 'd'], dtype='object')



The categories can be changed, even extended beyond the actual values used.


```python
actual_categories = ['a', 'b', 'c', 'd', 'e']
cat_s2 = cat_s.cat.set_categories(actual_categories)
cat_s2
```




    0    a
    1    b
    2    c
    3    d
    4    a
    5    b
    6    c
    7    d
    dtype: category
    Categories (5, object): [a, b, c, d, e]




```python
cat_s.value_counts()
```




    d    2
    c    2
    b    2
    a    2
    dtype: int64




```python
cat_s2.value_counts()
```




    d    2
    c    2
    b    2
    a    2
    e    0
    dtype: int64



Unused categories can be removed with the `remove_unused_categories()` method.


```python
cat_s3 = cat_s[cat_s.isin(['a', 'b'])]
cat_s3
```




    0    a
    1    b
    4    a
    5    b
    dtype: category
    Categories (4, object): [a, b, c, d]




```python
cat_s3.cat.remove_unused_categories()
```




    0    a
    1    b
    4    a
    5    b
    dtype: category
    Categories (2, object): [a, b]



Once final example use-case for Categorical is to create dummy variables for modeling.


```python
cat_s = pd.Series(['a', 'b', 'c', 'd'] * 2, dtype='category')
pd.get_dummies(cat_s)
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
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## 12.2 Advanced groupby use

### Group transformations and "unwrapped" groupbys

As discussed earlier, the `apply()` method in grouped operations perform a transformation on the group, one at a time.
There is a similar method called `transform()` that has the following constraints on the function it can use:

- it can produce a scalar value
- it can produce an object of the same shape as the input
- it must *not* mutate the input


```python
df = pd.DataFrame({'key': ['a', 'b', 'c'] * 4,
                   'value': np.arange(12.0)})
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
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>c</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>a</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>b</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>c</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>a</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>b</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>c</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>a</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>b</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>c</td>
      <td>11.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
g = df.groupby('key').value
g.mean()
```




    key
    a    4.5
    b    5.5
    c    6.5
    Name: value, dtype: float64



The following transformation returns a Series of the same length as the input with the means of each group.


```python
g.transform(lambda x: x.mean())
```




    0     4.5
    1     5.5
    2     6.5
    3     4.5
    4     5.5
    5     6.5
    6     4.5
    7     5.5
    8     6.5
    9     4.5
    10    5.5
    11    6.5
    Name: value, dtype: float64




```python
# We can call `mean` by name because it is a built-in aggregation method.
g.transform('mean')
```




    0     4.5
    1     5.5
    2     6.5
    3     4.5
    4     5.5
    5     6.5
    6     4.5
    7     5.5
    8     6.5
    9     4.5
    10    5.5
    11    6.5
    Name: value, dtype: float64



## 12.3 Techniques for method chaining

The `assign()` method is useful for adding columns in method chains.
It does *not* modify in-place, but instead returns a new DataFrame.
The following two statements are equivalent:

```python
df2 = df.copy()
df2['k'] = v

df2 = df.assign(k=v)
```

Another useful tool for functional programming in Python is the *callable*.
Essentially, the entire DataFrame can be passed as a parameter to the supplied function; here is an example.


```python
pd.read_csv("assets/examples/ex1.csv")
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
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>hello</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>8</td>
      <td>world</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
      <td>12</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>




```python
(pd.read_csv("assets/examples/ex1.csv")
    [lambda x: x.b < 7])
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
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>hello</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>8</td>
      <td>world</td>
    </tr>
  </tbody>
</table>
</div>



### The pipe method

The `pipe()` method is useful for chaining functions that can pass arguments to each other sequentially.

```python
a = function1(df, arg1=v1)
b = function2(a, arg3=v3)
c = function3(b, arg4=v4)
```

```python
c = (df.pipe(function1, arg1=v1)
    .pipe(function2, arg3=v3)
    .pipe(function3, arg4=v4))
```
