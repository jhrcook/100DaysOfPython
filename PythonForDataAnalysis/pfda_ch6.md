# Chapter 6. Data Loading, Storage, and File Formats

The three main ways to access data is by reading in on-disk files, loading from databases, requesting via a web API.

## 6.1 Reading and writing data in text format

There are various way to read in tabular data as a DataFrame in pandas. 
The functions I am most likely to frequently use are `read_csv()`, `read_table()`, `read_excel()`, `read_pickle()`, `read_json()`, and `read_sql()`.

These functions are generally do a few similar processes on the input file:

1. Index the rows and columns.
2. Type inference and data conversion.
3. Datetime parsing.
4. Fixing unlean data such as footers, data with commas (especially numerics), etc.

These functions have many parameters to adjust to the quirks of the data; therefore, if you run into a problem, look through a documentation to see if any of the parameters are made for fixing it.

Here are some simple examples of reading data in with pandas.


```python
!cat assets/examples/ex1.csv
```

    a,b,c,d,message
    1,2,3,4,hello
    5,6,7,8,world
    9,10,11,12,foo


```python
import pandas as pd
import numpy as np

np.random.seed(0)

ex_dir = 'assets/examples/'
df = pd.read_csv(ex_dir + 'ex1.csv')
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
!cat assets/examples/ex2.csv
```

    1,2,3,4,hello
    5,6,7,8,world
    9,10,11,12,foo


```python
pd.read_csv(ex_dir + 'ex2.csv', header=None)
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
pd.read_csv(ex_dir + 'ex2.csv', names=['a', 'b', 'c', 'd', 'message'])
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
names = ['a', 'b', 'c', 'd', 'message']
pd.read_csv(ex_dir + 'ex2.csv', names=names, index_col='message')
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
    <tr>
      <th>message</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>hello</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>world</th>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>8</td>
    </tr>
    <tr>
      <th>foo</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>



The DataFrame can be read in with multiple hierarchical indices.


```python
!cat assets/examples/csv_mindex.csv
```

    key1,key2,value1,value2
    one,a,1,2
    one,b,3,4
    one,c,5,6
    one,d,7,8
    two,a,9,10
    two,b,11,12
    two,c,13,14
    two,d,15,16



```python
parsed = pd.read_csv(ex_dir + 'csv_mindex.csv', index_col=['key1', 'key2'])
parsed
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
      <th>value1</th>
      <th>value2</th>
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
      <th rowspan="4" valign="top">one</th>
      <th>a</th>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>b</th>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>c</th>
      <td>5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>d</th>
      <td>7</td>
      <td>8</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">two</th>
      <th>a</th>
      <td>9</td>
      <td>10</td>
    </tr>
    <tr>
      <th>b</th>
      <td>11</td>
      <td>12</td>
    </tr>
    <tr>
      <th>c</th>
      <td>13</td>
      <td>14</td>
    </tr>
    <tr>
      <th>d</th>
      <td>15</td>
      <td>16</td>
    </tr>
  </tbody>
</table>
</div>



Troublesome rows can be skipped explicitly.


```python
!cat assets/examples/ex4.csv
```

    # hey!
    a,b,c,d,message
    # just wanted to make things more difficult for you
    # who reads CSV files with computers, anyway?
    1,2,3,4,hello
    5,6,7,8,world
    9,10,11,12,foo


```python
pd.read_csv(ex_dir + 'ex4.csv')
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
      <th></th>
      <th></th>
      <th># hey!</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <td>message</td>
    </tr>
    <tr>
      <th># just wanted to make things more difficult for you</th>
      <th>NaN</th>
      <th>NaN</th>
      <th>NaN</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th># who reads CSV files with computers</th>
      <th>anyway?</th>
      <th>NaN</th>
      <th>NaN</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <td>hello</td>
    </tr>
    <tr>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <td>world</td>
    </tr>
    <tr>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.read_csv(ex_dir + 'ex4.csv', skiprows=[0, 2, 3])
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



The common missing data sentinels, such as `NA` and `NULL`, are automatically idenfied by pandas.


```python
!cat assets/examples/ex5.csv
```

    something,a,b,c,d,message
    one,1,2,3,4,NA
    two,5,6,,8,world
    three,9,10,11,12,foo


```python
result = pd.read_csv(ex_dir + 'ex5.csv')
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

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>something</th>
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
      <td>one</td>
      <td>1</td>
      <td>2</td>
      <td>3.0</td>
      <td>4</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>two</td>
      <td>5</td>
      <td>6</td>
      <td>NaN</td>
      <td>8</td>
      <td>world</td>
    </tr>
    <tr>
      <th>2</th>
      <td>three</td>
      <td>9</td>
      <td>10</td>
      <td>11.0</td>
      <td>12</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>



Some of the other more common arguments used when reading in data with pandas are `sep`/`delimiter`, `header`, `index_col`, `names`, `skiprows`, `parse_dates`, `nrows`, `chunksize` (for reading in files piecemeal), `skip_footer`.

### Reading test files in pieces

Before continuing with large DataFrames, set the print-out to be smaller.


```python
pd.options.display.max_rows = 10
```

We can then read in a large file and have it nicely printed out.


```python
result = pd.read_csv(ex_dir + 'ex6.csv')
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
      <th>key</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.467976</td>
      <td>-0.038649</td>
      <td>-0.295344</td>
      <td>-1.824726</td>
      <td>L</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.358893</td>
      <td>1.404453</td>
      <td>0.704965</td>
      <td>-0.200638</td>
      <td>B</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.501840</td>
      <td>0.659254</td>
      <td>-0.421691</td>
      <td>-0.057688</td>
      <td>G</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.204886</td>
      <td>1.074134</td>
      <td>1.388361</td>
      <td>-0.982404</td>
      <td>R</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.354628</td>
      <td>-0.133116</td>
      <td>0.283763</td>
      <td>-0.837063</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9995</th>
      <td>2.311896</td>
      <td>-0.417070</td>
      <td>-1.409599</td>
      <td>-0.515821</td>
      <td>L</td>
    </tr>
    <tr>
      <th>9996</th>
      <td>-0.479893</td>
      <td>-0.650419</td>
      <td>0.745152</td>
      <td>-0.646038</td>
      <td>E</td>
    </tr>
    <tr>
      <th>9997</th>
      <td>0.523331</td>
      <td>0.787112</td>
      <td>0.486066</td>
      <td>1.093156</td>
      <td>K</td>
    </tr>
    <tr>
      <th>9998</th>
      <td>-0.362559</td>
      <td>0.598894</td>
      <td>-1.843201</td>
      <td>0.887292</td>
      <td>G</td>
    </tr>
    <tr>
      <th>9999</th>
      <td>-0.096376</td>
      <td>-1.012999</td>
      <td>-0.657431</td>
      <td>-0.573315</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>10000 rows × 5 columns</p>
</div>



We could have also just read in the first few rows.


```python
pd.read_csv(ex_dir + 'ex6.csv', nrows=5)
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
      <th>key</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.467976</td>
      <td>-0.038649</td>
      <td>-0.295344</td>
      <td>-1.824726</td>
      <td>L</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.358893</td>
      <td>1.404453</td>
      <td>0.704965</td>
      <td>-0.200638</td>
      <td>B</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.501840</td>
      <td>0.659254</td>
      <td>-0.421691</td>
      <td>-0.057688</td>
      <td>G</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.204886</td>
      <td>1.074134</td>
      <td>1.388361</td>
      <td>-0.982404</td>
      <td>R</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.354628</td>
      <td>-0.133116</td>
      <td>0.283763</td>
      <td>-0.837063</td>
      <td>Q</td>
    </tr>
  </tbody>
</table>
</div>



Or the file could be read in in pieces.


```python
chunker = pd.read_csv(ex_dir + 'ex6.csv', chunksize=1000)
chunker
```




    <pandas.io.parsers.TextFileReader at 0x122787050>




```python
tot = pd.Series([])
for piece in chunker:
    tot = tot.add(piece['key'].value_counts(), fill_value=0)

tot = tot.sort_values(ascending=False)
tot.head()
```




    E    368.0
    X    364.0
    L    346.0
    O    343.0
    Q    340.0
    dtype: float64



### Writing data to text format


```python
data = pd.read_csv(ex_dir + 'ex5.csv')
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
      <th>something</th>
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
      <td>one</td>
      <td>1</td>
      <td>2</td>
      <td>3.0</td>
      <td>4</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>two</td>
      <td>5</td>
      <td>6</td>
      <td>NaN</td>
      <td>8</td>
      <td>world</td>
    </tr>
    <tr>
      <th>2</th>
      <td>three</td>
      <td>9</td>
      <td>10</td>
      <td>11.0</td>
      <td>12</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.to_csv(ex_dir + 'out.csv')
!cat assets/examples/out.csv
```

    ,something,a,b,c,d,message
    0,one,1,2,3.0,4,
    1,two,5,6,,8,world
    2,three,9,10,11.0,12,foo



```python
data.to_csv(ex_dir + 'out.csv', sep='|')
!cat assets/examples/out.csv
```

    |something|a|b|c|d|message
    0|one|1|2|3.0|4|
    1|two|5|6||8|world
    2|three|9|10|11.0|12|foo



```python
data.to_csv(ex_dir + 'out.csv', index=False, columns=['a', 'b', 'c'])
!cat assets/examples/out.csv
```

    a,b,c
    1,2,3.0
    5,6,
    9,10,11.0


### JSON data

JSON stnads for "JavaScript Object Notation" and is very nearly valid Python code.


```python
obj = """
    {"name": "Wes",
     "places_lived": ["United States", "Spain", "Germany"],
     "pet": null,
     "siblings": [{"name": "Scott", "age": 30, "pets": ["Zeus", "Zuko"]},
                  {"name": "Katie", "age": 38,
                   "pets": ["Sixes", "Stache", "Cisco"]}]
} """
obj
```




    '\n    {"name": "Wes",\n     "places_lived": ["United States", "Spain", "Germany"],\n     "pet": null,\n     "siblings": [{"name": "Scott", "age": 30, "pets": ["Zeus", "Zuko"]},\n                  {"name": "Katie", "age": 38,\n                   "pets": ["Sixes", "Stache", "Cisco"]}]\n} '



There are many JSON parsing libraries, but we will use 'json'.


```python
import json

result = json.loads(obj)
result
```




    {'name': 'Wes',
     'places_lived': ['United States', 'Spain', 'Germany'],
     'pet': None,
     'siblings': [{'name': 'Scott', 'age': 30, 'pets': ['Zeus', 'Zuko']},
      {'name': 'Katie', 'age': 38, 'pets': ['Sixes', 'Stache', 'Cisco']}]}




```python
asjson = json.dumps(result)
asjson
```




    '{"name": "Wes", "places_lived": ["United States", "Spain", "Germany"], "pet": null, "siblings": [{"name": "Scott", "age": 30, "pets": ["Zeus", "Zuko"]}, {"name": "Katie", "age": 38, "pets": ["Sixes", "Stache", "Cisco"]}]}'



There are many ways to go from JSON to DataFrame.
One way is to pass a list of dictionaries and select a subset of the data fields.


```python
siblings = pd.DataFrame(result['siblings'], columns=['names', 'age'])
siblings
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
      <th>names</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>30</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>38</td>
    </tr>
  </tbody>
</table>
</div>



There is also the `read_json()` function from pandas.


```python
!cat assets/examples/example.json
```

    [{"a": 1, "b": 2, "c": 3},
     {"a": 4, "b": 5, "c": 6},
     {"a": 7, "b": 8, "c": 9}]



```python
data = pd.read_json(ex_dir + 'example.json')
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>8</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.to_json()
```




    '{"a":{"0":1,"1":4,"2":7},"b":{"0":2,"1":5,"2":8},"c":{"0":3,"1":6,"2":9}}'



### XML and HTML: web scraping

For this section, we installed a few popular libariesfor reading and writing HTML and XML.

```bash
conda install lxml beautifulsoup4 html5lib
```

The pandas `read_html()` searches for and parses tabular data, often within `<table><\table>` tags.


```python
tables = pd.read_html(ex_dir + 'fdic_failed_bank_list.html')
len(tables)
failures = tables[0]
failures.head()
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
      <th>Bank Name</th>
      <th>City</th>
      <th>ST</th>
      <th>CERT</th>
      <th>Acquiring Institution</th>
      <th>Closing Date</th>
      <th>Updated Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Allied Bank</td>
      <td>Mulberry</td>
      <td>AR</td>
      <td>91</td>
      <td>Today's Bank</td>
      <td>September 23, 2016</td>
      <td>November 17, 2016</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The Woodbury Banking Company</td>
      <td>Woodbury</td>
      <td>GA</td>
      <td>11297</td>
      <td>United Bank</td>
      <td>August 19, 2016</td>
      <td>November 17, 2016</td>
    </tr>
    <tr>
      <th>2</th>
      <td>First CornerStone Bank</td>
      <td>King of Prussia</td>
      <td>PA</td>
      <td>35312</td>
      <td>First-Citizens Bank &amp; Trust Company</td>
      <td>May 6, 2016</td>
      <td>September 6, 2016</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Trust Company Bank</td>
      <td>Memphis</td>
      <td>TN</td>
      <td>9956</td>
      <td>The Bank of Fayette County</td>
      <td>April 29, 2016</td>
      <td>September 6, 2016</td>
    </tr>
    <tr>
      <th>4</th>
      <td>North Milwaukee State Bank</td>
      <td>Milwaukee</td>
      <td>WI</td>
      <td>20364</td>
      <td>First-Citizens Bank &amp; Trust Company</td>
      <td>March 11, 2016</td>
      <td>June 16, 2016</td>
    </tr>
  </tbody>
</table>
</div>



XML can be parsed using lxml or BeautifulSoup.
(The author includes an example of how one could do this; I just read it, but did not take notes.)

## 6.2 Binary Data Formats

One of the most common binary serialization protocols in Python is *pickle* serialization.
All pandas data structures have a `to_pickle()` method.


```python
frame = pd.read_csv(ex_dir + 'ex1.csv')
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
frame.to_pickle(ex_dir + 'frame_pickle')
```


```python
pd.read_pickle(ex_dir + 'frame_pickle')
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



There are also the HDF5, MessagePack, bcolz, and Feather serialization formats.

### HDF5

File format intended for storing large scientific data arrays.
It has interfaces for many other languages (including MATLAB).


```python
frame = pd.DataFrame({'a': np.random.randn(100)})
store = pd.HDFStore(ex_dir + 'mydata.h5')
store['obj1'] = frame
store['obj1_col'] = frame['a']
store
```




    <class 'pandas.io.pytables.HDFStore'>
    File path: assets/examples/mydata.h5




```python
store['obj1']
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.764052</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.400157</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.978738</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.240893</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.867558</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>0.706573</td>
    </tr>
    <tr>
      <th>96</th>
      <td>0.010500</td>
    </tr>
    <tr>
      <th>97</th>
      <td>1.785870</td>
    </tr>
    <tr>
      <th>98</th>
      <td>0.126912</td>
    </tr>
    <tr>
      <th>99</th>
      <td>0.401989</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 1 columns</p>
</div>



`HSF5Store()` supports two schemas, `'fixed'` and `'table'`.
The latter is slower, but supports queries.


```python
store.put('obj2', frame, format='table')
store.select('obj2', where=['index >= 10 and index <= 15'])
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>0.144044</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1.454274</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.761038</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.121675</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.443863</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.333674</td>
    </tr>
  </tbody>
</table>
</div>




```python
store.close()
```

The `read_hdf()` function provides easy access to a HDF5 file.


```python
frame.to_hdf(ex_dir + 'mydata.h5', 'obj3', format='table')
pd.read_hdf(ex_dir + 'mydata.h5', 'obj3', where=['index < 5'])
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.764052</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.400157</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.978738</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.240893</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.867558</td>
    </tr>
  </tbody>
</table>
</div>



### Reading Microsoft Excel files

pandas supports reading tabular data stored in Excel (≥2003).
Internally, pandas uses the *xlrd* and *openpyxl* libraries to read XLS and XLSX files.

If there are multiple sheets in an Excel file to be used, it is likely faster to use `ExcelFile()`.


```python
xlsx = pd.ExcelFile(ex_dir + 'ex1.xlsx')
xlsx.sheet_names
```




    ['Sheet1']




```python
xlsx.parse(sheet_name='Sheet1')
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
      <th>Unnamed: 0</th>
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
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>hello</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>8</td>
      <td>world</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
      <td>12</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>



Otherwise, `read_excel()` just returns a DataFrame, immediately.


```python
frame = pd.read_excel(ex_dir + 'ex1.xlsx')
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
      <th>Unnamed: 0</th>
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
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>hello</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>8</td>
      <td>world</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
      <td>12</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>



Excel files can be written using an `ExcelWriter` object and `to_excel()` or by passing the file name to `to_excel()`.


```python
writer = pd.ExcelWriter(ex_dir  + 'ex2.xlsx')
frame.to_excel(writer, 'Sheet1')
writer.save()
```


```python
frame.to_excel(ex_dir + 'ex2.xlsx')
```

### Interacting with Web APIs

One of the most popular libraries for interacting with web APIs is [*requests*](https://pypi.org/project/requests/2.7.0/) (Real Python put together a [tutorial](https://realpython.com/python-requests/), too).

This example requests the last 30 issues from the pandas GitHub page.


```python
import requests

url = 'https://api.github.com/repos/pandas-dev/pandas/issues'
resp = requests.get(url)
resp
```




    <Response [200]>



The response can then be parsedto JSON.


```python
data = resp.json()  # convert to JSON
data[0]  # show first issue
```




    {'url': 'https://api.github.com/repos/pandas-dev/pandas/issues/29411',
     'repository_url': 'https://api.github.com/repos/pandas-dev/pandas',
     'labels_url': 'https://api.github.com/repos/pandas-dev/pandas/issues/29411/labels{/name}',
     'comments_url': 'https://api.github.com/repos/pandas-dev/pandas/issues/29411/comments',
     'events_url': 'https://api.github.com/repos/pandas-dev/pandas/issues/29411/events',
     'html_url': 'https://github.com/pandas-dev/pandas/issues/29411',
     'id': 517663592,
     'node_id': 'MDU6SXNzdWU1MTc2NjM1OTI=',
     'number': 29411,
     'title': 'Memory leak in Dataframe.memory_usage',
     'user': {'login': 'hyfjjjj',
      'id': 7194638,
      'node_id': 'MDQ6VXNlcjcxOTQ2Mzg=',
      'avatar_url': 'https://avatars2.githubusercontent.com/u/7194638?v=4',
      'gravatar_id': '',
      'url': 'https://api.github.com/users/hyfjjjj',
      'html_url': 'https://github.com/hyfjjjj',
      'followers_url': 'https://api.github.com/users/hyfjjjj/followers',
      'following_url': 'https://api.github.com/users/hyfjjjj/following{/other_user}',
      'gists_url': 'https://api.github.com/users/hyfjjjj/gists{/gist_id}',
      'starred_url': 'https://api.github.com/users/hyfjjjj/starred{/owner}{/repo}',
      'subscriptions_url': 'https://api.github.com/users/hyfjjjj/subscriptions',
      'organizations_url': 'https://api.github.com/users/hyfjjjj/orgs',
      'repos_url': 'https://api.github.com/users/hyfjjjj/repos',
      'events_url': 'https://api.github.com/users/hyfjjjj/events{/privacy}',
      'received_events_url': 'https://api.github.com/users/hyfjjjj/received_events',
      'type': 'User',
      'site_admin': False},
     'labels': [],
     'state': 'open',
     'locked': False,
     'assignee': None,
     'assignees': [],
     'milestone': None,
     'comments': 0,
     'created_at': '2019-11-05T09:52:21Z',
     'updated_at': '2019-11-05T09:52:21Z',
     'closed_at': None,
     'author_association': 'NONE',
     'body': '#### Code Sample, a copy-pastable example if possible\r\n\r\n```python\r\nimport numpy as np\r\nimport pandas as pd\r\nimport gc\r\nimport os\r\nimport psutil\r\n\r\ndef get_process_memory():\r\n  return round(psutil.Process(os.getpid()).memory_info().rss / float(2 ** 20), 2)\r\n\r\ntest_dict = {}\r\nfor i in range(0, 50):\r\n  test_dict[i] = np.empty(10)\r\n\r\ndfs = []\r\nfor i in range(0, 1000):\r\n  df = pd.DataFrame(test_dict)\r\n  dfs.append(df)\r\n\r\ngc.collect()\r\n# before\r\nprint(\'memory usage (before "memory_usage"):\\t{} MB\'.format(get_process_memory()))\r\n\r\nfor df in dfs:\r\n  df.memory_usage(index=True, deep=True)\r\n\r\ngc.collect()\r\n# after\r\nprint(\'memory usage (after "memory_usage"):\\t{} MB\'.format(get_process_memory()))\r\n\r\n```\r\n#### Problem description\r\n\r\nDataframe\'s memory_usage function has memory leak. Memory usage after executing \'memory_usage\' function  should be the same as before.\r\n\r\n<img width="399" alt="截屏2019-11-05下午5 44 25" src="https://user-images.githubusercontent.com/7194638/68196715-f390ce00-fff3-11e9-939a-e84d850673e8.png">\r\n\r\n#### Expected Output\r\n\r\nNone\r\n\r\n#### Output of ``pd.show_versions()``\r\n\r\n<details>\r\n\r\nINSTALLED VERSIONS\r\n------------------\r\ncommit: None\r\npython: 2.7.16.final.0\r\npython-bits: 64\r\nOS: Darwin\r\nOS-release: 19.0.0\r\nmachine: x86_64\r\nprocessor: i386\r\nbyteorder: little\r\nLC_ALL: None\r\nLANG: zh_CN.UTF-8\r\nLOCALE: None.None\r\n\r\npandas: 0.24.2\r\npytest: None\r\npip: 19.3.1\r\nsetuptools: 19.6.1\r\nCython: 0.29.13\r\nnumpy: 1.16.5\r\nscipy: None\r\npyarrow: None\r\nxarray: None\r\nIPython: None\r\nsphinx: None\r\npatsy: None\r\ndateutil: 2.8.1\r\npytz: 2019.3\r\nblosc: None\r\nbottleneck: None\r\ntables: None\r\nnumexpr: None\r\nfeather: None\r\nmatplotlib: None\r\nopenpyxl: None\r\nxlrd: None\r\nxlwt: None\r\nxlsxwriter: None\r\nlxml.etree: None\r\nbs4: None\r\nhtml5lib: None\r\nsqlalchemy: None\r\npymysql: None\r\npsycopg2: None\r\njinja2: None\r\ns3fs: None\r\nfastparquet: None\r\npandas_gbq: None\r\npandas_datareader: None\r\ngcsfs: None\r\n\r\n</details>\r\n'}



Each element in `data` is a dictionary containing a single GitHub issue.
This can be turned into a DataFrame.


```python
issues = pd.DataFrame(data, columns=['number', 'title', 'labels', 'state'])
issues
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
      <th>number</th>
      <th>title</th>
      <th>labels</th>
      <th>state</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>29411</td>
      <td>Memory leak in Dataframe.memory_usage</td>
      <td>[]</td>
      <td>open</td>
    </tr>
    <tr>
      <th>1</th>
      <td>29410</td>
      <td>Fixed SS03 errors</td>
      <td>[]</td>
      <td>open</td>
    </tr>
    <tr>
      <th>2</th>
      <td>29409</td>
      <td>Formatting issues with column width truncation...</td>
      <td>[{'id': 13101118, 'node_id': 'MDU6TGFiZWwxMzEw...</td>
      <td>open</td>
    </tr>
    <tr>
      <th>3</th>
      <td>29408</td>
      <td>DataFrame.equals incorrect `See Also` section ...</td>
      <td>[{'id': 134699, 'node_id': 'MDU6TGFiZWwxMzQ2OT...</td>
      <td>open</td>
    </tr>
    <tr>
      <th>4</th>
      <td>29406</td>
      <td>CLN: assorted cleanups</td>
      <td>[]</td>
      <td>open</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>25</th>
      <td>29364</td>
      <td>26302 add typing to assert star equal funcs</td>
      <td>[{'id': 1280988427, 'node_id': 'MDU6TGFiZWwxMj...</td>
      <td>open</td>
    </tr>
    <tr>
      <th>26</th>
      <td>29361</td>
      <td>TYPING: scalar type that matches lib.is_scalar</td>
      <td>[{'id': 1280988427, 'node_id': 'MDU6TGFiZWwxMj...</td>
      <td>open</td>
    </tr>
    <tr>
      <th>27</th>
      <td>29357</td>
      <td>ensure consistent structure for groupby on ind...</td>
      <td>[{'id': 233160, 'node_id': 'MDU6TGFiZWwyMzMxNj...</td>
      <td>open</td>
    </tr>
    <tr>
      <th>28</th>
      <td>29356</td>
      <td>TST: new test for subset of a MultiIndex dtype</td>
      <td>[{'id': 127685, 'node_id': 'MDU6TGFiZWwxMjc2OD...</td>
      <td>open</td>
    </tr>
    <tr>
      <th>29</th>
      <td>29355</td>
      <td>TST: Test type issue fix in empty groupby from...</td>
      <td>[{'id': 78527356, 'node_id': 'MDU6TGFiZWw3ODUy...</td>
      <td>open</td>
    </tr>
  </tbody>
</table>
<p>30 rows × 4 columns</p>
</div>


