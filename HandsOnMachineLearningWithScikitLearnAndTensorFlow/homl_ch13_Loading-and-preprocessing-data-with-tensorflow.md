# Loading and Preprocessing Data with TensorFlow

The *Data API* availabe from TensorFlow makes handling large data sets that don't fit into RAM much easier. In general, a dataset object is created and told where to find files. It then manages the rest of the implementation details including multithreading, queueing, batching, and prefetching.


```python
import numpy as np
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf 
from tensorflow import keras
from pathlib import Path

%matplotlib inline
np.random.seed(0)
plt.style.use('seaborn-whitegrid')
assets_dir = Path('assets', 'ch13')
```

## The Data API

The Data API is centered around the concept of a *dataset*, a sequence of data items.
For simplicity, the first example below shows a dataset held in memory created using `from_tensor_slices()`.
It is an iterable.


```python
X = tf.range(10)
dataset = tf.data.Dataset.from_tensor_slices(X)
dataset
```




    <TensorSliceDataset shapes: (), types: tf.int32>




```python
for t in dataset:
    print(t)
```

    tf.Tensor(0, shape=(), dtype=int32)
    tf.Tensor(1, shape=(), dtype=int32)
    tf.Tensor(2, shape=(), dtype=int32)
    tf.Tensor(3, shape=(), dtype=int32)
    tf.Tensor(4, shape=(), dtype=int32)
    tf.Tensor(5, shape=(), dtype=int32)
    tf.Tensor(6, shape=(), dtype=int32)
    tf.Tensor(7, shape=(), dtype=int32)
    tf.Tensor(8, shape=(), dtype=int32)
    tf.Tensor(9, shape=(), dtype=int32)


### Chaining transformations

The dataset object has many transformation methods that return new datasets, making it easy to chain the methods.


```python
dataset = dataset.repeat(3).batch(7)
for t in dataset:
    print(t)
```

    tf.Tensor([0 1 2 3 4 5 6], shape=(7,), dtype=int32)
    tf.Tensor([7 8 9 0 1 2 3], shape=(7,), dtype=int32)
    tf.Tensor([4 5 6 7 8 9 0], shape=(7,), dtype=int32)
    tf.Tensor([1 2 3 4 5 6 7], shape=(7,), dtype=int32)
    tf.Tensor([8 9], shape=(2,), dtype=int32)


Alternatively, `map()` can be used to apply custom transformations.


```python
dataset = dataset.map(lambda x: x * 2)
```

The `apply()` method applies a function to the dataset as a whole, not each item individually.
The following example removes the batches created above.


```python
dataset = dataset.apply(tf.data.experimental.unbatch())
```

The `filter()` method makes it each to apply a filter to each data point.


```python
dataset = dataset.filter(lambda x: x < 10)
```

The `take()` method can be used to look at just a few data points.


```python
for t in dataset.take(3):
    print(t)
```

    tf.Tensor(0, shape=(), dtype=int32)
    tf.Tensor(2, shape=(), dtype=int32)
    tf.Tensor(4, shape=(), dtype=int32)


### Shuffling the data

The shuffling method uses a buffer in memory to prepare data points before they are used and then randomly pulls one to be used when requested.
The size of the buffer must be declared so that all of the RAM is not consumed, but it also a bottleneck on how well the data will be shuffled.


```python
dataset = tf.data.Dataset.range(10).repeat(3)
dataset = dataset.shuffle(buffer_size=5, seed=0).batch(7)
for t in dataset:
    print(t)
```

    tf.Tensor([2 1 5 0 7 6 0], shape=(7,), dtype=int64)
    tf.Tensor([4 3 9 8 2 3 4], shape=(7,), dtype=int64)
    tf.Tensor([8 6 0 7 2 3 9], shape=(7,), dtype=int64)
    tf.Tensor([5 1 4 1 5 9 6], shape=(7,), dtype=int64)
    tf.Tensor([7 8], shape=(2,), dtype=int64)


One solution to the limits of `shuffle()` is to pre-shuffle the data.
For example, the `shuf` command in Linux shuffles the lines of a file.
Still, we will want to shuffle the file on each epoch.
To shuffle the data further, it is common the split the data into multiple files and then read in data from them randomly simultaneously, interleaving their records.
This method, paired with `shuffle()`, is usually a great way to shuffle the data and is easily accomplished with the Data API.

For this example, I split the housing data into 10 files.


```python
from sklearn.datasets import fetch_california_housing

cal_housing = fetch_california_housing()
df = pd.DataFrame(cal_housing.data)
df.columns = cal_housing.feature_names
df['MedianHousingValue'] = cal_housing.target
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
      <th>MedInc</th>
      <th>HouseAge</th>
      <th>AveRooms</th>
      <th>AveBedrms</th>
      <th>Population</th>
      <th>AveOccup</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>MedianHousingValue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8.3252</td>
      <td>41.0</td>
      <td>6.984127</td>
      <td>1.023810</td>
      <td>322.0</td>
      <td>2.555556</td>
      <td>37.88</td>
      <td>-122.23</td>
      <td>4.526</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8.3014</td>
      <td>21.0</td>
      <td>6.238137</td>
      <td>0.971880</td>
      <td>2401.0</td>
      <td>2.109842</td>
      <td>37.86</td>
      <td>-122.22</td>
      <td>3.585</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.2574</td>
      <td>52.0</td>
      <td>8.288136</td>
      <td>1.073446</td>
      <td>496.0</td>
      <td>2.802260</td>
      <td>37.85</td>
      <td>-122.24</td>
      <td>3.521</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5.6431</td>
      <td>52.0</td>
      <td>5.817352</td>
      <td>1.073059</td>
      <td>558.0</td>
      <td>2.547945</td>
      <td>37.85</td>
      <td>-122.25</td>
      <td>3.413</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.8462</td>
      <td>52.0</td>
      <td>6.281853</td>
      <td>1.081081</td>
      <td>565.0</td>
      <td>2.181467</td>
      <td>37.85</td>
      <td>-122.25</td>
      <td>3.422</td>
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
    </tr>
    <tr>
      <th>20635</th>
      <td>1.5603</td>
      <td>25.0</td>
      <td>5.045455</td>
      <td>1.133333</td>
      <td>845.0</td>
      <td>2.560606</td>
      <td>39.48</td>
      <td>-121.09</td>
      <td>0.781</td>
    </tr>
    <tr>
      <th>20636</th>
      <td>2.5568</td>
      <td>18.0</td>
      <td>6.114035</td>
      <td>1.315789</td>
      <td>356.0</td>
      <td>3.122807</td>
      <td>39.49</td>
      <td>-121.21</td>
      <td>0.771</td>
    </tr>
    <tr>
      <th>20637</th>
      <td>1.7000</td>
      <td>17.0</td>
      <td>5.205543</td>
      <td>1.120092</td>
      <td>1007.0</td>
      <td>2.325635</td>
      <td>39.43</td>
      <td>-121.22</td>
      <td>0.923</td>
    </tr>
    <tr>
      <th>20638</th>
      <td>1.8672</td>
      <td>18.0</td>
      <td>5.329513</td>
      <td>1.171920</td>
      <td>741.0</td>
      <td>2.123209</td>
      <td>39.43</td>
      <td>-121.32</td>
      <td>0.847</td>
    </tr>
    <tr>
      <th>20639</th>
      <td>2.3886</td>
      <td>16.0</td>
      <td>5.254717</td>
      <td>1.162264</td>
      <td>1387.0</td>
      <td>2.616981</td>
      <td>39.37</td>
      <td>-121.24</td>
      <td>0.894</td>
    </tr>
  </tbody>
</table>
<p>20640 rows × 9 columns</p>
</div>




```python
cal_housing_files = []
for i, split_df in enumerate(np.split(df, 10)):
    file_path = assets_dir.joinpath('ca_housing_' + str(i) + '.csv')
    cal_housing_files.append(file_path)
    split_df.to_csv(file_path, index=False)
```


```python
cal_housing_files
```




    [PosixPath('assets/ch13/ca_housing_0.csv'),
     PosixPath('assets/ch13/ca_housing_1.csv'),
     PosixPath('assets/ch13/ca_housing_2.csv'),
     PosixPath('assets/ch13/ca_housing_3.csv'),
     PosixPath('assets/ch13/ca_housing_4.csv'),
     PosixPath('assets/ch13/ca_housing_5.csv'),
     PosixPath('assets/ch13/ca_housing_6.csv'),
     PosixPath('assets/ch13/ca_housing_7.csv'),
     PosixPath('assets/ch13/ca_housing_8.csv'),
     PosixPath('assets/ch13/ca_housing_9.csv')]



Now we can create a dataset object and point to all three of the files.
Alternatively, we could pass a pattern such as `'assets/ch13/ca_housing_*.csv'`.


```python
cal_housing_strs = [x.as_posix() for x in cal_housing_files]
filepath_dataset = tf.data.Dataset.list_files(cal_housing_strs, seed=0)
```

Then the `interleave()` function can pull randomly from the files.
Make sure to skip the first row since it contains the headers.


```python
dataset = filepath_dataset.interleave(
    lambda fp: tf.data.TextLineDataset(fp).skip(1)
)
```


```python
for line in dataset.take(5):
    print(line.numpy())
```

    b'3.9375,51.0,5.231805929919138,1.0188679245283019,1012.0,2.7277628032345014,34.14,-118.2,2.17'
    b'8.3252,41.0,6.984126984126984,1.0238095238095237,322.0,2.5555555555555554,37.88,-122.23,4.526'
    b'2.1691,31.0,5.322115384615385,1.0384615384615385,1326.0,3.1875,36.68,-119.8,0.667'
    b'2.1378,39.0,3.7617834394904457,1.1184713375796178,1379.0,1.756687898089172,33.77,-118.17,1.804'
    b'4.3359,47.0,5.070281124497992,1.0502008032128514,1514.0,3.040160642570281,34.14,-118.19,2.092'


Note that the data are still byte strings that need to be processed further, still.

### Preprocessing the data


```python

```
