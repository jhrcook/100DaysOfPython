# Chapter 2. End-to-End Machine Learning Project

Here are the steps we will follow:

1. Look at the big picture.
2. Get the data.
3. Discover and visualize the data to gain insights.
4. Prepare the data for ML algorithms.
5. Select a model and train.
6. Present the solution.
7. Launch, monitor, and maintain the system.

## Working with real data

Some popular open data repositories include:

- [UC Irvine ML Repository](http://archive.ics.uci.edu/ml/index.php)
- [Kaggle datasets](https://www.kaggle.com/datasets)
- [Amazon's AWS datasets](https://registry.opendata.aws)
- [Wikipedias list of machine learning datasets](https://en.wikipedia.org/wiki/List_of_datasets_for_machine-learning_research
- [this](https://www.quora.com/Where-can-I-find-large-datasets-open-to-the-public) Quora.com questoins
- the [Datasets subreddit](https://www.reddit.com/r/datasets)

There are a few "meta-portals":

- http://dataportals.org/
- http://opendatamonitor.eu/
- http://quandl.com/

This example analysis will use the California Housing Prices dataset from the StatLib repository based on data from the 1990 CA census.
The goal is the create a model to predict the median housing price in any district.

### Download the data


```python
import os, tarfile
from six.moves import urllib
import pandas as pd
import numpy as np

DOWNLOAD_ROOT = 'https://raw.githubusercontent.com/ageron/handson-ml/master/'
HOUSING_PATH = 'assets/homl/housing'
HOUSING_URL = DOWNLOAD_ROOT + 'datasets/housing/housing.tgz'

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    ''' Download and extract housing data.'''
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    
    tgz_path = os.path.join(housing_path, 'housing.tgz')
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

def load_housing_data(housing_path=HOUSING_PATH):
    '''Load in housing data as a pandas DataFrame.'''
    csv_path = os.path.join(housing_path, 'housing.csv')
    return pd.read_csv(csv_path)

# Download the data.
fetch_housing_data()

# Loading the data as a pandas DataFrame.
housing = load_housing_data()
housing.head()
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
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-122.23</td>
      <td>37.88</td>
      <td>41.0</td>
      <td>880.0</td>
      <td>129.0</td>
      <td>322.0</td>
      <td>126.0</td>
      <td>8.3252</td>
      <td>452600.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-122.22</td>
      <td>37.86</td>
      <td>21.0</td>
      <td>7099.0</td>
      <td>1106.0</td>
      <td>2401.0</td>
      <td>1138.0</td>
      <td>8.3014</td>
      <td>358500.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-122.24</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>1467.0</td>
      <td>190.0</td>
      <td>496.0</td>
      <td>177.0</td>
      <td>7.2574</td>
      <td>352100.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-122.25</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>1274.0</td>
      <td>235.0</td>
      <td>558.0</td>
      <td>219.0</td>
      <td>5.6431</td>
      <td>341300.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-122.25</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>1627.0</td>
      <td>280.0</td>
      <td>565.0</td>
      <td>259.0</td>
      <td>3.8462</td>
      <td>342200.0</td>
      <td>NEAR BAY</td>
    </tr>
  </tbody>
</table>
</div>




```python
housing.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 20640 entries, 0 to 20639
    Data columns (total 10 columns):
    longitude             20640 non-null float64
    latitude              20640 non-null float64
    housing_median_age    20640 non-null float64
    total_rooms           20640 non-null float64
    total_bedrooms        20433 non-null float64
    population            20640 non-null float64
    households            20640 non-null float64
    median_income         20640 non-null float64
    median_house_value    20640 non-null float64
    ocean_proximity       20640 non-null object
    dtypes: float64(9), object(1)
    memory usage: 1.6+ MB



```python
# Number of missing data points per column.
housing.isnull().sum()
```




    longitude               0
    latitude                0
    housing_median_age      0
    total_rooms             0
    total_bedrooms        207
    population              0
    households              0
    median_income           0
    median_house_value      0
    ocean_proximity         0
    dtype: int64




```python
# The values of 'ocean_proximity' are categorical.
housing.ocean_proximity.value_counts()
```




    <1H OCEAN     9136
    INLAND        6551
    NEAR OCEAN    2658
    NEAR BAY      2290
    ISLAND           5
    Name: ocean_proximity, dtype: int64




```python
housing.describe()
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
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20433.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-119.569704</td>
      <td>35.631861</td>
      <td>28.639486</td>
      <td>2635.763081</td>
      <td>537.870553</td>
      <td>1425.476744</td>
      <td>499.539680</td>
      <td>3.870671</td>
      <td>206855.816909</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.003532</td>
      <td>2.135952</td>
      <td>12.585558</td>
      <td>2181.615252</td>
      <td>421.385070</td>
      <td>1132.462122</td>
      <td>382.329753</td>
      <td>1.899822</td>
      <td>115395.615874</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-124.350000</td>
      <td>32.540000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>0.499900</td>
      <td>14999.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-121.800000</td>
      <td>33.930000</td>
      <td>18.000000</td>
      <td>1447.750000</td>
      <td>296.000000</td>
      <td>787.000000</td>
      <td>280.000000</td>
      <td>2.563400</td>
      <td>119600.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-118.490000</td>
      <td>34.260000</td>
      <td>29.000000</td>
      <td>2127.000000</td>
      <td>435.000000</td>
      <td>1166.000000</td>
      <td>409.000000</td>
      <td>3.534800</td>
      <td>179700.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>-118.010000</td>
      <td>37.710000</td>
      <td>37.000000</td>
      <td>3148.000000</td>
      <td>647.000000</td>
      <td>1725.000000</td>
      <td>605.000000</td>
      <td>4.743250</td>
      <td>264725.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>-114.310000</td>
      <td>41.950000</td>
      <td>52.000000</td>
      <td>39320.000000</td>
      <td>6445.000000</td>
      <td>35682.000000</td>
      <td>6082.000000</td>
      <td>15.000100</td>
      <td>500001.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
import matplotlib.pyplot as plt
%matplotlib inline

# Some good default values to use for my Jupyter notebook.
plt.style.use('seaborn-whitegrid')
plt.rc('figure', figsize=(8, 5), facecolor='white')

housing.hist(bins=50)
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x121520b50>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1224b9bd0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1224fbed0>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x12252fb90>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x12256ff10>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1225a3bd0>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x1225e3410>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x122618c10>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x122623790>]],
          dtype=object)




![png](homl_ch02_files/homl_ch02_6_1.png)


A few things to note about the data:

1. The 'median_income' has been scaled and capped to a range of 0.5 and 15.
2. The 'housing_median_age' and 'median_house_value' have been capped, the latter potentially being a problem because that data is the predicted value.
3. The attributes have different scales (discussed later).
4. Many of the histograms have a long tail - a good sign to transform the data, depending on the model.

### Create a test set




```python
from sklearn.model_selection import train_test_split

# Split the data into training and testing data.
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=0)
```

The author wasnted to ensure that the median income was properly sampled in the training and testing data.
Therefore, he created a categorical variable and used `sklearn.model_selection.StratifiedShuffleSplit()` to split the data into training and testing sets while maintaining equal distributions of this category.


```python
housing['income_cat'] = np.ceil(housing.median_income / 1.5)
housing['income_cat'].where(housing.income_cat < 5.0, 5.0, inplace=True)
housing.head()
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
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
      <th>ocean_proximity</th>
      <th>income_cat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-122.23</td>
      <td>37.88</td>
      <td>41.0</td>
      <td>880.0</td>
      <td>129.0</td>
      <td>322.0</td>
      <td>126.0</td>
      <td>8.3252</td>
      <td>452600.0</td>
      <td>NEAR BAY</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-122.22</td>
      <td>37.86</td>
      <td>21.0</td>
      <td>7099.0</td>
      <td>1106.0</td>
      <td>2401.0</td>
      <td>1138.0</td>
      <td>8.3014</td>
      <td>358500.0</td>
      <td>NEAR BAY</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-122.24</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>1467.0</td>
      <td>190.0</td>
      <td>496.0</td>
      <td>177.0</td>
      <td>7.2574</td>
      <td>352100.0</td>
      <td>NEAR BAY</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-122.25</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>1274.0</td>
      <td>235.0</td>
      <td>558.0</td>
      <td>219.0</td>
      <td>5.6431</td>
      <td>341300.0</td>
      <td>NEAR BAY</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-122.25</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>1627.0</td>
      <td>280.0</td>
      <td>565.0</td>
      <td>259.0</td>
      <td>3.8462</td>
      <td>342200.0</td>
      <td>NEAR BAY</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
```


```python
# Compare splits
housing['income_cat'].value_counts() / len(housing)
```




    3.0    0.350581
    2.0    0.318847
    4.0    0.176308
    5.0    0.114438
    1.0    0.039826
    Name: income_cat, dtype: float64




```python
strat_train_set['income_cat'].value_counts() / len(strat_train_set)
```




    3.0    0.350594
    2.0    0.318859
    4.0    0.176296
    5.0    0.114402
    1.0    0.039850
    Name: income_cat, dtype: float64




```python
strat_test_set['income_cat'].value_counts() / len(strat_test_set)
```




    3.0    0.350533
    2.0    0.318798
    4.0    0.176357
    5.0    0.114583
    1.0    0.039729
    Name: income_cat, dtype: float64




```python
# Remove unneeded columns.
strat_test_set.drop(['income_cat'], axis=1, inplace=True)
strat_train_set.drop(['income_cat'], axis=1, inplace=True)
```

## Discover and visualize the data to gain insights


```python
# A copy of the training data to play around with.
housing = strat_train_set.copy()
```


```python
housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1)

```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a2544e210>




![png](homl_ch02_files/homl_ch02_18_1.png)



```python
housing.plot(kind='scatter',
             x = 'longitude',
             y = 'latitude',
             alpha = 0.4,
             s=housing.population/100,
             label='population',
             c='median_house_value',
             cmap=plt.get_cmap('jet'), colorbar=True)
plt.legend()
```




    <matplotlib.legend.Legend at 0x122b53190>




![png](homl_ch02_files/homl_ch02_19_1.png)



```python
corr_matrix = housing.corr()
corr_matrix
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
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>longitude</th>
      <td>1.000000</td>
      <td>-0.924903</td>
      <td>-0.108097</td>
      <td>0.041547</td>
      <td>0.065183</td>
      <td>0.096933</td>
      <td>0.051827</td>
      <td>-0.013645</td>
      <td>-0.043236</td>
    </tr>
    <tr>
      <th>latitude</th>
      <td>-0.924903</td>
      <td>1.000000</td>
      <td>0.010784</td>
      <td>-0.034359</td>
      <td>-0.064436</td>
      <td>-0.106834</td>
      <td>-0.069613</td>
      <td>-0.081262</td>
      <td>-0.145570</td>
    </tr>
    <tr>
      <th>housing_median_age</th>
      <td>-0.108097</td>
      <td>0.010784</td>
      <td>1.000000</td>
      <td>-0.359036</td>
      <td>-0.317509</td>
      <td>-0.293215</td>
      <td>-0.300756</td>
      <td>-0.113458</td>
      <td>0.107144</td>
    </tr>
    <tr>
      <th>total_rooms</th>
      <td>0.041547</td>
      <td>-0.034359</td>
      <td>-0.359036</td>
      <td>1.000000</td>
      <td>0.929552</td>
      <td>0.853612</td>
      <td>0.918026</td>
      <td>0.196382</td>
      <td>0.137469</td>
    </tr>
    <tr>
      <th>total_bedrooms</th>
      <td>0.065183</td>
      <td>-0.064436</td>
      <td>-0.317509</td>
      <td>0.929552</td>
      <td>1.000000</td>
      <td>0.874315</td>
      <td>0.980162</td>
      <td>-0.009282</td>
      <td>0.053544</td>
    </tr>
    <tr>
      <th>population</th>
      <td>0.096933</td>
      <td>-0.106834</td>
      <td>-0.293215</td>
      <td>0.853612</td>
      <td>0.874315</td>
      <td>1.000000</td>
      <td>0.903795</td>
      <td>0.003431</td>
      <td>-0.023797</td>
    </tr>
    <tr>
      <th>households</th>
      <td>0.051827</td>
      <td>-0.069613</td>
      <td>-0.300756</td>
      <td>0.918026</td>
      <td>0.980162</td>
      <td>0.903795</td>
      <td>1.000000</td>
      <td>0.011840</td>
      <td>0.069177</td>
    </tr>
    <tr>
      <th>median_income</th>
      <td>-0.013645</td>
      <td>-0.081262</td>
      <td>-0.113458</td>
      <td>0.196382</td>
      <td>-0.009282</td>
      <td>0.003431</td>
      <td>0.011840</td>
      <td>1.000000</td>
      <td>0.688883</td>
    </tr>
    <tr>
      <th>median_house_value</th>
      <td>-0.043236</td>
      <td>-0.145570</td>
      <td>0.107144</td>
      <td>0.137469</td>
      <td>0.053544</td>
      <td>-0.023797</td>
      <td>0.069177</td>
      <td>0.688883</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
corr_matrix.median_house_value.sort_values(ascending=False)
```




    median_house_value    1.000000
    median_income         0.688883
    total_rooms           0.137469
    housing_median_age    0.107144
    households            0.069177
    total_bedrooms        0.053544
    population           -0.023797
    longitude            -0.043236
    latitude             -0.145570
    Name: median_house_value, dtype: float64




```python
from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes])
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x1a262aa190>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1a262c9510>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1a262f9d10>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1a26334550>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x1a263e5d50>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1a2625f590>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1a25db5d90>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1a261545d0>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x1a2615d150>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1a25d07ad0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1a25d72e10>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1a25f39650>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x1a25f6ee50>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1a25fb0690>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1a2641ee90>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1a2645e6d0>]],
          dtype=object)




![png](homl_ch02_files/homl_ch02_22_1.png)


From the above scatter matrix, the most promizing attribute seems to be median income.
The following plot is a closer look.


```python
housing.plot(kind='scatter',
             x='median_income',
             y='median_house_value',
             alpha=0.05)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a27091650>




![png](homl_ch02_files/homl_ch02_24_1.png)


The actual numbers of rooms and bedrooms are not very useful.
Instead, we want the average number of rooms and bedrooms per house.


```python
housing['rooms_per_household'] = housing.total_rooms / housing.households
housing['bedrooms_per_room'] = housing.total_bedrooms / housing.households
housing['population_per_household'] = housing.population / housing.households

# Check the correlations again.
corr_matrix = housing.corr()
corr_matrix.median_house_value.sort_values(ascending=False)
```




    median_house_value          1.000000
    median_income               0.688883
    rooms_per_household         0.157620
    total_rooms                 0.137469
    housing_median_age          0.107144
    households                  0.069177
    total_bedrooms              0.053544
    population                 -0.023797
    population_per_household   -0.026888
    longitude                  -0.043236
    bedrooms_per_room          -0.048998
    latitude                   -0.145570
    Name: median_house_value, dtype: float64



## Prepare the data for machine learning algorithms

It is important to have a functional flow for this preparation such the transofrmations are easily reproducible and extendible.
Also, the same transformations will need to be done on the input data when the model is in use.
Finally, it allows for faster prototyping.

To begin, we reset the training data and separated the labels.


```python
housing = strat_train_set.drop('median_house_value', axis=1)
housing_labels = strat_train_set['median_house_value'].copy()
```

### Data cleaning


```python

```
