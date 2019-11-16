# Chapter 13. Introduction to Modeling Libraries in Python

## 13.1 Interfacing between pandas and model code

Use the `.values` property to turn a pandas DataFrame to a numpy array.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

%matplotlib inline
```


```python
data = pd.DataFrame({
    'x0': [1, 2, 3, 4, 5],
    'x1': [0.01, -0.01, 0.25, -4.1, 0.],
    'y': [-1.5, 0., 3.6, 1.3, -2.]
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
      <th>x0</th>
      <th>x1</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.01</td>
      <td>-1.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>-0.01</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.25</td>
      <td>3.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>-4.10</td>
      <td>1.3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.00</td>
      <td>-2.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.columns
```




    Index(['x0', 'x1', 'y'], dtype='object')




```python
data.values
```




    array([[ 1.  ,  0.01, -1.5 ],
           [ 2.  , -0.01,  0.  ],
           [ 3.  ,  0.25,  3.6 ],
           [ 4.  , -4.1 ,  1.3 ],
           [ 5.  ,  0.  , -2.  ]])



A numpy array can easily be converted back to a pandas DataFrame.


```python
df2 = pd.DataFrame(data.values, columns=['one', 'two', 'three'])
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
      <th>one</th>
      <th>two</th>
      <th>three</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>0.01</td>
      <td>-1.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>-0.01</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.0</td>
      <td>0.25</td>
      <td>3.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.0</td>
      <td>-4.10</td>
      <td>1.3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>0.00</td>
      <td>-2.0</td>
    </tr>
  </tbody>
</table>
</div>



Use the `loc()` method to select a subset of columns.


```python
model_cols = ['x0', 'x1']
data.loc[:, model_cols].values
```




    array([[ 1.  ,  0.01],
           [ 2.  , -0.01],
           [ 3.  ,  0.25],
           [ 4.  , -4.1 ],
           [ 5.  ,  0.  ]])



The next example demonstrates the process of turning a categorical column into columns of dummary variables.


```python
# Add a Categorical column
data['category'] = pd.Categorical(['a', 'b', 'a', 'a', 'b'],
                                  categories=['a', 'b'])
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
      <th>x0</th>
      <th>x1</th>
      <th>y</th>
      <th>category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.01</td>
      <td>-1.5</td>
      <td>a</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>-0.01</td>
      <td>0.0</td>
      <td>b</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.25</td>
      <td>3.6</td>
      <td>a</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>-4.10</td>
      <td>1.3</td>
      <td>a</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.00</td>
      <td>-2.0</td>
      <td>b</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Create a dummy DataFrame and join the reuslt.
dummies = pd.get_dummies(data.category, prefix='category')
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
      <th>category_a</th>
      <th>category_b</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_with_dummies = data.drop('category', axis=1).join(dummies)
data_with_dummies
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
      <th>x0</th>
      <th>x1</th>
      <th>y</th>
      <th>category_a</th>
      <th>category_b</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.01</td>
      <td>-1.5</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>-0.01</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.25</td>
      <td>3.6</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>-4.10</td>
      <td>1.3</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.00</td>
      <td>-2.0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## 13.2 Creating model descriptions with Patsy

The Patsy library is for describing statistical models in an R-like syntax.


```python
data = pd.DataFrame({
    'x0': [1, 2, 3, 4, 5],
    'x1': [0.01, -0.01, 0.25, -4.1, 0.],
    'y': [-1.5, 0., 3.6, 1.3, -2.]
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
      <th>x0</th>
      <th>x1</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.01</td>
      <td>-1.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>-0.01</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.25</td>
      <td>3.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>-4.10</td>
      <td>1.3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.00</td>
      <td>-2.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
import patsy

y, X = patsy.dmatrices('y ~ x0 + x1', data)
y
```




    DesignMatrix with shape (5, 1)
         y
      -1.5
       0.0
       3.6
       1.3
      -2.0
      Terms:
        'y' (column 0)




```python
X
```




    DesignMatrix with shape (5, 3)
      Intercept  x0     x1
              1   1   0.01
              1   2  -0.01
              1   3   0.25
              1   4  -4.10
              1   5   0.00
      Terms:
        'Intercept' (column 0)
        'x0' (column 1)
        'x1' (column 2)




```python
np.asarray(y)
```




    array([[-1.5],
           [ 0. ],
           [ 3.6],
           [ 1.3],
           [-2. ]])




```python
np.asarray(X)
```




    array([[ 1.  ,  1.  ,  0.01],
           [ 1.  ,  2.  , -0.01],
           [ 1.  ,  3.  ,  0.25],
           [ 1.  ,  4.  , -4.1 ],
           [ 1.  ,  5.  ,  0.  ]])



The  intercept can be suppressed by adding a `+ 0` term.


```python
patsy.dmatrices('y ~ x0 + x1 + 0', data)[1]
```




    DesignMatrix with shape (5, 2)
      x0     x1
       1   0.01
       2  -0.01
       3   0.25
       4  -4.10
       5   0.00
      Terms:
        'x0' (column 0)
        'x1' (column 1)



The Patsy design matrix can be passed directly to algorithms like the `linalg.lstsq()` method from numpy.


```python
coef, resid, _, _ = np.linalg.lstsq(X, y, rcond=None)
coef
```




    array([[ 0.31290976],
           [-0.07910564],
           [-0.26546384]])




```python
resid
```




    array([19.63791494])




```python
coef = pd.Series(coef.squeeze(), index=X.design_info.column_names)
coef
```




    Intercept    0.312910
    x0          -0.079106
    x1          -0.265464
    dtype: float64



### Data transformations in Patsy formulae

Python code can be in a Patsy formula string.


```python
y, X = patsy.dmatrices('y ~ x0 + np.log(np.abs(x1) + 1)', data)
X
```




    DesignMatrix with shape (5, 3)
      Intercept  x0  np.log(np.abs(x1) + 1)
              1   1                 0.00995
              1   2                 0.00995
              1   3                 0.22314
              1   4                 1.62924
              1   5                 0.00000
      Terms:
        'Intercept' (column 0)
        'x0' (column 1)
        'np.log(np.abs(x1) + 1)' (column 2)



Some common transofrmations include `standardize()` and `center()` which sets the mean to 0 with standard deviation to 1 and substracting the mean, respectively.


```python
y, X = patsy.dmatrices('y ~ standardize(x0) + center(x1)', data=data)
X
```




    DesignMatrix with shape (5, 3)
      Intercept  standardize(x0)  center(x1)
              1         -1.41421        0.78
              1         -0.70711        0.76
              1          0.00000        1.02
              1          0.70711       -3.33
              1          1.41421        0.77
      Terms:
        'Intercept' (column 0)
        'standardize(x0)' (column 1)
        'center(x1)' (column 2)



The transformations for the training data should be repeated, but not recalculated, for the test data.
Patsy can do these transformations of the test data using the values from the training data.


```python
new_data = pd.DataFrame({
    'x0': [6, 7, 8, 9],
    'x1': [3.1, -0.5, 0, 2.3],
    'y': [1, 2, 3, 4]
})
new_data
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
      <th>x0</th>
      <th>x1</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>3.1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>-0.5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>0.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9</td>
      <td>2.3</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
new_X = patsy.build_design_matrices([X.design_info], new_data)
new_X
```




    [DesignMatrix with shape (4, 3)
       Intercept  standardize(x0)  center(x1)
               1          2.12132        3.87
               1          2.82843        0.27
               1          3.53553        0.77
               1          4.24264        3.07
       Terms:
         'Intercept' (column 0)
         'standardize(x0)' (column 1)
         'center(x1)' (column 2)]



Because the `+` symbol in Patsy formulae do not mean to add the columns, if this is the desired behaviour, the `I()` function must be used.


```python
y, X = patsy.dmatrices('y ~ I(x0 + x1)', data=data)
X
```




    DesignMatrix with shape (5, 2)
      Intercept  I(x0 + x1)
              1        1.01
              1        1.99
              1        3.25
              1       -0.10
              1        5.00
      Terms:
        'Intercept' (column 0)
        'I(x0 + x1)' (column 1)



### Categorical data and Patsy

Patsy converts categorical data to dummy variables automatically.
If there is an intercept, one of the levels will be left out to avoid colinearity.


```python
data = pd.DataFrame({
    'key1': ['a', 'a', 'b', 'b', 'a', 'b', 'a', 'b'],
    'key2': [0, 1, 0, 1, 0, 1, 0, 0],
    'v1':[1,2,3,4,5,6,7,8],
    'v2': [-1, 0, 2.5, -0.5, 4.0, -1.2, 0.2, -1.7]
})

y, X = patsy.dmatrices('v2 ~ key1', data=data)
X
```




    DesignMatrix with shape (8, 2)
      Intercept  key1[T.b]
              1          0
              1          0
              1          1
              1          1
              1          0
              1          1
              1          0
              1          1
      Terms:
        'Intercept' (column 0)
        'key1' (column 1)




```python
# Without an intercept.
y, X = patsy.dmatrices('v2 ~ 0 + key1', data=data)
X
```




    DesignMatrix with shape (8, 2)
      key1[a]  key1[b]
            1        0
            1        0
            0        1
            0        1
            1        0
            0        1
            1        0
            0        1
      Terms:
        'key1' (columns 0:2)



A numerical column can be interpreted as a categorical variable using the `C()` function.


```python
y, X = patsy.dmatrices('v2 ~ C(key2)', data=data)
X
```




    DesignMatrix with shape (8, 2)
      Intercept  C(key2)[T.1]
              1             0
              1             1
              1             0
              1             1
              1             0
              1             1
              1             0
              1             0
      Terms:
        'Intercept' (column 0)
        'C(key2)' (column 1)



With multiple terms, we can introduce interactions.


```python
# Add a new categorical column.
data['key2'] = data['key2'].map({0: 'zero', 1: 'one'})
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
      <th>key1</th>
      <th>key2</th>
      <th>v1</th>
      <th>v2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>zero</td>
      <td>1</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a</td>
      <td>one</td>
      <td>2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>b</td>
      <td>zero</td>
      <td>3</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>b</td>
      <td>one</td>
      <td>4</td>
      <td>-0.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>a</td>
      <td>zero</td>
      <td>5</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>b</td>
      <td>one</td>
      <td>6</td>
      <td>-1.2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>a</td>
      <td>zero</td>
      <td>7</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>b</td>
      <td>zero</td>
      <td>8</td>
      <td>-1.7</td>
    </tr>
  </tbody>
</table>
</div>




```python
y, X = patsy.dmatrices('v2 ~ key1 + key2', data=data)
X
```




    DesignMatrix with shape (8, 3)
      Intercept  key1[T.b]  key2[T.zero]
              1          0             1
              1          0             0
              1          1             1
              1          1             0
              1          0             1
              1          1             0
              1          0             1
              1          1             1
      Terms:
        'Intercept' (column 0)
        'key1' (column 1)
        'key2' (column 2)




```python
# Interaction terms are expressed with the `a:b` notation like in R.
y, X  = patsy.dmatrices('v2 ~ key1 + key2 + key1:key2', data=data)
X
```




    DesignMatrix with shape (8, 4)
      Intercept  key1[T.b]  key2[T.zero]  key1[T.b]:key2[T.zero]
              1          0             1                       0
              1          0             0                       0
              1          1             1                       1
              1          1             0                       0
              1          0             1                       0
              1          1             0                       0
              1          0             1                       0
              1          1             1                       1
      Terms:
        'Intercept' (column 0)
        'key1' (column 1)
        'key2' (column 2)
        'key1:key2' (column 3)



## 13.3 Introduction to statsmodels

The statsmodels library is used to fit many kinds of statistical models (mainly frequentist), perform statistical tests, and data exploration and visualization.

### Estimating linear models

There are two main APIs for statsmodels, one is array-based and the other formula-based.


```python
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Some random data for example modeling.
def dnorm(mean, variance, size=1):
    if isinstance(size, int):
        size = size,
    return mean + np.sqrt(variance) * np.random.randn(*size)

# For reproducibility
np.random.seed(12345)

N=100
X = np.c_[dnorm(0, 0.4, size=N),
          dnorm(0, 0.6, size=N),
          dnorm(0, 0.2, size=N)]

eps = dnorm(0, 0.1, size=N)
beta = [0.1, 0.3, 0.5]

y = np.dot(X, beta) + eps

X[:5]
```




    array([[-0.12946849, -1.21275292,  0.50422488],
           [ 0.30291036, -0.43574176, -0.25417986],
           [-0.32852189, -0.02530153,  0.13835097],
           [-0.35147471, -0.71960511, -0.25821463],
           [ 1.2432688 , -0.37379916, -0.52262905]])




```python
y[:5]
```




    array([ 0.42786349, -0.67348041, -0.09087764, -0.48949442, -0.12894109])



The `sm.add_constant()` function can add a column for the intercept.


```python
X_model = sm.add_constant(X)
X_model[:5]
```




    array([[ 1.        , -0.12946849, -1.21275292,  0.50422488],
           [ 1.        ,  0.30291036, -0.43574176, -0.25417986],
           [ 1.        , -0.32852189, -0.02530153,  0.13835097],
           [ 1.        , -0.35147471, -0.71960511, -0.25821463],
           [ 1.        ,  1.2432688 , -0.37379916, -0.52262905]])



The `sm.OLS()` class can fit an OLS linear regression.


```python
model = sm.OLS(y, X)
model
```




    <statsmodels.regression.linear_model.OLS at 0x1c38557050>




```python
result = model.fit()
result
```




    <statsmodels.regression.linear_model.RegressionResultsWrapper at 0x1c37b88ad0>




```python
result.aic
```




    74.60927884012554




```python
print(result.summary())
```

    OLS Regression Results                                
    =======================================================================================
    Dep. Variable:                      y   R-squared (uncentered):                   0.430
    Model:                            OLS   Adj. R-squared (uncentered):              0.413
    Method:                 Least Squares   F-statistic:                              24.42
    Date:                Fri, 15 Nov 2019   Prob (F-statistic):                    7.44e-12
    Time:                        22:22:30   Log-Likelihood:                         -34.305
    No. Observations:                 100   AIC:                                      74.61
    Df Residuals:                      97   BIC:                                      82.42
    Df Model:                           3                                                  
    Covariance Type:            nonrobust                                                  
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    x1             0.1783      0.053      3.364      0.001       0.073       0.283
    x2             0.2230      0.046      4.818      0.000       0.131       0.315
    x3             0.5010      0.080      6.237      0.000       0.342       0.660
    ==============================================================================
    Omnibus:                        4.662   Durbin-Watson:                   2.201
    Prob(Omnibus):                  0.097   Jarque-Bera (JB):                4.098
    Skew:                           0.481   Prob(JB):                        0.129
    Kurtosis:                       3.243   Cond. No.                         1.74
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



```python
# Access the model's coefficients.
result.params
```




    array([0.17826108, 0.22303962, 0.50095093])



The following is a simillar example using the formula API.
Note how an intercept is automatically included.


```python
data = pd.DataFrame(X, columns=['col0', 'col1', 'col2'])
data['y'] = y
data[:5]
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
      <th>col0</th>
      <th>col1</th>
      <th>col2</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.129468</td>
      <td>-1.212753</td>
      <td>0.504225</td>
      <td>0.427863</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.302910</td>
      <td>-0.435742</td>
      <td>-0.254180</td>
      <td>-0.673480</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.328522</td>
      <td>-0.025302</td>
      <td>0.138351</td>
      <td>-0.090878</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.351475</td>
      <td>-0.719605</td>
      <td>-0.258215</td>
      <td>-0.489494</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.243269</td>
      <td>-0.373799</td>
      <td>-0.522629</td>
      <td>-0.128941</td>
    </tr>
  </tbody>
</table>
</div>




```python
results = smf.ols('y ~ col0 + col1 + col2', data=data).fit()
print(results.summary())
```

    OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                      y   R-squared:                       0.435
    Model:                            OLS   Adj. R-squared:                  0.418
    Method:                 Least Squares   F-statistic:                     24.68
    Date:                Fri, 15 Nov 2019   Prob (F-statistic):           6.37e-12
    Time:                        22:22:30   Log-Likelihood:                -33.835
    No. Observations:                 100   AIC:                             75.67
    Df Residuals:                      96   BIC:                             86.09
    Df Model:                           3                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept      0.0336      0.035      0.952      0.343      -0.036       0.104
    col0           0.1761      0.053      3.320      0.001       0.071       0.281
    col1           0.2248      0.046      4.851      0.000       0.133       0.317
    col2           0.5148      0.082      6.304      0.000       0.353       0.677
    ==============================================================================
    Omnibus:                        4.504   Durbin-Watson:                   2.223
    Prob(Omnibus):                  0.105   Jarque-Bera (JB):                3.957
    Skew:                           0.475   Prob(JB):                        0.138
    Kurtosis:                       3.222   Cond. No.                         2.38
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



```python
results.params
```




    Intercept    0.033559
    col0         0.176149
    col1         0.224826
    col2         0.514808
    dtype: float64



The `y` for new data can be predicted using the model, too.


```python
results.predict(data[:5])
```




    0   -0.002327
    1   -0.141904
    2    0.041226
    3   -0.323070
    4   -0.100535
    dtype: float64



## 13.4 Introduction to scikit-learn

The scikit-lean library contains a broad selection of supervised and unsupervised machine learning methods.
The example below is about passenger survival rates on the Titanic.


```python
# Load training data.
train = pd.read_csv('assets/datasets/titanic/train.csv')

# Load test data.
test = pd.read_csv('assets/datasets/titanic/test.csv')

train[:4]
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Check for missing data.
train.isnull().sum()
```




    PassengerId      0
    Survived         0
    Pclass           0
    Name             0
    Sex              0
    Age            177
    SibSp            0
    Parch            0
    Ticket           0
    Fare             0
    Cabin          687
    Embarked         2
    dtype: int64




```python
test.isnull().sum()
```




    PassengerId      0
    Pclass           0
    Name             0
    Sex              0
    Age             86
    SibSp            0
    Parch            0
    Ticket           0
    Fare             1
    Cabin          327
    Embarked         0
    dtype: int64



There is missing data for age, so we will do a simple method of imputation and use the median value of the training data.


```python
# Median age of training data for imputing missing values.
impute_value = train.Age.median()

# Fill in missing data for training and testing data.
train['Age'] = train.Age.fillna(impute_value)
test['Age'] = test.Age.fillna(impute_value)
```

An `'IsFemale'` column is created as a dummy for `'Sex'`.


```python
train['IsFemale'] = (train['Sex'] == 'female').astype(int)
test['IsFemale'] = (test['Sex'] == 'female').astype(int)
```

The features for the model will be 'Pclass', 'IsFemale', and 'Age'.


```python
predictors = ['Pclass', 'IsFemale', 'Age']
X_train = train[predictors].values
X_test = test[predictors].values
y_train = train.Survived.values

X_train[:5]
```




    array([[ 3.,  0., 22.],
           [ 1.,  1., 38.],
           [ 3.,  1., 26.],
           [ 1.,  1., 35.],
           [ 3.,  0., 35.]])




```python
y_train[:5]
```




    array([0, 1, 1, 1, 0])



For the purposes of demonstration, a logistic regression model was used.


```python
from sklearn.linear_model import LogisticRegression

# Instantiate model.
model = LogisticRegression()

# Fit the model.
model.fit(X_train, y_train)
```

    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)





    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='warn', n_jobs=None, penalty='l2',
                       random_state=None, solver='warn', tol=0.0001, verbose=0,
                       warm_start=False)



The testing data can be fed to the fit model for making predictions.


```python
y_predict = model.predict(X_test)
y_predict[:10]
```




    array([0, 0, 0, 0, 1, 0, 1, 0, 1, 0])



Below is an example of using cross-validation to test the accuracy of a model without using the testing data.


```python
from sklearn.linear_model import LogisticRegressionCV

model_cv = LogisticRegressionCV(10, cv=10)
model_cv.fit(X_train, y_train)
```




    LogisticRegressionCV(Cs=10, class_weight=None, cv=10, dual=False,
                         fit_intercept=True, intercept_scaling=1.0, l1_ratios=None,
                         max_iter=100, multi_class='warn', n_jobs=None,
                         penalty='l2', random_state=None, refit=True, scoring=None,
                         solver='lbfgs', tol=0.0001, verbose=0)




```python
y_predict_cv = model_cv.predict(X_test)
```

We can actually see that this model made a few different predictions on the testing set than the original model without cross-validation.


```python
idx = y_predict_cv != y_predict
sum(idx)
```




    10




```python
test.iloc[idx]
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
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>IsFemale</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>41</th>
      <td>933</td>
      <td>1</td>
      <td>Franklin, Mr. Thomas Parham</td>
      <td>male</td>
      <td>28.0</td>
      <td>0</td>
      <td>0</td>
      <td>113778</td>
      <td>26.5500</td>
      <td>D34</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>94</th>
      <td>986</td>
      <td>1</td>
      <td>Birnbaum, Mr. Jakob</td>
      <td>male</td>
      <td>25.0</td>
      <td>0</td>
      <td>0</td>
      <td>13905</td>
      <td>26.0000</td>
      <td>NaN</td>
      <td>C</td>
      <td>0</td>
    </tr>
    <tr>
      <th>146</th>
      <td>1038</td>
      <td>1</td>
      <td>Hilliard, Mr. Herbert Henry</td>
      <td>male</td>
      <td>28.0</td>
      <td>0</td>
      <td>0</td>
      <td>17463</td>
      <td>51.8625</td>
      <td>E46</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>148</th>
      <td>1040</td>
      <td>1</td>
      <td>Crafton, Mr. John Bertram</td>
      <td>male</td>
      <td>28.0</td>
      <td>0</td>
      <td>0</td>
      <td>113791</td>
      <td>26.5500</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>191</th>
      <td>1083</td>
      <td>1</td>
      <td>Salomon, Mr. Abraham L</td>
      <td>male</td>
      <td>28.0</td>
      <td>0</td>
      <td>0</td>
      <td>111163</td>
      <td>26.0000</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>205</th>
      <td>1097</td>
      <td>1</td>
      <td>Omont, Mr. Alfred Fernand</td>
      <td>male</td>
      <td>28.0</td>
      <td>0</td>
      <td>0</td>
      <td>F.C. 12998</td>
      <td>25.7417</td>
      <td>NaN</td>
      <td>C</td>
      <td>0</td>
    </tr>
    <tr>
      <th>252</th>
      <td>1144</td>
      <td>1</td>
      <td>Clark, Mr. Walter Miller</td>
      <td>male</td>
      <td>27.0</td>
      <td>1</td>
      <td>0</td>
      <td>13508</td>
      <td>136.7792</td>
      <td>C89</td>
      <td>C</td>
      <td>0</td>
    </tr>
    <tr>
      <th>266</th>
      <td>1158</td>
      <td>1</td>
      <td>Chisholm, Mr. Roderick Robert Crispin</td>
      <td>male</td>
      <td>28.0</td>
      <td>0</td>
      <td>0</td>
      <td>112051</td>
      <td>0.0000</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>290</th>
      <td>1182</td>
      <td>1</td>
      <td>Rheims, Mr. George Alexander Lucien</td>
      <td>male</td>
      <td>28.0</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17607</td>
      <td>39.6000</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>313</th>
      <td>1205</td>
      <td>3</td>
      <td>Carr, Miss. Jeannie</td>
      <td>female</td>
      <td>37.0</td>
      <td>0</td>
      <td>0</td>
      <td>368364</td>
      <td>7.7500</td>
      <td>NaN</td>
      <td>Q</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


