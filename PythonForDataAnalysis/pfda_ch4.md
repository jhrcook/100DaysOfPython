# Chapter 4. NumPy Basics: Arrays and Vectorized Computation

NumPy stands for "Numerical Python" and it provides a solid foundation for numeric computation in Python.
Here are some of the main features covered in this book:

- fast vectorized array operations
- common array algorithms such as sorting and set operations
- efficient descriptive statistics
- expressing conditional logic using arrays (instead of looping)
- group-wise data manipulation

NumPy is designed for efficiency on large arrays of data; the author continues to describe some of the ways it is able to accomplish this.
Here is a simple demonstration of thie preformance NumPy.


```python
import numpy as np

my_array = np.arange(1000000)
my_list = list(range(1000000))
```


```python
%time for _ in range(10): my_array = my_array * 2
```

    CPU times: user 18.5 ms, sys: 12.7 ms, total: 31.3 ms
    Wall time: 74.1 ms



```python
%time for _ in range(10): my_list = [x * 2 for x in my_list]
```

    CPU times: user 984 ms, sys: 304 ms, total: 1.29 s
    Wall time: 1.84 s


## 4.1 The NumPy ndarray: a multidimensional array object

`ndarray` stands for N-dimensional array.
It is a fast flexible data structure with which we can conduct vectorized computations.
Here is a simple example.


```python
data = np.random.randn(2, 3)
data
```




    array([[-0.33033923,  0.78424616, -0.26785417],
           [ 0.29856233,  1.84409451,  1.25021299]])




```python
data * 10
```




    array([[-3.30339233,  7.84246159, -2.6785417 ],
           [ 2.98562333, 18.44094505, 12.50212994]])




```python
data + data
```




    array([[-0.66067847,  1.56849232, -0.53570834],
           [ 0.59712467,  3.68818901,  2.50042599]])



### Creating ndarrays


```python
data1 = [6, 7.5, 8, 0, 1]
arr1 = np.array(data1)
arr1
```




    array([6. , 7.5, 8. , 0. , 1. ])




```python
data2 = [[1, 2, 3, 4], [5, 6, 7, 8]]
arr2 = np.array(data2)
arr2
```




    array([[1, 2, 3, 4],
           [5, 6, 7, 8]])




```python
arr2.ndim
```




    2




```python
arr2.shape
```




    (2, 4)



NumPy tries to infer a good data type for the array based on data it is given.


```python
arr1.dtype
```




    dtype('float64')




```python
arr2.dtype
```




    dtype('int64')



Arrays can also be made using `zeros`, `ones`, and `empty` and passing in a tuple for the shape.


```python
np.zeros(10)
```




    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])




```python
numeric_strings = np.array(['1.25', '-9.6', '42'], dtype=np.string_)
numeric_strings.astype(float)
```




    array([ 1.25, -9.6 , 42.  ])



### Arithmetic is NumPy arrays

NumPy arrays can be operated on as vectors.


```python
arr = np.array([[1., 2., 3.], [4., 5., 6.]])
arr
```




    array([[1., 2., 3.],
           [4., 5., 6.]])




```python
arr * arr
```




    array([[ 1.,  4.,  9.],
           [16., 25., 36.]])




```python
arr  - arr
```




    array([[0., 0., 0.],
           [0., 0., 0.]])




```python
1/ arr
```




    array([[1.        , 0.5       , 0.33333333],
           [0.25      , 0.2       , 0.16666667]])




```python
arr ** 0.5
```




    array([[1.        , 1.41421356, 1.73205081],
           [2.        , 2.23606798, 2.44948974]])




```python
arr2 = np.array([[0., 4., 1.], [7., 2., 12.]])
arr2
```




    array([[ 0.,  4.,  1.],
           [ 7.,  2., 12.]])




```python
arr2 > arr
```




    array([[False,  True, False],
           [ True, False,  True]])



### Indexing and slicing

1-D arrays act similarly to Python lists.


```python
arr = np.arange(10)
arr[5]
```




    5




```python
arr[5:8]
```




    array([5, 6, 7])




```python
arr[5:8] = 12
arr
```




    array([ 0,  1,  2,  3,  4, 12, 12, 12,  8,  9])



A major distinciton between Python's lists and NumPy's arrays is that slices of an array are *views* on the original array.
Therefore, data is not copied, and any modifications will be reflected in the source array.


```python
arr_slice = arr[5:8]
arr_slice
```




    array([12, 12, 12])




```python
arr_slice[1] = 12345
arr
```




    array([    0,     1,     2,     3,     4,    12, 12345,    12,     8,
               9])



To get a copy of a ndarry, it must be explicitly copied.


```python
arr[5:8].copy()
```




    array([   12, 12345,    12])



Indexing multi-deminsional arrays requires the use of multidimensional indices.
Individual elements must be accessed recursively.


```python
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
arr2d[2]
```




    array([7, 8, 9])




```python
arr2d[0][2]
```




    3




```python
arr2d[0, 2]
```




    3




```python
arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
arr3d
```




    array([[[ 1,  2,  3],
            [ 4,  5,  6]],
    
           [[ 7,  8,  9],
            [10, 11, 12]]])




```python
arr3d[0]
```




    array([[1, 2, 3],
           [4, 5, 6]])




```python
old_values = arr3d[0].copy()
arr3d[0] = 42
arr3d
```




    array([[[42, 42, 42],
            [42, 42, 42]],
    
           [[ 7,  8,  9],
            [10, 11, 12]]])




```python
arr3d[0] = old_values
arr3d
```




    array([[[ 1,  2,  3],
            [ 4,  5,  6]],
    
           [[ 7,  8,  9],
            [10, 11, 12]]])




```python
arr3d[1, 0]
```




    array([7, 8, 9])




```python
arr3d[1][0]
```




    array([7, 8, 9])



We can still "slice" sections from a ndarry just like we would with a list.


```python
arr2d[:2, 1:]
```




    array([[2, 3],
           [5, 6]])




```python
arr2d[:, :1]
```




    array([[1],
           [4],
           [7]])




```python
arr2d[:2, 1:] = 0
arr2d
```




    array([[1, 0, 0],
           [4, 0, 0],
           [7, 8, 9]])



### Boolean indexing


```python
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7, 4)
names
```




    array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'], dtype='<U4')




```python
data
```




    array([[-1.08508534,  0.96194198, -1.70284703,  0.53831866],
           [-0.22939474,  0.17901741,  0.64462923,  0.95011127],
           [ 1.66067833,  1.54386301,  0.62022035, -0.98812697],
           [-0.64974777,  0.57832629, -0.51542191,  0.62006013],
           [ 1.09403373,  0.40614872, -1.30463546,  1.58902378],
           [ 0.52478652, -1.5343004 , -0.36373573, -1.04943203],
           [-1.26394603, -0.85122979, -2.21544374, -0.69594925]])




```python
names == 'Bob'
```




    array([ True, False, False,  True, False, False, False])




```python
data[names == 'Bob']
```




    array([[-1.08508534,  0.96194198, -1.70284703,  0.53831866],
           [-0.64974777,  0.57832629, -0.51542191,  0.62006013]])




```python
data[names != 'Bob']
```




    array([[-0.22939474,  0.17901741,  0.64462923,  0.95011127],
           [ 1.66067833,  1.54386301,  0.62022035, -0.98812697],
           [ 1.09403373,  0.40614872, -1.30463546,  1.58902378],
           [ 0.52478652, -1.5343004 , -0.36373573, -1.04943203],
           [-1.26394603, -0.85122979, -2.21544374, -0.69594925]])




```python
cond = names == 'Bob'
data[cond]
```




    array([[-1.08508534,  0.96194198, -1.70284703,  0.53831866],
           [-0.64974777,  0.57832629, -0.51542191,  0.62006013]])




```python
data[~cond]
```




    array([[-0.22939474,  0.17901741,  0.64462923,  0.95011127],
           [ 1.66067833,  1.54386301,  0.62022035, -0.98812697],
           [ 1.09403373,  0.40614872, -1.30463546,  1.58902378],
           [ 0.52478652, -1.5343004 , -0.36373573, -1.04943203],
           [-1.26394603, -0.85122979, -2.21544374, -0.69594925]])




```python
mask = (names == 'Bob') | (names == 'Will')
mask
```




    array([ True, False,  True,  True,  True, False, False])




```python
data[mask]
```




    array([[-1.08508534,  0.96194198, -1.70284703,  0.53831866],
           [ 1.66067833,  1.54386301,  0.62022035, -0.98812697],
           [-0.64974777,  0.57832629, -0.51542191,  0.62006013],
           [ 1.09403373,  0.40614872, -1.30463546,  1.58902378]])



### Fancy indexing

*Fancy indexing* (an actual term used for NumPy) is indexing using integer arrays.


```python
arr = np.empty((8, 4))
for i in range(8):
    arr[i] = i
arr
```




    array([[0., 0., 0., 0.],
           [1., 1., 1., 1.],
           [2., 2., 2., 2.],
           [3., 3., 3., 3.],
           [4., 4., 4., 4.],
           [5., 5., 5., 5.],
           [6., 6., 6., 6.],
           [7., 7., 7., 7.]])



To select out a subset or rows in a specific order, pass a list or ndarry of the indices.


```python
arr[[4, 3, 0, 6]]
```




    array([[4., 4., 4., 4.],
           [3., 3., 3., 3.],
           [0., 0., 0., 0.],
           [6., 6., 6., 6.]])



Using negative values indexes from the end.


```python
arr[[-3, -5, -7]]
```




    array([[5., 5., 5., 5.],
           [3., 3., 3., 3.],
           [1., 1., 1., 1.]])



Passing multiple indexing lists pulls the values corresponding to each tuple of the indices.


```python
arr = np.arange(32).reshape((8, 4))
arr
```




    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15],
           [16, 17, 18, 19],
           [20, 21, 22, 23],
           [24, 25, 26, 27],
           [28, 29, 30, 31]])




```python
arr[[1, 5, 7, 2], [0, 3, 1, 2]]
```




    array([ 4, 23, 29, 10])




```python
arr[[1, 5, 7, 2]][:, [0, 3, 1, 2]]
```




    array([[ 4,  7,  5,  6],
           [20, 23, 21, 22],
           [28, 31, 29, 30],
           [ 8, 11,  9, 10]])



### Transposing arrays and swapping axes

Transposing also just returns a *view* - *it does not change the underlying data.*
Arrays have a `transpose` method and a special `T` attribute.


```python
arr = np.arange(15).reshape((3, 5))
arr
```




    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14]])




```python
arr.T
```




    array([[ 0,  5, 10],
           [ 1,  6, 11],
           [ 2,  7, 12],
           [ 3,  8, 13],
           [ 4,  9, 14]])



For higher-dimensional arrays, `transpose` takes a tuple of the axis numbers to permute.


```python
arr = np.arange(16).reshape((2, 2, 4))
arr
```




    array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7]],
    
           [[ 8,  9, 10, 11],
            [12, 13, 14, 15]]])




```python
arr.transpose((1, 0, 2))
```




    array([[[ 0,  1,  2,  3],
            [ 8,  9, 10, 11]],
    
           [[ 4,  5,  6,  7],
            [12, 13, 14, 15]]])



The `T` method is a special case of the `swapaxes` method.


```python
arr.swapaxes(1, 2)
```




    array([[[ 0,  4],
            [ 1,  5],
            [ 2,  6],
            [ 3,  7]],
    
           [[ 8, 12],
            [ 9, 13],
            [10, 14],
            [11, 15]]])



## 4.2 Universal functions: fast element-wise array functions

A universal function, *ufunc*, performs element-wise operations on data in ndarrys.
They are vectorized.


```python
arr = np.arange(10)
arr
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
np.sqrt(arr)
```




    array([0.        , 1.        , 1.41421356, 1.73205081, 2.        ,
           2.23606798, 2.44948974, 2.64575131, 2.82842712, 3.        ])




```python
np.exp(arr)
```




    array([1.00000000e+00, 2.71828183e+00, 7.38905610e+00, 2.00855369e+01,
           5.45981500e+01, 1.48413159e+02, 4.03428793e+02, 1.09663316e+03,
           2.98095799e+03, 8.10308393e+03])



Other functions can take two arrays and operate on them value-wise.


```python
x = np.random.randn(8)
y = np.random.randn(8)
np.maximum(x, y)
```




    array([ 0.28729979,  0.1534349 ,  0.24391939, -0.49763428,  0.46122026,
            0.10359901,  0.01710523,  0.73134233])



Many *ufuncs* also take an optional `out` argument so they can operate in place.


```python
arr = np.random.randn(7) * 5
arr
```




    array([  1.39063097,  -9.34315368,   3.13845108, -15.70692903,
            -3.11904911,   6.44868051,   1.55349495])




```python
np.sqrt(arr)
```

    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/ipykernel_launcher.py:1: RuntimeWarning: invalid value encountered in sqrt
      """Entry point for launching an IPython kernel.





    array([1.17925017,        nan, 1.77156741,        nan,        nan,
           2.53942523, 1.24639278])




```python
arr
```




    array([  1.39063097,  -9.34315368,   3.13845108, -15.70692903,
            -3.11904911,   6.44868051,   1.55349495])




```python
np.sqrt(arr, arr)
```

    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/ipykernel_launcher.py:1: RuntimeWarning: invalid value encountered in sqrt
      """Entry point for launching an IPython kernel.





    array([1.17925017,        nan, 1.77156741,        nan,        nan,
           2.53942523, 1.24639278])




```python
arr
```




    array([1.17925017,        nan, 1.77156741,        nan,        nan,
           2.53942523, 1.24639278])



## 4.3 Array-oriented programming with arrays

