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

    CPU times: user 19.2 ms, sys: 13.3 ms, total: 32.5 ms
    Wall time: 31 ms



```python
%time for _ in range(10): my_list = [x * 2 for x in my_list]
```

    CPU times: user 908 ms, sys: 301 ms, total: 1.21 s
    Wall time: 1.47 s


## 4.1 The NumPy ndarray: a multidimensional array object

`ndarray` stands for N-dimensional array.
It is a fast flexible data structure with which we can conduct vectorized computations.
Here is a simple example.


```python
data = np.random.randn(2, 3)
data
```




    array([[-0.5091316 ,  0.50637148,  0.19441727],
           [-1.28432157,  0.72095519,  0.89429659]])




```python
data * 10
```




    array([[ -5.091316  ,   5.06371478,   1.94417272],
           [-12.84321566,   7.20955194,   8.94296593]])




```python
data + data
```




    array([[-1.0182632 ,  1.01274296,  0.38883454],
           [-2.56864313,  1.44191039,  1.78859319]])



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
