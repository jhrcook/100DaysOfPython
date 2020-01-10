# Chapter 12. Custom Models and Training with TensorFlow

Most of the time, the Keras API will be as far as a user needs to go into TensorFlow.
However, there will be times when we want to make custom pieces for our model and will be required to have additional control by using the rest of TensorFlow.
This is what is covered below.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras

np.random.seed(0)
```


```python
%matplotlib inline
```


```python
%load_ext ipycache
```

    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/IPython/config.py:13: ShimWarning: The `IPython.config` package has been deprecated since IPython 4.0. You should import from traitlets.config instead.
      "You should import from traitlets.config instead.", ShimWarning)
    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/ipycache.py:17: UserWarning: IPython.utils.traitlets has moved to a top-level traitlets package.
      from IPython.utils.traitlets import Unicode


## A quick tour of TensorFlow

The author provided a brief description of TensorFlow, elucidating its overall structure, and also pointing out some useful resources beyond the API (e.g. TensorFlow Hub and Extended (TFX)).

## Using TensorFlow like NumPy

TF is built around the flow of tensors from one operation to another.
Thus it is important to first understand how to use tensors before we can build custom pieces for TF.

### Tensors and operations

A tensor can be created using `tf.constant()`.
Below, a scalar is made and a 2x3 matrix is created.


```python
# Scalar
tf.constant(42)
```




    <tf.Tensor: id=65, shape=(), dtype=int32, numpy=42>




```python
# 2x3 matrix
tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
```




    <tf.Tensor: id=66, shape=(2, 3), dtype=float32, numpy=
    array([[1., 2., 3.],
           [4., 5., 6.]], dtype=float32)>




```python
t = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
t.shape
```




    TensorShape([2, 3])




```python
t.dtype
```




    tf.float32



Indexing in TF is similar to indexing in NumPy.


```python
t[:, :1]
```




    <tf.Tensor: id=71, shape=(2, 1), dtype=float32, numpy=
    array([[1.],
           [4.]], dtype=float32)>




```python
t[..., 0:1]
```




    <tf.Tensor: id=75, shape=(2, 1), dtype=float32, numpy=
    array([[1.],
           [4.]], dtype=float32)>



There are many tensor mathematical operations available.


```python
# Addition.
t + 10
```




    <tf.Tensor: id=77, shape=(2, 3), dtype=float32, numpy=
    array([[11., 12., 13.],
           [14., 15., 16.]], dtype=float32)>




```python
# Square each element.
tf.square(t)
```




    <tf.Tensor: id=78, shape=(2, 3), dtype=float32, numpy=
    array([[ 1.,  4.,  9.],
           [16., 25., 36.]], dtype=float32)>




```python
# Matrix multiplication (`@` was new in Python 3.5).
t @ tf.transpose(t)
```




    <tf.Tensor: id=81, shape=(2, 2), dtype=float32, numpy=
    array([[14., 32.],
           [32., 77.]], dtype=float32)>



Many of the function available from NumPy as available in TF, often with the same name.
However, there are times where the names differ, though it is usually because the operations are slightly different.

### Tensors and NumPy

Tensors "play nicely" with NumPy.
It is easy to convert between the two.


```python
# From NumPy to TF.
a = np.array([2, 4, 5], dtype=np.float32)
tf.constant(a)
```




    <tf.Tensor: id=82, shape=(3,), dtype=float32, numpy=array([2., 4., 5.], dtype=float32)>




```python
# From TF to NumPy.
t = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
t.numpy()
```




    array([[1., 2., 3.],
           [4., 5., 6.]], dtype=float32)




```python
# Addition of NumPy and TF.
a + t
```




    <tf.Tensor: id=91, shape=(2, 3), dtype=float32, numpy=
    array([[ 3.,  6.,  8.],
           [ 6.,  9., 11.]], dtype=float32)>



### Type conversions

TF does not perform automatic type casting!
This is true even for different floating point precisions.
Instead, it always raises an exception.


```python
# Throws an error.
# tf.constant(2.0, dtype=tf.float16) + tf.constant(2.0, dtype=tf.float32)

#> InvalidArgumentError: cannot compute AddV2 as input #1(zero-based) was 
#> expected to be a half tensor but is a float tensor [Op:AddV2] name: add/
```

The `tf.cast()` function can be used to change types.


```python
a = tf.constant(2.0, dtype=tf.float16)
b = tf.constant(2.0, dtype=tf.float32)
a + tf.cast(b, a.dtype)
```




    <tf.Tensor: id=101, shape=(), dtype=float16, numpy=4.0>



### Variables

TF's `Tensor` values are immutable.
The `Variable` type is mutable.
It behaves simillarly to `Tensor` with regards to mathematic operations and wokring with NumPy, though values can be assigned in place.


```python
v = tf.Variable([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
v
```




    <tf.Variable 'Variable:0' shape=(2, 3) dtype=float32, numpy=
    array([[1., 2., 3.],
           [4., 5., 6.]], dtype=float32)>




```python
v.assign(v * 2)
```




    <tf.Variable 'UnreadVariable' shape=(2, 3) dtype=float32, numpy=
    array([[ 2.,  4.,  6.],
           [ 8., 10., 12.]], dtype=float32)>




```python
v
```




    <tf.Variable 'Variable:0' shape=(2, 3) dtype=float32, numpy=
    array([[ 2.,  4.,  6.],
           [ 8., 10., 12.]], dtype=float32)>



Cells or slices can be assigned by calling the `assign()` method in the index.


```python
v[0, 1].assign(42)
```




    <tf.Variable 'UnreadVariable' shape=(2, 3) dtype=float32, numpy=
    array([[ 2., 42.,  6.],
           [ 8., 10., 12.]], dtype=float32)>




```python
v[:, 2].assign([-10.0, -20.0])
```




    <tf.Variable 'UnreadVariable' shape=(2, 3) dtype=float32, numpy=
    array([[  2.,  42., -10.],
           [  8.,  10., -20.]], dtype=float32)>



### Other data structures


```python

```
