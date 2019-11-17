# Lecture 2. TensorFlow Ops




```python
import numpy as np
import pandas as pd
import tensorflow as tf

np.random.seed(0)
```

## Constants, sequences, variables, ops

It is very easy to make a constant in TF.


```python
# A 1D vector constant.
a = tf.constant([2, 2], name='vector')
a
```




    <tf.Tensor: id=598, shape=(2,), dtype=int32, numpy=array([2, 2], dtype=int32)>




```python
# A 2D matrix constant.
b = tf.constant([[0, 1], [2, 3]], name='matrix')
b
```




    <tf.Tensor: id=599, shape=(2, 2), dtype=int32, numpy=
    array([[0, 1],
           [2, 3]], dtype=int32)>



There are a lot of convience functions for creating data structures of different dimensions.


```python
# A 2x3 matrix of zeros.
tf.zeros([2, 3], tf.int32)
```




    <tf.Tensor: id=602, shape=(2, 3), dtype=int32, numpy=
    array([[0, 0, 0],
           [0, 0, 0]], dtype=int32)>




```python
# Make a zeros constant of the same shape as another object.
tf.zeros_like(b)
```




    <tf.Tensor: id=603, shape=(2, 2), dtype=int32, numpy=
    array([[0, 0],
           [0, 0]], dtype=int32)>




```python
# There is a simillar function for ones.
tf.ones([2, 3], tf.int32)
```




    <tf.Tensor: id=606, shape=(2, 3), dtype=int32, numpy=
    array([[1, 1, 1],
           [1, 1, 1]], dtype=int32)>




```python
tf.ones_like(b)
```




    <tf.Tensor: id=609, shape=(2, 2), dtype=int32, numpy=
    array([[1, 1],
           [1, 1]], dtype=int32)>




```python
# Make an n-array of any shape with a custom fill value.
tf.fill([2, 3], 8)
```




    <tf.Tensor: id=612, shape=(2, 3), dtype=int32, numpy=
    array([[8, 8, 8],
           [8, 8, 8]], dtype=int32)>




```python
# Make an array from a sequence.
tf.linspace(10.0, 13.0, 4, name='lin-space')
```




    <tf.Tensor: id=616, shape=(4,), dtype=float32, numpy=array([10., 11., 12., 13.], dtype=float32)>




```python
tf.range(start=3, limit=18, delta=3)
```




    <tf.Tensor: id=620, shape=(5,), dtype=int32, numpy=array([ 3,  6,  9, 12, 15], dtype=int32)>




```python
tf.range(start=3, limit=1, delta=-0.5)
```




    <tf.Tensor: id=626, shape=(4,), dtype=float32, numpy=array([3. , 2.5, 2. , 1.5], dtype=float32)>




```python
tf.range(5)
```




    <tf.Tensor: id=630, shape=(5,), dtype=int32, numpy=array([0, 1, 2, 3, 4], dtype=int32)>



There are various functions for creating random data: `random_normal()`, `truncated_normal()`, `​random_uniform()`, `​random_shuffle()`, `​random_crop()`, `​multinomial()`, `​random_gamma()`, and `​set_random_seed()`.


```python
tf.random.normal(shape=[10], mean=10, stddev=1.0, seed=0, name='normal')
```




    <tf.Tensor: id=636, shape=(10,), dtype=float32, numpy=
    array([ 9.439541 , 11.0533085,  8.805745 , 10.948302 ,  7.985449 ,
           10.12768  ,  9.485002 ,  9.848843 , 10.739277 , 10.438746 ],
          dtype=float32)>



## Math Operations

There are a bunch of various division functions.


```python
a = tf.constant([2, 2], name='a')
b = tf.constant([[0, 1], [2, 3]], name='b')

tf.math.divide(a, b)
```




    <tf.Tensor: id=641, shape=(2, 2), dtype=float64, numpy=
    array([[       inf, 2.        ],
           [1.        , 0.66666667]])>



## Variables

Differences between constants are variables:

1. A constant does not change.
2. A constant's value is stored *with* the model graph whereas a variable is stored separately.

### Creating variables

Note that the function for creating a variable is `tf.Variable()` with an uppercase "V" because it is a class.
The `tf.constant()` function uses a lowercase "c" because it is an ops, not a class.

At the time of writting this course, TF encouraged developers to use the wrapper `get_variable()` to create a variable.
Here is the general structure:

```python
tf.get_variable(
    name,
    shape=None,
    dtype=None,
    initializer=None,
    regularizer=None,
    trainable=None,
    collections=None,
    caching_device=None,
    partitioner=None,
    validate_shape=True,
    use_resource=None,
    custom_getter=None,
    constraint=None
)
```

However, as of TF 2.0, they now suggest using `tf.Variable()`, directly.
Creating a variable is shown below.


```python
s = tf.Variable(tf.constant(2), 'scalar')
s
```




    <tf.Variable 'Variable:0' shape=() dtype=int32, numpy=2>




```python
m = tf.Variable([[0, 1], [2, 3]], 'matrix')
m
```




    <tf.Variable 'Variable:0' shape=(2, 2) dtype=int32, numpy=
    array([[0, 1],
           [2, 3]], dtype=int32)>




```python
W = tf.Variable(tf.zeros([10, 10]), 'big_matrix')
W
```




    <tf.Variable 'Variable:0' shape=(10, 10) dtype=float32, numpy=
    array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]], dtype=float32)>




```python
W.eval
```




    <bound method BaseResourceVariable.eval of <tf.Variable 'Variable:0' shape=(10, 10) dtype=float32, numpy=
    array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]], dtype=float32)>>



A variable can be assigned a value using the `assign()` method.


```python
W.assign(tf.random.normal([10, 10]))
W
```




    <tf.Variable 'Variable:0' shape=(10, 10) dtype=float32, numpy=
    array([[ 0.02936255, -0.86747646, -0.8452195 , -0.26652253,  0.11087855,
             0.41120422,  0.92334676, -0.33460006, -0.85756123,  0.8442435 ],
           [ 0.26037455, -0.17688881,  0.1632508 ,  0.44674233,  1.3234712 ,
             1.3052758 , -0.8651197 ,  1.2339976 , -1.3227814 ,  0.9247735 ],
           [-0.41094947,  0.7952504 , -1.7627925 , -1.2065858 ,  1.0230845 ,
            -0.22552626, -1.1214129 ,  0.74185956,  1.1253291 ,  1.1326234 ],
           [ 1.3066211 , -0.33423454,  0.9824326 ,  1.1420981 , -0.39754665,
             0.99293154,  0.24455707, -0.7633621 , -0.12773935,  0.32760194],
           [ 1.4027779 ,  0.36471343, -1.790517  ,  0.34080747, -1.1352329 ,
             0.40419906, -1.4473691 ,  0.8261736 , -2.2291663 ,  0.85217124],
           [ 1.774641  , -0.20230924, -0.44129497,  0.6964944 , -0.926713  ,
             0.3398413 ,  0.8431046 ,  0.5213242 ,  0.3134696 , -0.16407096],
           [ 2.230049  , -1.863732  , -0.3128602 , -0.31636953,  1.0183464 ,
            -0.12306501, -0.05538774,  0.6176986 ,  1.6328528 ,  1.7787921 ],
           [-0.76783967,  0.57597667,  1.2463351 , -0.03905262,  1.3955286 ,
            -0.49600318,  0.17775223,  0.8906394 ,  0.53174686,  1.412703  ],
           [-0.6675612 , -0.62404037,  1.333639  , -0.54178673, -0.13521735,
            -1.2285764 ,  0.90140593, -0.25594923,  0.43793267, -1.7960231 ],
           [ 1.3177958 , -1.0183132 , -1.1624463 , -0.01659734, -2.2292974 ,
            -0.0754303 , -0.7370742 , -2.0015604 , -0.13629597, -0.85789424]],
          dtype=float32)>




```python
a = tf.Variable(2, 'scalar')
a
```




    <tf.Variable 'Variable:0' shape=() dtype=int32, numpy=2>




```python
a.assign(a * 2)
```




    <tf.Variable 'UnreadVariable' shape=() dtype=int32, numpy=4>




```python
a.assign(a * 2)
```




    <tf.Variable 'UnreadVariable' shape=() dtype=int32, numpy=8>




```python
a.assign_add(2)
```




    <tf.Variable 'UnreadVariable' shape=() dtype=int32, numpy=10>




```python
a.assign_sub(3)
```




    <tf.Variable 'UnreadVariable' shape=() dtype=int32, numpy=7>


