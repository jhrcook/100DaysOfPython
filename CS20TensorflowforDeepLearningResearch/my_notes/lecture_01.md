# Lecture 1. Introduction to TensorFlow


```python
import tensorflow as tf
import numpy as np
```

## Background

Tesnorflow separates the definition of computations from their execution (I think this is changing in 2.0 with the introduction of eager execution).
For now though, the graph is assembled and then a session executes the operations.

### What is a "tensor?"

A tensor is an n-dimensionsal array:

- 0-D tensor: scalar
- 1-D tensor: vector
- 2-D tensor: matrix

### A simple example

The following is a small example model of adding two tensors (scalar values 3 and 5).


```python
a = tf.add(3, 5)
print(a)
```

    tf.Tensor(8, shape=(), dtype=int32)


We get the value of `a` using the `numpy()` method.


```python
a.numpy()
```




    8



### A more complicated example


```python
# Input data.
x = 2
y = 3

# The TF graph.
op1 = tf.add(x, y)
op2 = tf.multiply(x, y)
op3 = tf.pow(op2, op1)

# Extract the value.
op3.numpy()
```




    7776



## Why does TF use graphs?

1. So TF only runs the necessary computations.
2. The computations can be broken up and run in parallel on distributed systems.
3. The graph can be broken into smaller, differentiable pieces.
4. Many ML models are naturally graphs, already.


```python

```
