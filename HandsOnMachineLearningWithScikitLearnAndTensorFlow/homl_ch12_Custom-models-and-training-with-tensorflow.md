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
import pathlib

# A path to the assets folder for this notebook.
assets_path = pathlib.Path("assets", "ch12")

# Set NumPy seed.
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




    <tf.Tensor: id=0, shape=(), dtype=int32, numpy=42>




```python
# 2x3 matrix
tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
```




    <tf.Tensor: id=1, shape=(2, 3), dtype=float32, numpy=
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




    <tf.Tensor: id=6, shape=(2, 1), dtype=float32, numpy=
    array([[1.],
           [4.]], dtype=float32)>




```python
t[..., 0:1]
```




    <tf.Tensor: id=10, shape=(2, 1), dtype=float32, numpy=
    array([[1.],
           [4.]], dtype=float32)>



There are many tensor mathematical operations available.


```python
# Addition.
t + 10
```




    <tf.Tensor: id=12, shape=(2, 3), dtype=float32, numpy=
    array([[11., 12., 13.],
           [14., 15., 16.]], dtype=float32)>




```python
# Square each element.
tf.square(t)
```




    <tf.Tensor: id=13, shape=(2, 3), dtype=float32, numpy=
    array([[ 1.,  4.,  9.],
           [16., 25., 36.]], dtype=float32)>




```python
# Matrix multiplication (`@` was new in Python 3.5).
t @ tf.transpose(t)
```




    <tf.Tensor: id=16, shape=(2, 2), dtype=float32, numpy=
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




    <tf.Tensor: id=17, shape=(3,), dtype=float32, numpy=array([2., 4., 5.], dtype=float32)>




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




    <tf.Tensor: id=20, shape=(2, 3), dtype=float32, numpy=
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




    <tf.Tensor: id=24, shape=(), dtype=float16, numpy=4.0>



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

Some other data types to be aware or are listed below with brief descriptions:

* **sparse tensors**: Tensors with many zeros. The `tf.sparse` library has many specific operations.
* **tensor arrays**: Lists of tensors of the same shape and data type.
* **ragged tensors**: List of lists of tensors of the same shape and data type. The `tf.ragged` library has many specific operations.
* **string tensors**: Regular tensors of type `tf.string`. The `tf.strings` library has many specific operations.
* **sets**: Regular or sparse tensors.  The `tf.sets` library has many specific operations.
* **queues**: A way of storing tensors across multiple steps. The `tf.queue` library has many specific operations.

## Customizing models and training algorithm

### Custom loss functions

In the following example, we implement the [Huber loss function]() (though it is available in already in `tf.keras`).
The first step is to define a function that takes the labels and predictions as arguments and use TF operations to compute every instance's loss.
The function should return a tensor with the loss for each instance.
It is important to use TF operations for two reasons: 1) they are vectorized, thus faster; 2) you still benefit from TF's graph optimization features.


```python
def huber_loss_fxn(y_true, y_pred):
    """The Huber loss function for TF."""
    error = y_true - y_pred
    is_small_error = tf.abs(error) < 1
    squared_loss = tf.square(error) / 2
    linear_loss = tf.abs(error) - 0.5
    return tf.where(is_small_error, squared_loss, linear_loss)


# Create a simple model.
model = keras.models.Sequential([
    keras.layers.InputLayer(5),
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(1)
])

# Compile the model for use with the custom loss function.
model.compile(loss=huber_loss_fxn, optimizer='nadam')
```

### Saving and loading models that constain custom components

Keras does not save the function when a model is saved, just the *name* of the function.
Thus a dictionary must be passed to map the name to the function.


```python
model_path = assets_path.joinpath("model_with_custom_loss.h5")
model.save(model_path.as_posix())

custom_objects = {'huber_loss_fxn': huber_loss_fxn}

model = keras.models.load_model(model_path.as_posix(),
                                custom_objects=custom_objects)
```

Let's say, however, you wanted to include another parameter in the loss function.
For example, the current Huber loss function has a range of -1 to 1 for "small", but we want to let this be adjusted.
One way is to create a function factory to return a Huber loss function with a different threshold.
This parameter, though, will not be saved with the model and, thus, must be supplied when loading the model.


```python
def create_huber_loss_fxn(threshold=1.0):
    def huber_loss_fxn(y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < threshold
        squared_loss = tf.square(error) / 2
        linear_loss = threshold * tf.abs(error) - threshold**2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)
    return huber_loss_fxn


model.compile(loss=create_huber_loss_fxn(2.0), optimizer='nadam')

model.save(model_path.as_posix())
```

When the model is loaded in, the Huber loss function must be re-created with the same threshold.


```python
custom_objects = {'huber_loss_fxn': create_huber_loss_fxn(2.0)}

model = keras.models.load_model(model_path.as_posix(),
                                custom_objects=custom_objects)
```

This implementation is non-ideal because it requires the user to remember and supply the threshold.
Thus, a better method is to subclass the `keras.losses.Loss` class and implement the `get_config()` method.
This method allows Keras to save and load the parameter to JSON with the rest of the model.


```python
class HuberLoss(keras.losses.Loss):
    def __init__(self, threshold=1.0, **kwargs):
        self.threshold = threshold
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < self.threshold
        squared_loss = tf.square(error) / 2
        linear_loss = self.threshold * tf.abs(error) - self.threshold**2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, 'threshold': self.threshold}


model.compile(loss=HuberLoss(), optimizer='nadam')

model.save(model_path.as_posix())
```


```python
custom_objects = {'HuberLoss': HuberLoss}

model = keras.models.load_model(model_path.as_posix(),
                               custom_objects={'HuberLoss': HuberLoss})
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-28-d2ecf3240698> in <module>
          2 
          3 model = keras.models.load_model(model_path.as_posix(),
    ----> 4                                custom_objects={'HuberLoss': HuberLoss})
    

    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/tensorflow_core/python/keras/saving/save.py in load_model(filepath, custom_objects, compile)
        144   if (h5py is not None and (
        145       isinstance(filepath, h5py.File) or h5py.is_hdf5(filepath))):
    --> 146     return hdf5_format.load_model_from_hdf5(filepath, custom_objects, compile)
        147 
        148   if isinstance(filepath, six.string_types):


    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/tensorflow_core/python/keras/saving/hdf5_format.py in load_model_from_hdf5(filepath, custom_objects, compile)
        182       # Compile model.
        183       model.compile(**saving_utils.compile_args_from_training_config(
    --> 184           training_config, custom_objects))
        185 
        186       # Set optimizer weights.


    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/tensorflow_core/python/keras/saving/saving_utils.py in compile_args_from_training_config(training_config, custom_objects)
        225   loss_config = training_config['loss']  # Deserialize loss class.
        226   if isinstance(loss_config, dict) and 'class_name' in loss_config:
    --> 227     loss_config = losses.get(loss_config)
        228   loss = nest.map_structure(
        229       lambda obj: custom_objects.get(obj, obj), loss_config)


    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/tensorflow_core/python/keras/losses.py in get(identifier)
       1183     return deserialize(identifier)
       1184   if isinstance(identifier, dict):
    -> 1185     return deserialize(identifier)
       1186   elif callable(identifier):
       1187     return identifier


    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/tensorflow_core/python/keras/losses.py in deserialize(name, custom_objects)
       1172       module_objects=globals(),
       1173       custom_objects=custom_objects,
    -> 1174       printable_module_name='loss function')
       1175 
       1176 


    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/tensorflow_core/python/keras/utils/generic_utils.py in deserialize_keras_object(identifier, module_objects, custom_objects, printable_module_name)
        178     config = identifier
        179     (cls, cls_config) = class_and_config_for_serialized_keras_object(
    --> 180         config, module_objects, custom_objects, printable_module_name)
        181 
        182     if hasattr(cls, 'from_config'):


    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/tensorflow_core/python/keras/utils/generic_utils.py in class_and_config_for_serialized_keras_object(config, module_objects, custom_objects, printable_module_name)
        163     cls = module_objects.get(class_name)
        164     if cls is None:
    --> 165       raise ValueError('Unknown ' + printable_module_name + ': ' + class_name)
        166   return (cls, config['config'])
        167 


    ValueError: Unknown loss function: HuberLoss



```python
model.loss_functions
```


```python

```
