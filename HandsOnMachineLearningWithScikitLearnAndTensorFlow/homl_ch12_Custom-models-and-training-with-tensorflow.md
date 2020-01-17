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


model.compile(loss=HuberLoss(2.0), optimizer='nadam')

model.save(model_path.as_posix())
```

However, there seems to currently be a bug in loading models with custom loss classes.
Thus, loading the model does not work.


```python
# model = keras.models.load_model(model_path.as_posix(),
#                                custom_objects={'HuberLoss': HuberLoss})
```

### Custom activation functions, initializers, regularizers, and constraints

A similar processes is required to create custom activation functions, initializers, regularizers, and constraints.
Generally, you just need to create a function with the correct inputs and outputs.
Here are some examples followed by a layer using them.


```python
def my_softplus_activation_fxn(z):
    return tf.math.log(tf.exp(z) + 1.0)


def my_glorot_initializer(shape, dtype=tf.float32):
    stddev = tf.sqrt(2.0 / (shape[0] + shape[1]))
    return tf.random.normal(shape, stddev=stddev, dtype=dtype)


def my_l1_regularizer(weights):
    return tf.reduce_sum(tf.abs(0.01 * weights))


def my_positive_weights_constraint(weights):
    return tf.where(weights < 0.0, tf.zeros_like(weights), weights)


keras.layers.Dense(units=30,
                   activation=my_softplus_activation_fxn,
                   kernel_initializer=my_glorot_initializer,
                   kernel_regularizer=my_l1_regularizer,
                   kernel_constraint=my_positive_weights_constraint)
```




    <tensorflow.python.keras.layers.core.Dense at 0x63fb5f9d0>



If additional hyperparameters must be retained by the saved model, then you will have to subclass the appropriate TF class like shown previously.
*Remember to implement the `get_config()` method.*

### Custom metrics

The most basic custom metric is a function that takes two parameters, `y_true` and `y_pred`, and computes the metric given that information.
This function is called at each training batch and TF keeps a running average of the results.

Sometimes, however, we want to retain the previous values and accumulate the results, ourselves.
This is coming for *streaming metrics* (or *stateful metrics*, those that are continuously updated over the training.
One example is precision.
To do this, we create a `keras.metrics.Precision` object and pass it the the real and predicted values.


```python
precision = keras.metrics.Precision()

# Mock two training steps.
precision([0, 1, 1, 1, 0, 1, 0, 1], [1, 1, 0, 1, 0, 1, 0, 1])
precision([0, 1, 0, 0, 1, 0, 1, 1], [1, 0, 1, 1, 0, 0, 0, 0])

precision.result()
```




    <tf.Tensor: id=737, shape=(), dtype=float32, numpy=0.5>




```python
precision.variables
```




    [<tf.Variable 'true_positives:0' shape=(1,) dtype=float32, numpy=array([4.], dtype=float32)>,
     <tf.Variable 'false_positives:0' shape=(1,) dtype=float32, numpy=array([4.], dtype=float32)>]



The values can be reset using the `reset_states()` method.


```python
precision.reset_states()
```

If you need to create a custom streaming method, subclass `keras.metrics.Metric`.
Below is an example of implementing a custom streaming metric that tracks the Huber loss and number of instances seen so far.
Further, when asked for a result, it returns the ratio.


```python
class HuberMetric(keras.metrics.Metric):
    def __init__(self, threshold=1.0, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.huber_fn = create_huber_loss_fxn(threshold)
        self.total = self.add_weight('total', initializer='zeros')
        self.count = self.add_weight('count', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        metric = self.huber_fn(y_true, y_pred)
        self.total.assign_add(tf.reduce_sum(metric))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))
    
    def result(self):
        return self.total / self.count
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, 'threshold': self.threshold}
```

Here are some notes on the above class:

* The constructor uses the `add_weight()` method to create the variables to keep track of the desired information over the training batches. Alternatively, you could just create a `tf.Variable` for each and TF will automatically remember it.
* The `update_state()` is called when the object of the class gets called as a function. It is provided the real and predicted labels and batch weights (ignored in the example).
* The `result()` method returns the desired result.
* The `get_config()` method helps to remember any hyperparameters, used here to remember the threshold for the Huber loss.
* There is also a `reset_states()` method that, by default, resets all variables, though it can be overriden if desired.

### Custom layers

There are two common circumstances under which a custom layer is desireable. The first is if you want to implement a new layer architecture that isn't available in Keras. The second is if you have a identical blocks of layers that you don't want to repeat over and over. For instance, if you wanted to have a pattern a layers $\text{ABCABCABC}$, then you could create a layer $\text{D = ABC}$ and make a network $\text{DDD}$, instead.
The following are several ways to create a custom layer.

Some layers do not have any weights such as `Flatten` or `ReLU`.
If you want a layer without any weights, the easiest option is to create a `Lambda` layer and and provide it a function to use to transform its inputs.
Here is an example of an exponential layer.


```python
exponential_layer = keras.layers.Lambda(lambda x: tf.exp(x))
```

To create a *stateful layer*, a layer with weights, create a subclass of the `Layer` class.
Below is an example reimplementing a Dense layer.


```python
class MyDense(keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)

    def build(self, batch_input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=[batch_input_shape[-1], self.units],
                                      initializer='glorot_normal')
        self.bias = self.add_weight(name='bias',
                                    shape=[self.units],
                                    initializer='zeros')
        super().build(batch_input_shape)

    def call(self, X):
        return self.activation(X @ self.kernel + self.bias)

    def compute_output_shape(self, batch_input_shape):
        return tf.TensorShape(batch_input_shape.as_list()[:-1] + [self.units])

    def get_config(self):
        base_config = super().get_config()
        return {**base_config,
                'units': self.units,
                'activation': keras.activations.serialize(self.activation)}
```

The following are notes on the above class:

* The constructor method takes all of the hyperparameters as arguments and creates the necessary attributes. The parent constructor takes care of the standard information and the rest is customizable.
* The `build()` method creates the layers variables by calling `add_weight()`. This will be called when the layer is first used because only then will it know the number of connections to make. *Remember to call the parent `build()` method at the end.*
* The `call()` method performs the desired neuron function. In this case, it is a simple linear formula passed to the activation function.
* The `compute_output_shape()` method returns a tensor of the shape of the outputs of the layer. This method can often be ignored in TF Keras because it is automatically inferred. This does not apply to dynamic layers.
* The `get_config()` method is used to help Keras store and retrieve custom hyperparameters.

If the layer takes multiple inputs, such as `Concatenate`, the arguments to `call()` and `compute_output_shape()` will be tuples of the inputs and input's batch shape, respectively.
Below is a toy example of a layer that takes two inputs and returns three outputs.


```python
class MyToyMultiLayer(keras.layers.Layer):
    def call(self, X):
        X1, X2 = X
        return [X1 + X2, X1 * X2, X1 / X2]
    
    def compute_output_shape(self, batch_input_shape):
        b1, b2 = batch_input_shape
        return [b1, b1, b1]
```

If the layer must behave differently between training and testing, then a `training` argument must be added to `call()`.
It will be passed a boolean or `None` to indicate the if the model is in training or not.


```python
class MyGaussianNoiseLayer(keras.layers.Layer):
    def __init__(self, stddev=1, **kwargs):
        super().__init__(**kwargs)
        self.stddev = stddev
    
    def call(self, X, training=None):
        if training:
            noise = tf.random.normal(tf.shape(X), stddev=self.stddev)
            return X + noise
        else:
            return X
        
    def compute_output_shape(self, batch_input_shape):
        return batch_input_shape
```

### Custom models

As an example of the flexibility and power provided by TF, we will create the following custom model:

1. A Dense input layer.
2. A Residual Block composed of two Dense layers and an addition operator. The input data flows through the two Dense layers and the input data is again added to the output data which is then fed back through the residual block. This procedure is repeated three times.
3. Another Residual Block.
4. A Dense output layer.

To begin, we will create a custom layer for the Residual Block.


```python
class ResidualBlock(keras.layers.Layer):
    def __init__(self, n_layers, n_neurons, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [keras.layers.Dense(n_neurons,
                                          activation='elu',
                                          kernel_initializer='he_normal')
                       for _ in range(n_layers)]

    def call(self, X, training=None):
        Z = X
        for layer in self.hidden:
            Z = layer(Z)
        return X + Z

    def get_config(self):
        base_config = super().get_config()
        return {**base_config,
                'n_layers': n_layers,
                'n_neurons': n_neurons}
```

Now we can use the Subclassing API to create the model.


```python
class ResidualRegressor(keras.Model):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden1 = keras.layers.Dense(30,
                                          activation='elu',
                                          kernel_initializer='he_normal')
        self.block1 = ResidualBlock(2, 30)
        self.block2 = ResidualBlock(2, 30)
        self.out = keras.layers.Dense(output_dim)

    def call(self, X):
        Z = self.hidden1(X)
        for _ in range(1 + 3):
            Z = self.block1(Z)
        Z = self.block2(Z)
        return self.out(Z)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config,
                'output_dim': output_dim}
```

The model can now be trained and used like any other.
The `get_config()` methods for `ResidualBlock` and `ResidualRegressor` were implemented, so the model could be saved and loaded.

### Losses and metrics based on model internals

Previously, we created custom loss functions that accept the predicted values, the known values, and a binary training argument.
However, there are times we want to use additional information about the model to compute a specific loss function.
For example, we can design a regression model with a *reconstruction loss* that computes the mean squared difference between the original inputs and the reconstructed outputs (we did a similar thing when studying decomposition and dimensionality reduction).
To apply the custom loss, the reconstruction loss is computed and added to the normal MSE loss using the `add_loss()` method within the `call()` method.


```python
class ReconstructionRegressor(keras.Model):
    def __init__(self, output_dim, with_reconstruction_loss=True, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [keras.layers.Dense(units=10, activation='relu')
                      for _ in range(5)]
        self.out = keras.layers.Dense(output_dim)
        self.with_reconstruction_loss=with_reconstruction_loss
    
    def build(self, batch_input_shape):
        n_inputs = batch_input_shape[-1]
        # An additional output layer for the reconstruction.
        self.reconstruct_layer = keras.layers.Dense(n_inputs)
        super().build(batch_input_shape)
    
    def call(self, X):
        Z = X
        for layer in self.hidden:
            Z = layer(Z)
        
        # Apply reconstruction loss.
        if self.with_reconstruction_loss:
            reconstruction = self.reconstruct_layer(Z)
            recon_loss = tf.reduce_mean(tf.square(reconstruction - X))
            self.add_loss(0.05 * recon_loss)
        
        return self.out(Z)
```

A custom metric is created in a similar fashion.
The result must be a `keras.metrics` object and added to the others using the `add_metric()` method.
A new copy of `ReconstructionRegressor` is created below with this feature.


```python
class ReconstructionRegressor(keras.Model):
    def __init__(self, output_dim, with_reconstruction_loss=True, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [keras.layers.Dense(units=10, activation='relu')
                      for _ in range(5)]
        self.out = keras.layers.Dense(output_dim)
        self.with_reconstruction_loss=with_reconstruction_loss

    def build(self, batch_input_shape):
        self.reconstruct_layer = keras.layers.Dense(batch_input_shape[-1])
        super().build(batch_input_shape)

    def call(self, X):
        Z = X
        for layer in self.hidden:
            Z = layer(Z)

        # Apply reconstruction loss.
        if self.with_reconstruction_loss:
            reconstruction = self.reconstruct_layer(Z)
            recon_loss = tf.reduce_mean(tf.square(reconstruction - X))
            self.add_loss(0.05 * recon_loss)
        
        # Report reconstruction loss as a metric.
        # m = keras.metrics.Mean(name='reconstruction_loss')
        # m.update_state(recon_loss)
        # self.add_metric(m, name='reconstruction_loss')

        return self.out(Z)
```


```python
from sklearn.datasets import make_swiss_roll
import mpl_toolkits.mplot3d.axes3d as p3

X, y = make_swiss_roll(3000, noise=0.3, random_state=0)

fig = plt.figure()
ax = p3.Axes3D(fig)
ax.view_init(7, -80)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y)
plt.title("Swiss Roll", fontsize=14)
plt.show()
```


![png](homl_ch12_Custom-models-and-training-with-tensorflow_files/homl_ch12_Custom-models-and-training-with-tensorflow_68_0.png)



```python
# Without reconstruction.
model = ReconstructionRegressor(1, with_reconstruction_loss=False)
model.compile(loss='mse', optimizer='nadam')
history_no_recon = model.fit(X, y, 
                             epochs=20, 
                             validation_split=0.2, 
                             verbose=0)


# With reconstruction.
model_recon = ReconstructionRegressor(1, with_reconstruction_loss=True)
model_recon.compile(loss='mse', optimizer='nadam')
history_recon = model_recon.fit(X, y, 
                                epochs=20, 
                                validation_split=0.2, 
                                verbose=0)
```

To demonstrate that the reconstruction loss had an effect on training, the learning curves are shown below.
The one with reconstruction loss (right) has a less steep descent than the one without this regularization (left).


```python
fig = plt.figure(figsize=(10, 10))

for i, history in enumerate((history_no_recon, history_recon)):
    plt.subplot(2, 2, i+1)
    df = pd.DataFrame(history.history)
    plt.plot('loss', 'b-', data=df, label='loss')
    plt.plot('val_loss', 'r-', data=df, label='val loss')
    plt.xlabel('epoch')
    plt.axis([0, 19, 0, 80])
    plt.title(['Without reconstruction loss', 'With reconstruction loss'][i],
              fontsize=14)
    plt.legend(loc='best')

for i, m in enumerate([model, model_recon]):
    ax = fig.add_subplot(2, 2, i+3, projection='3d')
    y_pred = m.predict(X)
    ax.view_init(7, -80)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y_pred.ravel())
    plt.title("Prediction", fontsize=14)

plt.show()
```


![png](homl_ch12_Custom-models-and-training-with-tensorflow_files/homl_ch12_Custom-models-and-training-with-tensorflow_71_0.png)


### Computing gradients using Autodiff

For this section, the following tow function will be used.

$
y = 3 w_1^2 + 2 w_1 w_2
$


```python
def  f(w1, w2):
    return 3 * w1**2 + 2 * w1 * w2
```

Using calculus, we can find  the partial derivatives w.r.t each variable, $w_1$ and $w_2$.

$
\frac{\delta w_1}{\delta y} = 6 w_1 +2 w_2 \qquad
\frac{\delta w_2}{\delta y} = 2 w_1
$

Thus, if ($w_1$, $w_2$) = (5, 3), then, using the partial derivatives, the gradient at that point would be (36, 10).
For a large ANN, finding the partial derivatives by hand would be an immense challenge.
One solution is to find the partial derivatives by slightly changing one parameter and measuring its effect on the output.


```python
w1, w2 = 5, 3
eps = 1e-6

# Estimated partial derivative of w1.
(f(w1+eps, w2) - f(w1, w2)) / eps
```




    36.000003007075065




```python
# Estimated partial derivative of w2.
(f(w1, w2+eps) - f(w1, w2)) / eps
```




    10.000000003174137



Though rather accurate, this would take too long for a large ANN.
Thus, we instead use autodiff.

First, two variables (`tf.Variable`) are defined
Then, a `tf.GradientTape` context is created.
It automatically records every operation that involves a variable.
With the `GradientTape` available, `z` is defined as a function of the two variables.
Finally, we ask the `GradientTape` object to find the derivatives.


```python
w1, w2 = tf.Variable(5.0), tf.Variable(3.0)

with tf.GradientTape()  as tape:
    z = f(w1, w2)

gradients = tape.gradient(z, [w1, w2])
gradients
```




    [<tf.Tensor: id=12552, shape=(), dtype=float32, numpy=36.0>,
     <tf.Tensor: id=12544, shape=(), dtype=float32, numpy=10.0>]



**To save memory, keep the bare minimum within the `tf.GradientTape()` block.**
It is possible to pause recording with `tape.stop_recording()`.

The tape is immediately erased after `gradient()` is called.
However, setting `persistent=True` in `tf.Gradient()` can prevent this - just make sure to erase the tape manually using `del tape`.


```python
w1, w2 = tf.Variable(5.0), tf.Variable(3.0)

with tf.GradientTape(persistent=True)  as tape:
    z = f(w1, w2)

print(tape.gradient(z, w1))
print(tape.gradient(z, w2))
del tape
```

    tf.Tensor(36.0, shape=(), dtype=float32)
    tf.Tensor(10.0, shape=(), dtype=float32)


It is possible to prevent backpropagation through a part of the function by "marking" it with `tf.stop_gradient()`.


```python
def  f2(w1, w2):
    return 3 * w1**2 + tf.stop_gradient(2 * w1 * w2)

with tf.GradientTape() as tape:
    z = f2(w1, w2)

tape.gradient(z, [w1, w2])
```




    [<tf.Tensor: id=12622, shape=(), dtype=float32, numpy=30.0>, None]



Sometimes there are numerical difficulties with autodiff.
For example, it has trouble finding the derivative of the softplus function.
Fortunately, we can calculate it analytically and provide the derivative function manually.
The function `my_better_softplus()`, decorated with `tf.custom_gradient` now  returns its normal output and a function used to compute its gradient provided the gradients so far passed down by backpropagation.


```python
@tf.custom_gradient
def my_better_softplus(z):
    exp = tf.exp(z)
    def my_softplus_gradients(grad):
        return grad / (1 + 1 / exp)
    return tf.math.log(exp + 1), my_softplus_gradients
```

### Custom training loops

One reason to create a custom training loop is to implement a custom or multiple training methods.
For example, the Wide and Deep model was trained by the original authors with two optimizers, one for each route.
However, for most instances, the default `fit()` method should be used as a custom training loop will be more error prone.

To provide and example of a custom training loop, we must first create some example data and define an example model.


```python
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

# Create mock regression data.
X, y = make_regression(n_samples=2000,
                       n_features=5,
                       n_informative=5,
                       n_targets=1,
                       noise=0,
                       shuffle=True,
                       random_state=0)

# Split training and testing data.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Scale the input.
std_scaler = StandardScaler()
X_train = std_scaler.fit_transform(X_train)
X_test = std_scaler.fit(X_test)

# PCA and plot of mock data.
pca = PCA(n_components=0.95, random_state=0)
tsne = TSNE(n_components=2, random_state=0)
X_reduced = pca.fit_transform(X_train)
X_reduced = tsne.fit_transform(X_reduced)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_train, alpha=0.3)
plt.title('PCA & t-SNE of mock regression data', fontsize=14)
plt.xlabel('$z_1$', fontsize=12)
plt.ylabel('$z_2$', fontsize=12)
plt.show()

# Construct model.
l2_reg = keras.regularizers.l2(l=0.05)
model = keras.models.Sequential([
    keras.layers.Dense(units=30,
                       activation='elu',
                       kernel_initializer='he_normal',
                       kernel_regularizer=l2_reg),
    keras.layers.Dense(units=1, kernel_regularizer=l2_reg)
])
```


![png](homl_ch12_Custom-models-and-training-with-tensorflow_files/homl_ch12_Custom-models-and-training-with-tensorflow_86_0.png)


Then we must create a function that randomly samples a batch of instances from the training set and a function that prints out the status of the training.


```python
def random_batch(X, y, batch_size=32):
    """Sample a random batch of training data."""
    idx = np.random.randint(len(X), size=batch_size)
    return X[idx], y[idx]


def print_training_status(iteration, total, loss, metrics):
    metrics = ' - '.join([f'{m.name}: {np.round(m.result(), 4)}' 
                          for m in [loss] + (metrics or [])])
    end = '' if iteration < total else '\n'
    print(f'\r{iteration}/{total} - ' + metrics, end=end)
```

Now we must define the hyperparameters for the trining and choose an optimizer, the loss function, and the metrics (here, MAE).


```python
n_epochs = 5
batch_size = 32
n_steps = len(X_train) // batch_size
optimizer = keras.optimizers.Nadam(learning_rate=0.01)
loss_fxn = keras.losses.mean_absolute_error
mean_loss = keras.metrics.Mean()
metrics = [keras.metrics.MeanAbsoluteError()]
```

Finally, we can write the custom loop.
A step-by-step explanation follows the code, below.


```python
for epoch in range(1, n_epochs + 1):
    print(f'Epoch {epoch} of {n_epochs}')
    for step in range(1, n_steps + 1):
        X_batch, y_batch = random_batch(X_train, y_train)
        with tf.GradientTape() as tape:
            y_pred = model(X_batch, training=True)
            main_loss = tf.reduce_mean(loss_fxn(y_batch, y_pred))
            loss = tf.add_n([main_loss] + model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        mean_loss(loss)
        for metric in metrics:
            metric(y_batch, y_pred)
        print_training_status(step * batch_size, len(y_train), mean_loss, metrics)
    print_training_status(len(y_train), len(y_train), mean_loss, metrics)
    for metric in [mean_loss] + metrics:
        metric.reset_states()
```

    Epoch 1 of 5
    WARNING:tensorflow:Layer sequential_1 is casting an input tensor from dtype float64 to the layer's dtype of float32, which is new behavior in TensorFlow 2.  The layer has dtype float32 because it's dtype defaults to floatx.
    
    If you intended to run this layer in float32, you can safely ignore this warning. If in doubt, this warning is likely only an issue if you are porting a TensorFlow 1.X model to TensorFlow 2.
    
    To change all layers to have dtype float64 by default, call `tf.keras.backend.set_floatx('float64')`. To change just this layer, pass dtype='float64' to the layer constructor. If you are the author of this layer, you can disable autocasting by passing autocast=False to the base Layer constructor.
    
    1500/1500 - mean: 81.21199798583984 - mean_absolute_error: 79.62300109863281
    Epoch 2 of 5
    1500/1500 - mean: 79.32990264892578 - mean_absolute_error: 78.76760101318366
    Epoch 3 of 5
    1500/1500 - mean: 81.2311019897461 - mean_absolute_error: 80.972702026367195
    Epoch 4 of 5
    1500/1500 - mean: 80.2770004272461 - mean_absolute_error: 80.036003112792972
    Epoch 5 of 5
    1500/1500 - mean: 77.1583023071289 - mean_absolute_error: 77.103797912597668


Below is a walk-through of the custom training loop:

* There are two nested loops, one for each epoch and one for the batches within the training epoch.
* A random batch sample is taken, `X_batch` and `y_batch`.
* Within the `GradientTape` block, a prediction is made by the model and used to compute the main loss using MAE, in this case. The total loss if found by adding up any other losses computed by the model; in this case, each layer has an $\ell_1$ regularization loss value.
* The gradients are computed from the `GradientTape` and passed to the optimizer to perform a gradient descent step.
* The mean loss and metrics for the epoch are updated and displayed.
* At the end of the epoch, the status bar is displayed again before reseting the metrics.

This training loop does not account for layers that behave differently between training and testing, such as `BatchNormalization` or `Dropout`.
This can be done by setting `training=True` and ensuring this message is propagated to the layers.

## TensorFlow functions and graphs

In TF 2, the graphs are not as central and are musch easier to use.
For an example, we start with a trivial function that returns the cube of its input.


```python
def cube(x):
    return x**3
```


```python
cube(2)
```




    8




```python
cube(tf.constant(2.0))
```




    <tf.Tensor: id=70592, shape=(), dtype=float32, numpy=8.0>



We then use `tf.function()` to turn `cube()` into a *TensorFlow Function*. 


```python
tf_cube = tf.function(cube)
tf_cube
```




    <tensorflow.python.eager.def_function.Function at 0x1a46626ed0>



It can still be used like normal, but will always return tensors.


```python
tf_cube(2)
```




    <tf.Tensor: id=70598, shape=(), dtype=int32, numpy=8>



Alternatively, we could have declared the function a TF Function at definition by including a decorator.


```python
@tf.function
def tf_cube(x):
    return x**3
```

The original function is still available, though.


```python
tf_cube.python_function(2)
```




    8



In the background, TF is computing the function graph and making optimizations eagerly.
It also can run the function faster by optimizing based on the graph's structure (e.g. parallel execution of independent computations).
For tensor inputs, a new graph is computed for each different shape and cached for later.
For normal Python inputs, a new graph is computed for every input, even if they are the same shape - thus, irresponsible use of TF Functions can take a lot of RAM.

---

The author continued with a good explanation of what TF is doing under the hood and provided some tips on not disrupting the process and getting good performance.
I choose not to copy this information here, though I took notes in the book.
