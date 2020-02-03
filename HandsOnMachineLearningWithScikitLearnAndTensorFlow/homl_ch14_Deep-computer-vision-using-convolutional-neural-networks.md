# Deep Computer Vision Using Convolutional Neural Networks


```python
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow.keras as keras

np.random.seed(0)
sns.set_style('whitegrid')
%matplotlib inline
```

## Convolutional layers
Convolutional layers are not fully connected to the layers before nor after them.
Instead, each neuron connects to a specific "field" in the previous layer.
This architecture allows the network to conventratoin on small, low-level features in lower layers, and then construct more complex features in the higher layers.

Convolutional layers are arranged as 2D arrays of neurons, not 1D like we have used for NN, thus far.

A neuron located in row $i$, column $j$ of a layer is connected to the neurons in the previous layer in rows $i$ to $i + f_h - 1$, columns $j$ to $j + f_w - 1$, where $f_h$ and $f_w$ are the height and width of the receptive field of the neuron.
If two layers have the same dimensions, it is common to pad the edges with zeros so that each neuron has the same receptive field area.

### Filters

During training, a neuron may create a *filter* (or *convolutional kernel*).
This is where the neuron ignores (sets the weights to zero) parts of the receptive field.
They are usually created in lower levels and provide building blocks to the neurons downstream.
A layer of neurons all using the same filter outputs a *feature map*.

### Stacking multiple feature maps

One convolutional layer has multiple filters that each one output a feature map.
The layer has one neuron per pixel of the feature map and all of the neurons have the same parameters (weights and bias).
Each feature map uses a different sets of parameters.
A layer's neurons takes as input all of the feature maps of the previous layer in its receptive field.

> "The fact that all neurons in a feature map share the same parameters dramatically reduces the number of parameters in the model.
Once the CNN has learned to recognize a pattern in one location, it can recognize it in any other location.
In contrast, one a regular DNN has learned to recognize a pattern in one location, it can recognize it only in that particular location."

The output of a neuron is a weighted sum of its bias term and the values across all feature maps of the previous layer in its receptive field.

### TensorFlow implementation

Below are some dimensions to be aware of:

1. each image is a 3D tensor of shape $[height, width, channels]$
2. a mini-batch is represented as a 4D tensor of shape $[mini\text{-}batch size, height, width, channels]$
3. the weights of a convolutional layer are 4D tensors of shape $[f_h, f_w, f_{n'}, f_n]$, where $f_n$ is the number of feature maps of the layer and $f_{n'}$ is the $f_n$ of the previous layer.
4. the bias term is a 1D tensor of length $f_n$  (one per feature map)

Below is a simple example of loading two color images, applying a filter to each, and displaying one of the feature maps.


```python
from sklearn.datasets import load_sample_image

def plot_image(img, ax=None, cmap=None):
    if ax is None:
        ax = plt.gca()
    
    ax.imshow(img, cmap=cmap)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    return None


china = load_sample_image("china.jpg") / 255
flower = load_sample_image("flower.jpg") / 255

fig, axes = plt.subplots(1, 2, figsize=(10, 6))
plot_image(china, axes[0])
plot_image(flower, axes[1])
plt.show()
```


![png](homl_ch14_Deep-computer-vision-using-convolutional-neural-networks_files/homl_ch14_Deep-computer-vision-using-convolutional-neural-networks_3_0.png)



```python
# Combine the images into a batch and get dimensions.
images = np.array([china, flower])
batch_size, height, width, channels = images.shape

# Apply two filters, one with a vertical strip and one with a 
# horizontal strip.
filters = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
filters[:, 3, :, 0] = 1  # filter 1: vertical line
filters[3, :, :, 1] = 1  # filter 2: horizontal line

# Get the feature map based on these filters.
outputs = tf.nn.conv2d(images, filters, strides=1, padding='SAME')

# Plot the results.
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
ax_idx=1
for i in range(2):
    for j in range(2):
        plot_image(outputs[i, :, :, j], ax=axes[i, j], cmap='gray')
        ax_idx = ax_idx + 1
    
plt.show()
```


![png](homl_ch14_Deep-computer-vision-using-convolutional-neural-networks_files/homl_ch14_Deep-computer-vision-using-convolutional-neural-networks_4_0.png)


In the above code, we created two filters, one with a vertical strip and the other with a horizontal strip.
We used the low-level `tf.nn.conv2d()` function to apply the filters to the batch of two images.

Here we explicitly defined the filters, though for real CCNs, these can be left as trainable parameters.
This can be easily done by creating a `Conv2D` layer using Keras.
The example layer below has 32 filters, 3x3 each, with a stride of 1.
As you can see, there are many hyperparameters to tune, and this can be quite time-consuming.
We will learn about some successful architectures used before to build some base knowledge of what kind of values to use and adjust.


```python
conv = keras.layers.Conv2D(filters=32, 
                           kernel_size=3, 
                           strides=1, 
                           padding='same',
                           activation='relu')
```

### Memory requirements

CNNs require a lot of RAM, especially during training.
If there is not enough, you can try reducing the mini-batch size, removing layers, increasing the stride, using 16-bit floats (instead of the default 32 for TF).

## Pooling layers

The goal of a pooling layer is to *subsample* the input image in order to reduce the computational load, memory usage, and number of parameters in the model.

A pooling layer is simillar to a convolutional layer in that each neuron is only connected to a subset of the previous layer (the *receptive field*), however a pooling layer has no weights.
Instead it aggregates its receptive field using an aggregation function such as max or mean.
For example, the neurons of a *max pooling layer* take the value of the maximum value in their receptive field.
The other input values are dropped.
While this is obviously destructive, it also introduces some invariance to translation, rotation, or scaling of the image.
(See the book's figures 14-8 and 14-9 on pages 457 and 458, respectively, for some helpful visualizations.)

There are some downsides to pooling layers as some information is lost.
Further, they are not always used in CNNs, for example, when pixel-level accuracy is desired for the task at hand.

### TensorFlow implementation

The following are examples of max and average pooling layers in TF.


```python
max_pool = keras.layers.MaxPool2D(pool_size=2)
avg_pool = keras.layers.AvgPool2D(pool_size=2)
```

In practice, most CNNs now use max pooling instead of average pooling because, though average pooling preserves more information for the next layer, max pooling tends to preserve only the most important signal.
Also, it offers better translation invariance and requires less compute.

Max and average pooling can also be performed along the depth dimension, pooling across layers.
This is less common, though it can allow the CNN to learn multiple layers for the rotation (or some other variation) to a pattern in the image and then the depthwise pooling layer can select the appropriate one for a given input.

If you want to implement a depthwise max pooling layer, using the `tf.nn.max_pool()` object from TF (it is not available through Keras).

Lastly, there is a *global average pooling layer*. 
This computes the mean of each entire feature map, outputting just a single number per feature map per instance.
While this is quite destructive, it can be useful as the output layer.
We will see an example of this later in the chapter.
Below is an example of creating a global average pooling layer in Keras.


```python
global_avg_pool = keras.layers.GlobalAveragePooling2D()
```

## CNN Architectures

The general CNN architecture is to have a repeating pattern of a few convolutional layers, each followed by a ReLU layer, then a pooling layer.
After a few of these, there is a regular fully-connected feedforward nerual network at the top of the stack finished with an output layer (e.g. a softmax).

Do not use too large of convolutional kernels.
Use two layers of 3x3 instead of one 5x5.
The one exception to this is in the first layer where a 5x5 convolutional kernel with a stride of 2 helps to reduce the initial image to a more manageable size.

The author provided a model for the Fashion MNIST data set.
Below, I implement and train the model.
Some of the code is taken from the notebook on Chapter 10.


```python
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

# Split the training data into training and validation.
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full,
                                                      y_train_full,
                                                      test_size=0.2,
                                                      random_state=0)


class FashionImageFlatten(BaseEstimator, TransformerMixin):
    """Flatten the 28x28 MNIST Fashion image."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.reshape(len(X), 28*28)


class FashionImageReshape(BaseEstimator, TransformerMixin):
    """Reshape the 28x28 MNIST Fashion image."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.reshape(len(X), 28, 28, 1)


# A pipeline for pre-processing the MNIST Fashion data.
fashion_preprocesser = Pipeline([
    ('flattener', FashionImageFlatten()),
    ('minmax_scaler', MinMaxScaler()),
    ('reshaper', FashionImageReshape())
])

X_train = fashion_preprocesser.fit_transform(X_train)
X_valid = fashion_preprocesser.transform(X_valid)

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", 
               "Shirt", "Sneaker", "Bag", "Ankle boot"]

X_train.shape
```




    (48000, 28, 28, 1)




```python
fig = plt.figure(figsize=(12, 6))
for i in range(40):
    plt.subplot(4, 10, i+1)
    plt.imshow(X_train[i, :, :, 0], cmap='gray_r')
    plt.title(class_names[y_train[i]])
    plt.axis('off')

plt.show()
```


![png](homl_ch14_Deep-computer-vision-using-convolutional-neural-networks_files/homl_ch14_Deep-computer-vision-using-convolutional-neural-networks_13_0.png)



```python
model = keras.models.Sequential([
    keras.layers.Conv2D(filters=64, kernel_size=7,
                        activation='relu', padding='same', 
                        input_shape=[28, 28, 1]),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
    keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
    keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
    keras.layers.MaxPooling2D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Nadam(),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)
```


```python
# history = model.fit(
#     X_train, y_train, 
#     epochs=10,
#     validation_data=(X_valid, y_valid),
#     callbacks=[
#         keras.callbacks.EarlyStopping(patience=5)
#     ]
# )
```

    Train on 48000 samples, validate on 12000 samples
    Epoch 1/10
    38752/48000 [=======================>......] - ETA: 1:11 - loss: 0.8170 - accuracy: 0.7118WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-57-da195b4ef516> in <module>
          4     validation_data=(X_valid, y_valid),
          5     callbacks=[
    ----> 6         keras.callbacks.EarlyStopping(patience=5)
          7     ]
          8 )


    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/tensorflow_core/python/keras/engine/training.py in fit(self, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_freq, max_queue_size, workers, use_multiprocessing, **kwargs)
        726         max_queue_size=max_queue_size,
        727         workers=workers,
    --> 728         use_multiprocessing=use_multiprocessing)
        729 
        730   def evaluate(self,


    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/tensorflow_core/python/keras/engine/training_v2.py in fit(self, model, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_freq, **kwargs)
        322                 mode=ModeKeys.TRAIN,
        323                 training_context=training_context,
    --> 324                 total_epochs=epochs)
        325             cbks.make_logs(model, epoch_logs, training_result, ModeKeys.TRAIN)
        326 


    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/tensorflow_core/python/keras/engine/training_v2.py in run_one_epoch(model, iterator, execution_function, dataset_size, batch_size, strategy, steps_per_epoch, num_samples, mode, training_context, total_epochs)
        121         step=step, mode=mode, size=current_batch_size) as batch_logs:
        122       try:
    --> 123         batch_outs = execution_function(iterator)
        124       except (StopIteration, errors.OutOfRangeError):
        125         # TODO(kaftan): File bug about tf function and errors.OutOfRangeError?


    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/tensorflow_core/python/keras/engine/training_v2_utils.py in execution_function(input_fn)
         84     # `numpy` translates Tensors to values in Eager mode.
         85     return nest.map_structure(_non_none_constant_value,
    ---> 86                               distributed_function(input_fn))
         87 
         88   return execution_function


    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/tensorflow_core/python/eager/def_function.py in __call__(self, *args, **kwds)
        455 
        456     tracing_count = self._get_tracing_count()
    --> 457     result = self._call(*args, **kwds)
        458     if tracing_count == self._get_tracing_count():
        459       self._call_counter.called_without_tracing()


    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/tensorflow_core/python/eager/def_function.py in _call(self, *args, **kwds)
        485       # In this case we have created variables on the first call, so we run the
        486       # defunned version which is guaranteed to never create variables.
    --> 487       return self._stateless_fn(*args, **kwds)  # pylint: disable=not-callable
        488     elif self._stateful_fn is not None:
        489       # Release the lock early so that multiple threads can perform the call


    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/tensorflow_core/python/eager/function.py in __call__(self, *args, **kwargs)
       1821     """Calls a graph function specialized to the inputs."""
       1822     graph_function, args, kwargs = self._maybe_define_function(args, kwargs)
    -> 1823     return graph_function._filtered_call(args, kwargs)  # pylint: disable=protected-access
       1824 
       1825   @property


    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/tensorflow_core/python/eager/function.py in _filtered_call(self, args, kwargs)
       1139          if isinstance(t, (ops.Tensor,
       1140                            resource_variable_ops.BaseResourceVariable))),
    -> 1141         self.captured_inputs)
       1142 
       1143   def _call_flat(self, args, captured_inputs, cancellation_manager=None):


    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/tensorflow_core/python/eager/function.py in _call_flat(self, args, captured_inputs, cancellation_manager)
       1222     if executing_eagerly:
       1223       flat_outputs = forward_function.call(
    -> 1224           ctx, args, cancellation_manager=cancellation_manager)
       1225     else:
       1226       gradient_name = self._delayed_rewrite_functions.register()


    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/tensorflow_core/python/eager/function.py in call(self, ctx, args, cancellation_manager)
        509               inputs=args,
        510               attrs=("executor_type", executor_type, "config_proto", config),
    --> 511               ctx=ctx)
        512         else:
        513           outputs = execute.execute_with_cancellation(


    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/tensorflow_core/python/eager/execute.py in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
         59     tensors = pywrap_tensorflow.TFE_Py_Execute(ctx._handle, device_name,
         60                                                op_name, inputs, attrs,
    ---> 61                                                num_outputs)
         62   except core._NotOkStatusException as e:
         63     if name is not None:


    KeyboardInterrupt: 


### LeNet-5

The LeNet-5 architecture is one of the most widely known for CNNs.
It is shown in the following table.

| Layer | Type            | Maps | Size  | Kernel size | Stride | Activation |
|-------|-----------------|------|-------|-------------|--------|------------|
| Out   | fully-connected | -    | 10    | -           | -      | RBF        |
| F6    | fully-connected | -    | 84    | -           | -      | tanh       |
| C5    | convolution     | 120  | 1x1   | 5x5         | 1      | tanh       |
| S4    | avg. pooling    | 16   | 5x5   | 2x2         | 2      | tanh       |
| C3    | convolution     | 16   | 10x10 | 5x5         | 1      | tanh       |
| S2    | avg. pooling    | 6    | 14x14 | 2x2         | 2      | tanh       |
| C1    | convolution     | 6    | 28x28 | 5x5         | 1      | tanh       |
| In    | input           | 1    | 32x32 | -           | -      | -          |

The average pooling layers are slightly more complex than normal.
First, they multiply the mean they calculate by a trainable parameter (one per map) and they include a learnable bias term (one per map)

### AlexNet

AlexNet won ImageNet in 2012 by a large margin.
It is simillar to LeNet-5, but much larger and deeper.
It was the first to stack convolutional layers on top of each other without a pooling layer.
They used two regularization techniques during training: 50% dropout on the final dense layers and data augmentation (rotating, rescaling, changing the constrast and brightness, etc.).
They also used *local response normalization* (LRN) where the most strongly activated neurons inhibit other neurons located in the same position in the neighboring feature maps.
This encourages the feature maps to specialize and learn different patterns.
It is available in TF at `tf.nn.local_response_normalization()` and can be used in Keras by wrapping it in a `Lambda` layer.

### GoogLeNet




```python

```
