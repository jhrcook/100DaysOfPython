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
**TODO**.


```python

```
