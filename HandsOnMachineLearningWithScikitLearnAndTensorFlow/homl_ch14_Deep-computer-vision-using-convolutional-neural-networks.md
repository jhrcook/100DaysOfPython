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
# %%cache -d caches ch14_example_CNN.pkl history

# history = model.fit(
#     X_train, y_train, 
#     epochs=5,
#     validation_data=(X_valid, y_valid),
#     callbacks=[
#         keras.callbacks.EarlyStopping(patience=5)
#     ]
# )
```

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

GoogLeNet won ImageNet in 2014.
Their success came mainly from using a deeper network with more efficient use of parameters.
By using *inception modules* (explained next) GoogLeNet actually had 10 times *fewer* parameters than AlexNet.

An inception module first copies the input into 4 and feeds it to 4 separate modules.
The first is a 1x1 + 1 (S) comvolutional layer, meaning it has a 1x1 kernel, a stride of 1, and uses "same" padding.
The second module is two convolutional layers, the first a 1x1 + 1 (S), but the second has a larger kernel, 3x3.
The third module again has two convolutional layers, but the second has a kernel of 5x5.
Finally, the fourth module has two layers, the first is a max pooling layer with a kernel of 3x3 and the second layer is a 1x1 + 1 (S) convolutional layer.
These four modules feed into a *depth concatenation layer* that stacks the feature maps from the last convolutional layer of each module.

The 1x1 kernels, though are unable to recognize spatial patterns, can capture patterns along the depth dimension.
Also, they output fewer feature maps than their inputs, so they are actually reducing dimensionality.
Finally, when a 1x1 layer is paired with a 3x3 or 5x5 layer, they act like one very powerful convolutional layer.
Instead of sweeping a simple linear classifier across the image, like a normal convolutional layer does, this pair of layers sweeps a two-layer NN across the image.

Basically, we can treat the inception module as a very powerful convolutional layer capable of recognizing complex patterns.

Finally, we can get to the GoogLeNet CCN architecture.
It is a very tall stack that I will not reproduce here.
The first 3rd looks like a normal CNN, then there are many inception modules with max pooling layers interspersed.
It is topped with a dense layer leading to a softmax output layer.

There were a few auxiliary classifiers places in the middle of the CNN to contribute to the overall loss as an attempt to mitigate vanishing gradients.
However, it was later demonstrated that these had little effect.

Google has since improved upon GoogLeNet with Inception-v3 and v4 with improvements to the inception modules.

### VGGNet

VGGGNet came in 2nd place in 2014 with a faily standard architecture.
The main architecture consisted of repeating blocks of 2 or 3 convolutional layers topped with a pooling layer.
It was topped with a dense network with 2 hidden layers and an output layer.
The convlutional layers used 3x3 kernels but had many filters.

### ResNet

ResNet won ImageNet in 2015 by following the trend of deeper networks with fewer parameters.
The key to training such a deep network was the use of *skip connections*, where the signal feeding into a layer is also added to the output of a layer located further along the stack.

The skip connection works by implementing *residual learning*.
Usually, a normal NN needs to model a target function $h(\textbf{x})$.
Adding $\textbf{x}$ to the output forces the model to learn $f(\textbf{x}) = h(\textbf{x}) - \textbf{x}$, instead.
Overall, it helps the network learn by propagating the original signal further into the network.
Thus, other layers can begin learning even if previous layers have not, yet.

ResNet looks very simillar to GoogLeNet at the beginning and end, but uses many convolutional layers with skip connections instead of inception layers.
Sometimes, the skip connection had to be fed through a simple convolutional layer in order to match the dimensions required by its destination.

Google's Inception-v4 merged the architecture ideas from GoogLeNet with ResNet and achieved still higher rates of top-five accuracy in ImageNet.

### Xception

The Xception archtiecture is considered a mixture of GoogLeNet and ResNet, too.
Instead of inception modules, though, Xception using *depthwise separable convolution layers* (or just *separable convolution layer* for short).
Each layer is composed of two sets of layers, the first is comprised of spatial-only filters whereas the second is comprised of depthwise-only filters.
This is simillar to an incpetion module except in the inception modules, the second set of convlutional layers were not restricted to depthwise-only filters.

The author notes that separable convolution layers generally perform better that normal convolutional layers and with fewer parameters.
Thus, he recommends considering them as the default option for a convolutional network, except after layers with few channels, such as the input layers.

### SENet

The Squeeze-and-Excitation Network (SENet) won ImageNet in 2017 with a 2.25% top-five error rate.
SENet used inception modules (GoogLeNet and residual units (ResNet), but boosted their performance by adding a small NN, an *SE-block*, to every unit in the original architecture.
Basically, for each module (either an inception module or a residual unit), the output was passed directly to the next layer and through an SE-block which passed its output to the next layer, too.
An SE-block only looks for associations in the dept dimension.
Thus, if two feature maps are commonly associated with one-another, then the SE-block "recalibrates" the output to accentuate one when the other is high.

Each SE-block has just 3 layers, a global average pooling layer, a hidden dense layer with a ReLU activation function, and an output dense layer with a sigmoid activation function.
If there are 256 feature maps passed to the global average pooling layer, it will output 256 values representing the response of each filter.
The hidden layer has much fewer neurons than the input, typically 16 times fewer, (thus the name "squeeze") and effectively is a lower-dimensionally embedding of the pooled feature maps.
The output layer again has the same number of neurons as the number of input feature maps so that it can provide a "recalibrated" value for each. 
Because it uses the sigmoid activation function, it outputs a value between 0 and 1 that can be multiplied against the output of each feature map from the original module.

## Implementing a ResNet-34 CNN using Keras

Normally, if you want to use a CNN, you should use transfer learning with a pretrained model.
However, for the sake of an example, below is the implementation of ResNet-34 (34 layers deep) using Keras.


```python
# A Redidual Unit layer in Keras.
class ResidualUnit(keras.layers.Layer):
    def __init__(self, filters, strides=1, activation='relu', **kwargs):
        super().__init__(**kwargs)

        self.activation = keras.activations.get(activation)

        self.main_layers = [
            keras.layers.Conv2D(filters, 3, strides=strides,
                                padding='same', use_bias=False),
            keras.layers.BatchNormalization(),
            self.activation,
            keras.layers.Conv2D(filters, 3, strides=1,
                                padding='same', use_bias=False),
            keras.layers.BatchNormalization()
        ]

        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                keras.layers.Conv2D(filters, 1, strides=strides,
                                    padding='same', use_bias=False),
                keras.layers.BatchNormalization()
            ]

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)

        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)

        return self.activation(Z + skip_Z)


# ResNet-34 CNN using Keras Sequential API.
model = keras.models.Sequential([
    keras.layers.Conv2D(64, 7, strides=2, input_shape=[224, 224, 3],
                        padding='same', use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')
])

prev_filters = 64
for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
    strides = 1 if filters == prev_filters else 2
    model.add(ResidualUnit(filters, strides=strides))
    prev_filters = filters

model.add(keras.layers.GlobalAvgPool2D())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(10, activation='softmax'))
```


```python
model.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_6 (Conv2D)            (None, 112, 112, 64)      9408      
    _________________________________________________________________
    batch_normalization (BatchNo (None, 112, 112, 64)      256       
    _________________________________________________________________
    activation (Activation)      (None, 112, 112, 64)      0         
    _________________________________________________________________
    max_pooling2d_4 (MaxPooling2 (None, 56, 56, 64)        0         
    _________________________________________________________________
    residual_unit (ResidualUnit) (None, 56, 56, 64)        74240     
    _________________________________________________________________
    residual_unit_1 (ResidualUni (None, 56, 56, 64)        74240     
    _________________________________________________________________
    residual_unit_2 (ResidualUni (None, 56, 56, 64)        74240     
    _________________________________________________________________
    residual_unit_3 (ResidualUni (None, 28, 28, 128)       230912    
    _________________________________________________________________
    residual_unit_4 (ResidualUni (None, 28, 28, 128)       295936    
    _________________________________________________________________
    residual_unit_5 (ResidualUni (None, 28, 28, 128)       295936    
    _________________________________________________________________
    residual_unit_6 (ResidualUni (None, 28, 28, 128)       295936    
    _________________________________________________________________
    residual_unit_7 (ResidualUni (None, 14, 14, 256)       920576    
    _________________________________________________________________
    residual_unit_8 (ResidualUni (None, 14, 14, 256)       1181696   
    _________________________________________________________________
    residual_unit_9 (ResidualUni (None, 14, 14, 256)       1181696   
    _________________________________________________________________
    residual_unit_10 (ResidualUn (None, 14, 14, 256)       1181696   
    _________________________________________________________________
    residual_unit_11 (ResidualUn (None, 14, 14, 256)       1181696   
    _________________________________________________________________
    residual_unit_12 (ResidualUn (None, 14, 14, 256)       1181696   
    _________________________________________________________________
    residual_unit_13 (ResidualUn (None, 7, 7, 512)         3676160   
    _________________________________________________________________
    residual_unit_14 (ResidualUn (None, 7, 7, 512)         4722688   
    _________________________________________________________________
    residual_unit_15 (ResidualUn (None, 7, 7, 512)         4722688   
    _________________________________________________________________
    global_average_pooling2d_1 ( (None, 512)               0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 512)               0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 10)                5130      
    =================================================================
    Total params: 21,306,826
    Trainable params: 21,289,802
    Non-trainable params: 17,024
    _________________________________________________________________


## Using pretrained models from Keras

It is usually best to use a pretrained model instead of training one personally for these massive CNNs.
Many pretrained models are available in `keras.applications`.
For example, ResNet-50 trained on ImageNet can be loaded in one line of code.


```python
model = keras.applications.resnet50.ResNet50(weights='imagenet')
```

This model expects an input image of 224 x 224.
The best way to resize an image is to use TF's `tf.image.resize()` or `resize_with_pad()` functions.


```python
test_image = np.array([
    load_sample_image("flower.jpg") / 255, 
    load_sample_image("china.jpg") / 255
])
test_image = tf.image.resize_with_pad(test_image, 224, 224)
```


```python
fig, axes = plt.subplots(1, 2, figsize=(8, 5))
for ax, img in zip(axes, test_image):
    plot_image(img, ax)
plt.show()
```


![png](homl_ch14_Deep-computer-vision-using-convolutional-neural-networks_files/homl_ch14_Deep-computer-vision-using-convolutional-neural-networks_23_0.png)


Each pretrained model expects the input to be preprocessed in a specific way (e.g. scaling from 0 to 1).
Thus, each model has a `preprocess_input()` method to handle this preparation step.
These functions assume pixel values from 0 to 255, so I multiplied by 255 below to undo the scaling to 0 to 1, done above.


```python
test_image = keras.applications.resnet50.preprocess_input(test_image * 255)
```

Finally, we can use the model to make predictions.
The output is a matrix of image x class (row x column).
The `decode_predictions()` method is useful to obtaining the top $k$ predictions.


```python
Y_proba = model.predict(test_image)
top_k = keras.applications.resnet50.decode_predictions(Y_proba, top=5)
for image_idx in range(len(test_image)):
    print(f'Image #{image_idx}:')
    for class_id, name, y_proba, in top_k[image_idx]:
        print(f'\t{class_id} - {name:12s} {y_proba * 100:.2f}%')
    print()
```

    Image #0:
    	n03530642 - honeycomb    37.23%
    	n11939491 - daisy        12.07%
    	n04522168 - vase         9.75%
    	n09229709 - bubble       4.29%
    	n07930864 - cup          3.45%
    
    Image #1:
    	n03028079 - church       23.95%
    	n02825657 - bell_cote    18.98%
    	n04346328 - stupa        10.08%
    	n03781244 - monastery    9.70%
    	n02980441 - castle       7.51%
    


The results were very impressive: the flower was correctly classified as a "daisy" as the second choice and the temple was correctly classified as a "monastery" in the fourth choice.

## Pretrained models for transfer learning

If you want a more specialized CNN, it is usually best to perform transfer learning with a pretrained model.
This will reduce the time and amount of training data required to achieve a high level of accuracy.

As an example, we will train a model to classify pictures of flowers, reusing a pretrained Xception model.
First, we must load the data set from TF.


```python
import tensorflow_datasets as tfds

dataset, info = tfds.load('tf_flowers', as_supervised=True, with_info=True)
```

    WARNING:absl:Warning: Setting shuffle_files=True because split=TRAIN and shuffle_files=None. This behavior will be deprecated on 2019-08-06, at which point shuffle_files=False will be the default for all splits.



```python
dataset_size = info.splits['train'].num_examples
dataset_size
```




    3670




```python
class_names = info.features['label'].names
class_names
```




    ['dandelion', 'daisy', 'tulips', 'sunflowers', 'roses']




```python
n_classes = info.features['label'].num_classes
n_classes
```




    5



We must first split the training data into train, validation, and testing datasets.


```python
test_split, valid_split, train_split = tfds.Split.TRAIN.subsplit([10, 15, 75])

test_set = tfds.load('tf_flowers', split=test_split, as_supervised=True)
valid_set = tfds.load('tf_flowers', split=valid_split, as_supervised=True)
train_set = tfds.load('tf_flowers', split=train_split, as_supervised=True)
```

The next step is to preprocess the images for the CNN.
It expects images to be 224 x 224, so the flowers must be resized.


```python
def preprocess_image(img, lbl):
    resized_img = tf.image.resize_with_pad(img, 224, 224)
    final_img  = keras.applications.xception.preprocess_input(resized_img)
    return final_img, lbl

batch_size = 32
train_set = train_set.shuffle(1000)
train_set = train_set.map(preprocess_image).batch(batch_size).prefetch(1)
valid_set = valid_set.map(preprocess_image).batch(batch_size).prefetch(1)
test_set = test_set.map(preprocess_image).batch(batch_size).prefetch(1)
```

Finally, we can load the model, without its top layer, add a new top layer, and train.
First, we must freeze the base model's layers.


```python
base_model = keras.applications.xception.Xception(weights='imagenet', 
                                                  include_top=False)
```


```python
for layer in base_model.layers:
    layer.trainable = False
```


```python
avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(n_classes, activation='softmax')(avg)
model = keras.Model(inputs=base_model.input, outputs=output)
```


```python
optimizer = keras.optimizers.Nadam(lr=0.2)
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=optimizer,
    metrics=['accuracy']
)
# history = model.fit(train_set, epochs=5, validation_data=valid_set)
```

It would take to long for me to actually run the re-training locally on my MacBook Pro.

After a few epochs, the accuracy should sit at 75% to 80%.
When that happens, the lower layers can be unfrozen and the training can continue.
*Don't forget to recompile the model after setting all layers to trainable.*


```python
for layer in base_model.layers:
    layer.trainable = True

optimizer = keras.optimizers.Nadam()
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=optimizer,
    metrics=['accuracy']
)
# history = model.fit(train_set, epochs=100, validation_data=valid_set)
```

## Classification and localization

Localization can be expressed as a regression task.
Four parameters must be predicted: the horizontal and vertical coordinates of the center and the height and width.
We can continue to use the same transfer learning process, just using a different output layer.
Here, we add a second dense output layer on the model with four output nodes and train using the MSE loss.


```python
base_model = keras.applications.xception.Xception(weights='imagenet',
                                                  include_top=False)
avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
class_output = keras.layers.Dense(n_classes, activation='softmax')(avg)
loc_output = keras.layers.Dense(4)(avg)

model = keras.Model(inputs = base_model.input,
                    outputs=[class_output, loc_output])

model.compile(loss=[keras.losses.SparseCategoricalCrossentropy(),
                    keras.losses.MeanSquaredError()],
              loss_weights=[0.8, 0.2],
              optimizer=keras.optimizers.Nadam(),
              metrics=['accuracy'])
```

Now, there is the issue of creating the training data for the bounding boxes.
This can either be done using a crowd-sourcing platform (e.g. AWS Mechanical Turks) or manually with the help of an open source tools (e.g. VGG Image  Annotator, LabelImg, etc.).
Once this data has been collected, the dataset must be fully assembled to be used in batches.
The dataset should be in the format `(images, (class_labels, bounding_boxes))`.
Once all of this has been accomplished, the model can be trained.

The most common cost function for bounding boxes is *Intersection over Union* (IoU) which is the area of the overlap of the predicted and target bounding boxes divided by the total area of their union.
In TF, this is implemented in `keras.metrics.MeanIoU`.

##  Object detection

The task of classifying and localizing multiple objects in an image is called *object detection*.

The original method for object detection was to break up the image into a grid and slide a CNN across it (more detail on this method is provided in the book, but not transcribed here).
Now, there is a faster and better approach called *fully convolutional network* (FCN).

### Fully convolutional networks

The idea for a FCN is that the final dense layer is exchanged with a convolutional layer with the number of filters equal to the number of neurons in the original output layer.
Now, the final layer can take any size image and output the same number of values.
If the image is larger, then the output comes in a grid with each cell recieving values for the desired outputs.
This occurs naturally based on the kernel size and stride of the output convolutional layer.
(This architecture is covered more fully and better in the book.)

The FCN is equivalent to taking a CNN and sliding it over a grid on the image, however, each image is only processed once, making this much more efficient than the older method.

### You Only Look Once (YOLO)

YOLO is an incredibly fast and accurate object detection architecture proposed in 2015, and updated in 2016 and 2018 with YOLOv2 and YOLOv3, respectively.
It is fast enough to run in real time on a video!

YOLOv3 has a few important distinctions:

1. It outputs 5 bounding boxes per grid cell, each one with an objectness score. It also outputs 20 class probabilities per grid cell, as it was trained on the PASCAL VOC dataset.
2. The center of the bounding boxes are expressed as offsets from (0, 0) in the top-left to (1, 1) in the bottom-right.
3. Before training YOLOv3, it first creates 5 "prototype" bounding boxes (using a KNN-based method based off of the training set's bounding box sizes), and then learns to rescale them during the training.

There are some models available in the TF Models project or TF Hub, but they evolve quickly as the field continually improves.

### Semantic segmentation

*Semantic segmentation* is the process of providing a class label to every pixel in an image.
The main difficulty is maintaining spatial resolution through a deep CNN where this is lost due to pooling layers and strides greater than 1.

One solution is to an *upsampling layer* that multiplies the resolution to regain the original dimensions.
One upsampling technique is a *transposed convolutional layer* that first stretches the image by inserting empty rows and columns filled with zeros.
This can be interpreted as using a fractional stride value, too.

Skip layers can also help with the loss of spatial resolution, namely that introduced by pooling layers.
Basically, the output of lower layers is added to the upsampled results of a later layer.

Finally, it is possible to scale an image beyond its original resolution using *super resolution* (but this topic was not discussed in this book).

There are plenty of models (some pretrained) avaialble on GitHub.
Also, *Mask R-CNN* is pretrained model in the TF Models project.
It is actually an *instance segmentation model* where each instance is kept separate instead of being lumped together with other instances (like a segmentation model would do).
It provides output of both bounding boxes with estimated class probabilities and a pixel mask that locates pixels in the bounding box that belong to each object.



```python
# Use this tutorial to help download and use the MobileNetV2 pre-trained model.
# https://www.tensorflow.org/tutorials/images/segmentation
```


```python

```
