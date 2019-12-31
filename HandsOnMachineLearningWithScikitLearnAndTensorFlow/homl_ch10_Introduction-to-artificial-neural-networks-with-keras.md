# Chapter 10. Introduction to Artificial Neural Networks with Keras


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.core.interactiveshell import InteractiveShell

np.random.seed(0)

plt.style.use('seaborn-whitegrid')
```


```python
%matplotlib inline
```


```python
%load_ext ipycache
```

    The ipycache extension is already loaded. To reload it, use:
      %reload_ext ipycache


## From biological to artificial neurons

The author briefly described the origins of the field of artificial neural networks (ANN) and how real neurons work; I will skip recounting that section here.

### Logical computations with neurons

The first artificial neuron was proposed as a function with one or more binary inputs and one binary output.
The output is activated when a certain number of inputs are activated.
Several of these neurons working together can reproduce standard logic gates.

### The Perceptron

The *perceptron* was invented in 1957 by Frank Rosenblatt.
It is *threshold logic unit* (TLU) (or a *linear threshold unit*, LTU).
Each input and the  output are numeric values (not binary) and each input has an associated weight.
The TLU computes the weighted sum of the inputs ($z = w_1 x_1 + w_2 x_2 + ... + w_n x_n = \textbf{x}^T \textbf{w}$) and applies a step function to the result to decide the output ($h_w(\textbf{x}) = \text{step}(z)$).
The most common step functions were the *Heaviside step function* and the *sign function*:

$$
\text{heaviside} (z) =
\begin{cases}
    0 \text{ if } z  <  0 \\ 
    1 \text{ if } z \ge 0 \\
\end{cases}
$$

$$
\text{sgn}(z) =
\begin{cases}
    -1 \text{if } z < 0 \\ 
    0  \text{if } z = 0 \\
    1  \text{if } z > 0 \\
\end{cases}
$$

A single TLU can be used for linear classification.
Training the TLU means finding the values for the weights ($w_1, w_2, ..., w_n = \textbf{w}$)

A perceptron is a single layer of TLUs with each TLU connected to all of the input neurons (creating a *fully connected layer*).
The input neurons are just simple passthrough functions where the output equals the input.
An extra *bias neuron* is usually added that always has the value 1 and is connected to each neuron; this helps with the linear algebra for calculating the output of the perceptron:

$$
h_{\textbf{W}, \textbf{b}} = \phi(\textbf{XW} + \textbf{b}) \\
$$

$$
\text{where }
\begin{cases}
    \textbf{X}: \text{represents the matrix of input features; one row per instance, one column per feature} \\
    \textbf{W}: \text{represents the weight matrix from the input neurons to the TLU neurons} \\
    \textbf{b}: \text{represents the weight vector from the bias neurons to the TLU neurons} \\
    \phi: \text{represents the activation function; the step function for TLUs} \\
\end{cases}
$$

The perceptron is trained using *Hebb's Rule*: the connection weight between two neurons tends to increase when they fire simultaneously.
The algorithm also accounts for the accuracy of the prediction; for each training instance, the weights are updated with the following *Perceptron learning rule*:

$$
w_{i,j}^{\text{next step}} = w_{i,j} + \eta (y_j - \hat{y_j}) x_i
$$

$$
\text{where }
\begin{cases}
    w_{i,j}: \text{the connection weight between the } i^{th} \text{ input neuron and } j^{th} \text{output neuron} \\
    x_i: \text{the } i^{th} \text{ input value of the training instance} \\
    \hat{y_j}: \text{the output of the } j^{th} \text{ output neuron for the training instance} \\
    y_j: \text{the target output of the } j^{th} \text{ output neuron for the training instance} \\
    \eta: \text{the learning rate} \\
\end{cases}
$$

The decision boundary for each output neuron is linear, so perceptrons are only capable of linear separation.
Though, when this is possible for the data, the perceptron will converge.

There are some critical limitations to a perceptron, notably that it cannot learn XOR logic, but many of these can be overcome by using multiple layers to create a *Multilayer Perceptron* (MLP).


```python
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

# Load the iris data and only use petal length and width (cm.).
iris = load_iris()
X = iris.data[:, (2, 3)]
y = iris.target

# Train a perceptron.
per_clf = Perceptron()
per_clf.fit(X, y)

# Plot the prediction results.
plt.scatter(X[:, 0], X[:, 1], c=per_clf.predict(X), cmap='Set1')
plt.title('Perceptron classifying the iris data', fontsize=14)
plt.xlabel(iris.feature_names[2], fontsize=12)
plt.ylabel(iris.feature_names[3], fontsize=12)
plt.show()
```


![png](homl_ch10_Introduction-to-artificial-neural-networks-with-keras_files/homl_ch10_Introduction-to-artificial-neural-networks-with-keras_5_0.png)


## The multilayer perceptron (MLP) and backpropagation

An MLP has one *input layer* built of passthrough neurons, one or more *hidden layers* of TLUs, and one *ouput layer* of TLUs. 
The layers near the input are called *lower layers* whereas the layers near the output are *upper layers*.
When an ANN contains a deep stack of hidden layers, it is called a *deep neural network* (DNN).

*Backpropagation* makes training a MLP possible.
It is able to compute the gradient of the nwtwork's error with regard to every single model parameter.
Therefore, the connection weights and bias terms can be tweaked optimally to reduce error.
Once the gradient is found, normal gradient descent can be followed.
Below is the most basic interpretation of the algorithm:

1. A mini-batch is fed into the input layers.
2. The data progresses through the layers and the results of each layer are remembered. This is the *forward pass*.
3. A final output is calculated and a loss function is used to determine the error.
4. The chain rule is used to measure how much each output connected contributed to the error.
5. The error is *backpropagated* using the chain rule to measure how much of the error came from each connection in the network.
6. The error gradient is used to compute the optimal adjustments for each connection weight.

In order to compute the derivative of each neuron, the step function had to be replaced.
Currently, it is common to use the logistic equation, hyperbolic tangent function, or rectified linear unit (ReLU):

$$
\begin{aligned}
\text{logistic: }& \quad \sigma(z) = 1 / (1 + e^{-z}) \\
\text{hyperbolic tangent: }& \quad \tanh(z) = 2 \sigma(2z) - 1 \\
\text{ReLU: }& \quad \text{ReLU}(z) = \max(0, z) \\
\end{aligned}
$$


```python
# Input points
x = np.linspace(-4, 4, 300)

# Neruon activation functions.

def step_fxn(x):
    """Step function for neuron activation function."""
    return [-1 if i < 0 else 1 for i in x]


def step_fxn_d(x):
    """Derivative of the step function."""
    return np.zeros(len(x))


def logistic_fxn(x):
    """Logisitc (sigmoid) function for neuron activation function."""
    return 1 / (1 + np.exp(-x))


def logistic_fxn_d(x):
    """The derivative of the logistic function."""
    return logistic_fxn(x) * (1 - logistic_fxn(x))


def hyperbolic_tan(x):
    """Hyperbolic tangent function for neuron activation function."""
    return 2 * logistic_fxn(2 * np.array(x)) - 1


def hyperbolic_tan_d(x):
    """The derivative of the hyperbolic tangent function."""
    return 1 - (np.tanh(x) ** 2)


def relu_fxn(x):
    """The ReLU for a neuron activation function."""
    return [max(0, i) for i in x]


def relu_fxn_d(x):
    """The derivative of the ReLU"""
    return [0 if i < 0 else 1 for i in x]


# Styles of line.
line_styles = {
    'step': 'r--',
    'logistic': 'b--',
    'tanh': 'g--',
    'relu': 'y--'
}

# Plotting each activation function.
fig = plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(x, step_fxn(x), line_styles['step'], label='step')
plt.plot(x, logistic_fxn(x), line_styles['logistic'], label='logistic')
plt.plot(x, hyperbolic_tan(x), line_styles['tanh'], label='tanh')
plt.plot(x, relu_fxn(x), line_styles['relu'], label='ReLU')
plt.axis([-4, 4, -1.1, 1.1])
plt.legend(loc='center right', fontsize=12)
plt.xlabel('$x$', fontsize=12)
plt.ylabel('$y$', fontsize=12)
plt.title('Activation Functions', fontsize=14)

# Plotting the derivatives of the activation function.
plt.subplot(1, 2, 2)
plt.plot(x, step_fxn_d(x), line_styles['step'], label='step')
plt.plot(x, logistic_fxn_d(x), line_styles['logistic'], label='logistic')
plt.plot(x, hyperbolic_tan_d(x), line_styles['tanh'], label='tanh')
plt.plot(x, relu_fxn_d(x), line_styles['relu'], label='ReLU')
plt.axis([-4, 4, -0.1, 1.1])
plt.legend(loc='center right', fontsize=12)
plt.xlabel('$x$', fontsize=12)
plt.ylabel('$y$', fontsize=12)
plt.title('Derivatives of Activation Functions', fontsize=14)

plt.show()
```


![png](homl_ch10_Introduction-to-artificial-neural-networks-with-keras_files/homl_ch10_Introduction-to-artificial-neural-networks-with-keras_7_0.png)


### Regression MLP

For a regression MLP, there must be one output neuron per output variable.
The neuron's activation function must be chosen depending on any constraints on the output (e.g. always positive, between 0 and 1) and no activation function can be used to leave the output value unconstrained.
The loss function most  commonly used in the MSE, though the mean absolute error (MAE) or Huber loss (a mixture of MSE and MAE) can be used if the training data has a lot of outliers.

### Classification MLP

If the decision is a binary classification, then only one neuron is required with a logistic activation function.
If there are multiple decisions, but each is binary, then there must be a neuron per decision, each with a logistic activation function.
If there is a single decision, but multiple possibilities (i.e. the image of 1 of 10 digits), then there must be an output neuron for each option with a *softmax function* across the whole output layer.
The softmax function ensures that each neuron is between 0 and 1 (inclusive) and that they sum to 1.
The loss function is usually the cross-entropy loss function.

---

>**Before moving on to using Keras to build ANNs, I followed the advice of the author and skipped to Exercise 1 for the chapter which was to experiment in the [TensorFlow Playground](https://playground.tensorflow.org/).**

---

## Implementing MLPs with Keras




```python
import tensorflow as tf
from tensorflow import keras

tf.__version__
```




    '2.0.0'




```python
keras.__version__
```




    '2.2.4-tf'



### Building an image classifier using the sequential API

For the first example, we will build an image classifier of the MNIST Fashion data set.
It contains 70,000 28x28 grayscale images of clothing of 10 classes.

Keras provides some functions for accessing commonly used data.


```python
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
```

Each image is a 28x28 array.


```python
X_train_full.shape
```




    (60000, 28, 28)



Each data point is an integer between 0 and 255.


```python
X_train_full.dtype
```




    dtype('uint8')



We need to make a validation set from the training data and scale the data to between 0 and 1.


```python
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin


# Split the training data into training and validation.
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full,
                                                      y_train_full,
                                                      test_size=5000,
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
        return X.reshape(len(X), 28, 28)


# A pipeline for pre-processing the MNIST Fashion data.
fashion_preprocesser = Pipeline([
    ('flattener', FashionImageFlatten()),
    ('minmax_scaler', MinMaxScaler()),
    ('reshaper', FashionImageReshape())
])
fashion_preprocesser.fit(X_train)

X_train = fashion_preprocesser.transform(X_train)
X_valid = fashion_preprocesser.transform(X_valid)
```

The names of the classes was not provided, so I had to enter them manually.
I found the known values in the [Keras documentation](https://keras.io/datasets/#fashion-mnist-database-of-fashion-articles).


```python
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", 
               "Shirt", "Sneaker", "Bag", "Ankle boot"]
```


```python
fig = plt.figure(figsize=(12, 6))
for i in range(40):
    plt.subplot(4, 10, i+1)
    plt.imshow(X_train[i, :, :])
    plt.title(class_names[y_train[i]])
    plt.axis('off')

plt.show()
```


![png](homl_ch10_Introduction-to-artificial-neural-networks-with-keras_files/homl_ch10_Introduction-to-artificial-neural-networks-with-keras_21_0.png)


The sequential API is used below to create the first neural network with two hidden layers.


```python
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation='relu', name='dense_1'))
model.add(keras.layers.Dense(100, activation='relu', name='dense_2'))
model.add(keras.layers.Dense(10, activation='softmax', name='dense_3'))
```

Here is an explanation of each line:

1. A sequential model is initialized. It is just a single stack of layers connected sequentially.
2. The first layer is the input nodes that just flattens the 28x28 image into a single array. Since it is the first layer, the shape of the input must be explicitly stated.
3. The first hidden layer is a dense network of 300 neurons each with a ReLU activation function.
4. The second hidden layer is a dense network of 100 neurons each with a ReLU activation function.
5. The output layer is a dense network of 10 neurons using a softmax activation function.

Alternatively, the model could have been declared directly when initializing the Sequential model by passing a list of the layers.
The model can be inspected with the `summary()` method.


```python
model.summary()
```

    Model: "sequential_5"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    flatten_4 (Flatten)          (None, 784)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 300)               235500    
    _________________________________________________________________
    dense_2 (Dense)              (None, 100)               30100     
    _________________________________________________________________
    dense_3 (Dense)              (None, 10)                1010      
    =================================================================
    Total params: 266,610
    Trainable params: 266,610
    Non-trainable params: 0
    _________________________________________________________________


The list of layers can be accessed and specifically indexed by name.


```python
model.layers
```




    [<tensorflow.python.keras.layers.core.Flatten at 0x1a5d276b10>,
     <tensorflow.python.keras.layers.core.Dense at 0x1a5d276410>,
     <tensorflow.python.keras.layers.core.Dense at 0x1a5d141090>,
     <tensorflow.python.keras.layers.core.Dense at 0x1a5d3985d0>]




```python
model.get_layer('dense_1')
```




    <tensorflow.python.keras.layers.core.Dense at 0x1a5d276410>




```python
weights, biases = model.get_layer('dense_1').get_weights()
```


```python
weights
```




    array([[ 0.0316055 ,  0.03785409,  0.01424924, ...,  0.05335559,
             0.03587231, -0.05958051],
           [ 0.02760936, -0.00760906, -0.00725891, ..., -0.01800857,
             0.0004844 ,  0.02706798],
           [ 0.00826715, -0.03997412, -0.04378692, ..., -0.02673616,
            -0.0112099 , -0.02638585],
           ...,
           [-0.03979227, -0.05836861, -0.06440987, ...,  0.02992193,
             0.05634876,  0.01985031],
           [ 0.00265782,  0.06890348,  0.03913575, ...,  0.00965305,
            -0.00145537,  0.00975165],
           [-0.04517608, -0.02456116, -0.06251057, ...,  0.07049152,
            -0.068276  , -0.05999173]], dtype=float32)




```python
biases
```




    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)



Once the model's structure is created, it must be compiled.
This is when the loss function, optimizer, and any other metrics to be calculated during training and evaluation are added to the model.
There are additional parameters that we will learn about and opt to set in the future for increased performance.


```python
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='sgd',
    metrics=['accuracy']
)
```

Now the model is ready to be trained using the `fit()` method.


```python
history = model.fit(X_train, y_train, 
                    epochs=30,
                    validation_data=(X_valid, y_valid))
```

    Train on 55000 samples, validate on 5000 samples
    Epoch 1/30
    55000/55000 [==============================] - 11s 205us/sample - loss: 0.7276 - accuracy: 0.7620 - val_loss: 0.5263 - val_accuracy: 0.8234
    Epoch 2/30
    55000/55000 [==============================] - 9s 173us/sample - loss: 0.4903 - accuracy: 0.8292 - val_loss: 0.4458 - val_accuracy: 0.8496
    Epoch 3/30
    55000/55000 [==============================] - 12s 210us/sample - loss: 0.4447 - accuracy: 0.8452 - val_loss: 0.4407 - val_accuracy: 0.8432
    Epoch 4/30
    55000/55000 [==============================] - 8s 150us/sample - loss: 0.4187 - accuracy: 0.8540 - val_loss: 0.4089 - val_accuracy: 0.8624
    Epoch 5/30
    55000/55000 [==============================] - 8s 145us/sample - loss: 0.3978 - accuracy: 0.8605 - val_loss: 0.4422 - val_accuracy: 0.8366
    Epoch 6/30
    55000/55000 [==============================] - 8s 148us/sample - loss: 0.3818 - accuracy: 0.8670 - val_loss: 0.3997 - val_accuracy: 0.8580
    Epoch 7/30
    55000/55000 [==============================] - 8s 149us/sample - loss: 0.3689 - accuracy: 0.8703 - val_loss: 0.3738 - val_accuracy: 0.8710
    Epoch 8/30
    55000/55000 [==============================] - 8s 153us/sample - loss: 0.3571 - accuracy: 0.8734 - val_loss: 0.3689 - val_accuracy: 0.8730
    Epoch 9/30
    55000/55000 [==============================] - 8s 153us/sample - loss: 0.3473 - accuracy: 0.8767 - val_loss: 0.3563 - val_accuracy: 0.8736
    Epoch 10/30
    55000/55000 [==============================] - 9s 158us/sample - loss: 0.3368 - accuracy: 0.8800 - val_loss: 0.3582 - val_accuracy: 0.8710
    Epoch 11/30
    55000/55000 [==============================] - 9s 157us/sample - loss: 0.3283 - accuracy: 0.8825 - val_loss: 0.3468 - val_accuracy: 0.8764
    Epoch 12/30
    55000/55000 [==============================] - 8s 154us/sample - loss: 0.3205 - accuracy: 0.8851 - val_loss: 0.3436 - val_accuracy: 0.8762
    Epoch 13/30
    55000/55000 [==============================] - 8s 154us/sample - loss: 0.3135 - accuracy: 0.8872 - val_loss: 0.3346 - val_accuracy: 0.8824
    Epoch 14/30
    55000/55000 [==============================] - 9s 157us/sample - loss: 0.3059 - accuracy: 0.8887 - val_loss: 0.3321 - val_accuracy: 0.8816
    Epoch 15/30
    55000/55000 [==============================] - 9s 155us/sample - loss: 0.2998 - accuracy: 0.8923 - val_loss: 0.3438 - val_accuracy: 0.8768
    Epoch 16/30
    55000/55000 [==============================] - 8s 155us/sample - loss: 0.2931 - accuracy: 0.8944 - val_loss: 0.3353 - val_accuracy: 0.8810
    Epoch 17/30
    55000/55000 [==============================] - 8s 148us/sample - loss: 0.2875 - accuracy: 0.8967 - val_loss: 0.3154 - val_accuracy: 0.8872
    Epoch 18/30
    55000/55000 [==============================] - 8s 154us/sample - loss: 0.2810 - accuracy: 0.8991 - val_loss: 0.3366 - val_accuracy: 0.8810
    Epoch 19/30
    55000/55000 [==============================] - 9s 157us/sample - loss: 0.2765 - accuracy: 0.9002 - val_loss: 0.3178 - val_accuracy: 0.8846
    Epoch 20/30
    55000/55000 [==============================] - 9s 157us/sample - loss: 0.2712 - accuracy: 0.9027 - val_loss: 0.3204 - val_accuracy: 0.8858
    Epoch 21/30
    55000/55000 [==============================] - 9s 157us/sample - loss: 0.2664 - accuracy: 0.9043 - val_loss: 0.3231 - val_accuracy: 0.8860
    Epoch 22/30
    55000/55000 [==============================] - 9s 159us/sample - loss: 0.2624 - accuracy: 0.9048 - val_loss: 0.3098 - val_accuracy: 0.8916
    Epoch 23/30
    55000/55000 [==============================] - 9s 158us/sample - loss: 0.2566 - accuracy: 0.9068 - val_loss: 0.3130 - val_accuracy: 0.8920
    Epoch 24/30
    55000/55000 [==============================] - 9s 158us/sample - loss: 0.2518 - accuracy: 0.9083 - val_loss: 0.3100 - val_accuracy: 0.8856
    Epoch 25/30
    55000/55000 [==============================] - 9s 160us/sample - loss: 0.2486 - accuracy: 0.9102 - val_loss: 0.3028 - val_accuracy: 0.8930
    Epoch 26/30
    55000/55000 [==============================] - 9s 159us/sample - loss: 0.2436 - accuracy: 0.9127 - val_loss: 0.3028 - val_accuracy: 0.8920
    Epoch 27/30
    55000/55000 [==============================] - 9s 156us/sample - loss: 0.2396 - accuracy: 0.9137 - val_loss: 0.3061 - val_accuracy: 0.8928
    Epoch 28/30
    55000/55000 [==============================] - 9s 159us/sample - loss: 0.2354 - accuracy: 0.9143 - val_loss: 0.3067 - val_accuracy: 0.8928
    Epoch 29/30
    55000/55000 [==============================] - 9s 158us/sample - loss: 0.2319 - accuracy: 0.9160 - val_loss: 0.3154 - val_accuracy: 0.8850
    Epoch 30/30
    55000/55000 [==============================] - 8s 154us/sample - loss: 0.2289 - accuracy: 0.9175 - val_loss: 0.3161 - val_accuracy: 0.8888



```python
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.show()
```


![png](homl_ch10_Introduction-to-artificial-neural-networks-with-keras_files/homl_ch10_Introduction-to-artificial-neural-networks-with-keras_36_0.png)


After tuning the hyperparamters (discussed later) and once we are satisfied with the models performance on the training data, we can evaluate its accuracy on the test data.


```python
X_test_processed = fashion_preprocesser.transform(X_test)
model.evaluate(X_test_processed, y_test, verbose=0)
```




    [0.34439084025621414, 0.88]



We can also use the model to make predictions.
Using the `predict()` method, we get a probability per class.


```python
X_new = fashion_preprocesser.transform(X_test[:3])
np.round(model.predict(X_new), 2)
```




    array([[0.  , 0.  , 0.  , 0.  , 0.  , 0.02, 0.  , 0.01, 0.  , 0.97],
           [0.  , 0.  , 1.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
           [0.  , 1.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ]],
          dtype=float32)



The `predict_class()` method returns the class with the highest probability.


```python
y_pred = model.predict_classes(X_new)

fig = plt.figure(figsize=(12, 6))
for i in range(len(X_new)):
    plt.subplot(1, 3, i+1)
    plt.imshow(X_new[i, :, :])
    plt.title(class_names[y_pred[i]])
    plt.axis('off')

plt.show()
```


![png](homl_ch10_Introduction-to-artificial-neural-networks-with-keras_files/homl_ch10_Introduction-to-artificial-neural-networks-with-keras_42_0.png)


### Building a regression MLP using the sequential API

We will build a regression ANN to predict housing prices in CA.


```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, 
                                                              housing.target, 
                                                              random_state=0)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, 
                                                      y_train_full, 
                                                      random_state=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)
```


```python
model = keras.models.Sequential([
    keras.layers.Dense(30, activation='relu', input_shape = X_train.shape[1:]),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(1, activation='relu')
])

model.compile(loss='mean_squared_error', optimizer='sgd')

history = model.fit(X_train, y_train,
                    epochs=20,
                    validation_data=(X_valid, y_valid))
```

    Train on 11610 samples, validate on 3870 samples
    Epoch 1/20
    11610/11610 [==============================] - 2s 139us/sample - loss: 0.8768 - val_loss: 0.5428
    Epoch 2/20
    11610/11610 [==============================] - 1s 89us/sample - loss: 0.4568 - val_loss: 0.4310
    Epoch 3/20
    11610/11610 [==============================] - 1s 89us/sample - loss: 0.4191 - val_loss: 0.4303
    Epoch 4/20
    11610/11610 [==============================] - 1s 90us/sample - loss: 0.5505 - val_loss: 0.5068
    Epoch 5/20
    11610/11610 [==============================] - 1s 94us/sample - loss: 0.4449 - val_loss: 0.4401
    Epoch 6/20
    11610/11610 [==============================] - 1s 89us/sample - loss: 0.3979 - val_loss: 0.4169
    Epoch 7/20
    11610/11610 [==============================] - 1s 89us/sample - loss: 0.3796 - val_loss: 0.4067
    Epoch 8/20
    11610/11610 [==============================] - 1s 88us/sample - loss: 0.3702 - val_loss: 0.3919
    Epoch 9/20
    11610/11610 [==============================] - 1s 89us/sample - loss: 0.3605 - val_loss: 0.3848
    Epoch 10/20
    11610/11610 [==============================] - 1s 89us/sample - loss: 0.3522 - val_loss: 0.3801
    Epoch 11/20
    11610/11610 [==============================] - 1s 88us/sample - loss: 0.3477 - val_loss: 0.3747
    Epoch 12/20
    11610/11610 [==============================] - 1s 86us/sample - loss: 0.3430 - val_loss: 0.3631
    Epoch 13/20
    11610/11610 [==============================] - 1s 86us/sample - loss: 0.3385 - val_loss: 0.3653
    Epoch 14/20
    11610/11610 [==============================] - 1s 85us/sample - loss: 0.3343 - val_loss: 0.3575
    Epoch 15/20
    11610/11610 [==============================] - 1s 85us/sample - loss: 0.3299 - val_loss: 0.3673
    Epoch 16/20
    11610/11610 [==============================] - 1s 90us/sample - loss: 0.3328 - val_loss: 0.3774
    Epoch 17/20
    11610/11610 [==============================] - 1s 85us/sample - loss: 0.3243 - val_loss: 0.3480
    Epoch 18/20
    11610/11610 [==============================] - 1s 86us/sample - loss: 0.3215 - val_loss: 0.3494
    Epoch 19/20
    11610/11610 [==============================] - 1s 86us/sample - loss: 0.3208 - val_loss: 0.3454
    Epoch 20/20
    11610/11610 [==============================] - 1s 85us/sample - loss: 0.3155 - val_loss: 0.3423



```python
model.summary()
```

    Model: "sequential_6"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_19 (Dense)             (None, 30)                270       
    _________________________________________________________________
    dense_20 (Dense)             (None, 100)               3100      
    _________________________________________________________________
    dense_21 (Dense)             (None, 1)                 101       
    =================================================================
    Total params: 3,471
    Trainable params: 3,471
    Non-trainable params: 0
    _________________________________________________________________



```python
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.show()
```


![png](homl_ch10_Introduction-to-artificial-neural-networks-with-keras_files/homl_ch10_Introduction-to-artificial-neural-networks-with-keras_47_0.png)



```python
model.evaluate(X_test, y_test, verbose=0)
```




    0.32709958502488545




```python
y_pred = model.predict(X_test)
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color='b', s=10, alpha=0.2)
plt.plot(np.linspace(0, 5, 10), np.linspace(0, 5, 10), 'k--')
plt.title('Evaluation of the regression MLP', fontsize=14)
plt.xlabel('Real', fontsize=12)
plt.ylabel('Predicted', fontsize=12)
plt.show()
```


![png](homl_ch10_Introduction-to-artificial-neural-networks-with-keras_files/homl_ch10_Introduction-to-artificial-neural-networks-with-keras_49_0.png)


### Building complex models using the functional API

As an example of a more complex ANN topology, we will build a *Wide & Deep* neural network.
This is where some or all of the input layers connect directly to the output layer.
This allows the ANN to learn both simple and deep rules.
This architecture is built below using Keras's *functional API*.


```python
input_ = keras.layers.Input(shape=X_train.shape[1:])
hidden1 = keras.layers.Dense(30, activation='relu')(input_)
hidden2 = keras.layers.Dense(100, activation='relu')(hidden1)
concat = keras.layers.Concatenate()([input_, hidden2])
output = keras.layers.Dense(1, activation='relu')(concat)
model = keras.Model(inputs=[input_], outputs=[output])
```

Here is a line-by-line explanation:

1. First, an `Input` object is created with the shape and data type specified.
2. A dense layer with 30 neurons was created and connected to the input layer by passing it as input to a function.
3. A dense layer with 30 neurons was created and connected to the first hidden layer.
4. A `Concatenate` layer concatenated the input and second hidden layer.
5. An output layer was made and given the concatenate layer.
6. Finally, a Keras `Model` was created and given the input and output layers.

This strings together the layers to make a network.
A layer is connected to another by passing one as the input to a function call of the other.

Next, everything is the same as before.


```python
model.compile(loss='mean_squared_error', optimizer='sgd')

history = model.fit(X_train, y_train,
                    epochs=20,
                    validation_data=(X_valid, y_valid))

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.show()
```

    Train on 11610 samples, validate on 3870 samples
    Epoch 1/20
    11610/11610 [==============================] - 2s 152us/sample - loss: 0.8164 - val_loss: 0.5035
    Epoch 2/20
    11610/11610 [==============================] - 1s 98us/sample - loss: 0.4466 - val_loss: 0.4419
    Epoch 3/20
    11610/11610 [==============================] - 1s 97us/sample - loss: 0.4170 - val_loss: 0.4270
    Epoch 4/20
    11610/11610 [==============================] - 1s 105us/sample - loss: 0.4013 - val_loss: 0.5051
    Epoch 5/20
    11610/11610 [==============================] - 1s 99us/sample - loss: 0.5224 - val_loss: 0.4491
    Epoch 6/20
    11610/11610 [==============================] - 1s 99us/sample - loss: 0.3978 - val_loss: 0.4101
    Epoch 7/20
    11610/11610 [==============================] - 1s 105us/sample - loss: 0.3849 - val_loss: 0.4103
    Epoch 8/20
    11610/11610 [==============================] - 1s 99us/sample - loss: 0.3788 - val_loss: 0.4083
    Epoch 9/20
    11610/11610 [==============================] - 1s 99us/sample - loss: 0.3718 - val_loss: 0.4048
    Epoch 10/20
    11610/11610 [==============================] - 1s 99us/sample - loss: 0.3673 - val_loss: 0.3954
    Epoch 11/20
    11610/11610 [==============================] - 1s 99us/sample - loss: 0.3649 - val_loss: 0.3911
    Epoch 12/20
    11610/11610 [==============================] - 1s 99us/sample - loss: 0.3603 - val_loss: 0.3855
    Epoch 13/20
    11610/11610 [==============================] - 1s 103us/sample - loss: 0.3583 - val_loss: 0.3822
    Epoch 14/20
    11610/11610 [==============================] - 1s 99us/sample - loss: 0.3546 - val_loss: 0.3876
    Epoch 15/20
    11610/11610 [==============================] - 1s 100us/sample - loss: 0.3548 - val_loss: 0.3858
    Epoch 16/20
    11610/11610 [==============================] - 1s 95us/sample - loss: 0.3489 - val_loss: 0.3978
    Epoch 17/20
    11610/11610 [==============================] - 1s 98us/sample - loss: 0.3458 - val_loss: 0.3714
    Epoch 18/20
    11610/11610 [==============================] - 1s 104us/sample - loss: 0.3449 - val_loss: 0.3721
    Epoch 19/20
    11610/11610 [==============================] - 1s 101us/sample - loss: 0.3429 - val_loss: 0.3754
    Epoch 20/20
    11610/11610 [==============================] - 1s 98us/sample - loss: 0.3374 - val_loss: 0.3664



![png](homl_ch10_Introduction-to-artificial-neural-networks-with-keras_files/homl_ch10_Introduction-to-artificial-neural-networks-with-keras_53_1.png)



```python
model.evaluate(X_test, y_test, verbose=0)
```




    0.35167020106500435




```python
y_pred_new = model.predict(X_test)
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color='b', s=10, alpha=0.2)
plt.plot(np.linspace(0, 5, 10), np.linspace(0, 5, 10), 'k--')
plt.title('Evaluation of the regression MLP (Wide & Deep)', fontsize=14)
plt.xlabel('Real', fontsize=12)
plt.ylabel('Predicted', fontsize=12)
plt.show()
```


![png](homl_ch10_Introduction-to-artificial-neural-networks-with-keras_files/homl_ch10_Introduction-to-artificial-neural-networks-with-keras_55_0.png)



```python
plt.figure(figsize=(8, 5))
plt.scatter(y_pred, y_pred_new, color='b', s=10, alpha=0.2)
plt.plot(np.linspace(0, 5, 10), np.linspace(0, 5, 10), 'k--')
plt.title('Comparison of the two models', fontsize=14)
plt.xlabel('MLP regression', fontsize=12)
plt.ylabel('Wide and Deep', fontsize=12)
plt.show()
```


![png](homl_ch10_Introduction-to-artificial-neural-networks-with-keras_files/homl_ch10_Introduction-to-artificial-neural-networks-with-keras_56_0.png)


It is also possible to have multiple inputs.
This can be useful if you only want some inputs to go through the hidden layers.
Below is an example where features 0 through 4 go through the wide path and 2 through 7 go through the hidden layers.


```python
# Two input layers.
input_A = keras.layers.Input(shape=[5], name='wide_input')
input_B = keras.layers.Input(shape=[6], name='deep_input')

# Hidden layers.
hidden1 = keras.layers.Dense(30, activation='relu')(input_B)
hidden2 = keras.layers.Dense(100, activation='relu')(hidden1)

# Concatenate.
concat = keras.layers.concatenate([input_A, hidden2])

# Output.
output = keras.layers.Dense(1, name='output')(concat)

# Make the model object.
model = keras.Model(inputs=[input_A, input_B], outputs=[output])
model.summary()
```

    Model: "model_5"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    deep_input (InputLayer)         [(None, 6)]          0                                            
    __________________________________________________________________________________________________
    dense_25 (Dense)                (None, 30)           210         deep_input[0][0]                 
    __________________________________________________________________________________________________
    wide_input (InputLayer)         [(None, 5)]          0                                            
    __________________________________________________________________________________________________
    dense_26 (Dense)                (None, 100)          3100        dense_25[0][0]                   
    __________________________________________________________________________________________________
    concatenate_5 (Concatenate)     (None, 105)          0           wide_input[0][0]                 
                                                                     dense_26[0][0]                   
    __________________________________________________________________________________________________
    output (Dense)                  (None, 1)            106         concatenate_5[0][0]              
    ==================================================================================================
    Total params: 3,416
    Trainable params: 3,416
    Non-trainable params: 0
    __________________________________________________________________________________________________


Compiling the model is the same as before.


```python
model.compile(loss='mse', optimizer=keras.optimizers.SGD(lr=1e-3))
```

To fit the model, we must pass two matrices, one per input.
Alternatively, a dictionary with keys of the input layer names and values of the input matrices can be passed.


```python
def split_input_matrices(A):
    return A[:, :5], A[:, 2:]


history = model.fit(split_input_matrices(X_train),
                    y_train,
                    epochs=20,
                    validation_data=(split_input_matrices(X_valid), y_valid))
```

    Train on 11610 samples, validate on 3870 samples
    Epoch 1/20
    11610/11610 [==============================] - 2s 156us/sample - loss: 1.7940 - val_loss: 0.8289
    Epoch 2/20
    11610/11610 [==============================] - 1s 97us/sample - loss: 0.7432 - val_loss: 0.6778
    Epoch 3/20
    11610/11610 [==============================] - 1s 96us/sample - loss: 0.6408 - val_loss: 0.6271
    Epoch 4/20
    11610/11610 [==============================] - 1s 96us/sample - loss: 0.6046 - val_loss: 0.6001
    Epoch 5/20
    11610/11610 [==============================] - 1s 96us/sample - loss: 0.5823 - val_loss: 0.5827
    Epoch 6/20
    11610/11610 [==============================] - 1s 90us/sample - loss: 0.5648 - val_loss: 0.5679
    Epoch 7/20
    11610/11610 [==============================] - 1s 85us/sample - loss: 0.5504 - val_loss: 0.5554
    Epoch 8/20
    11610/11610 [==============================] - 1s 99us/sample - loss: 0.5387 - val_loss: 0.5435
    Epoch 9/20
    11610/11610 [==============================] - 1s 96us/sample - loss: 0.5288 - val_loss: 0.5363
    Epoch 10/20
    11610/11610 [==============================] - 1s 105us/sample - loss: 0.5183 - val_loss: 0.5237
    Epoch 11/20
    11610/11610 [==============================] - 1s 96us/sample - loss: 0.5098 - val_loss: 0.5251
    Epoch 12/20
    11610/11610 [==============================] - 1s 96us/sample - loss: 0.5028 - val_loss: 0.5141
    Epoch 13/20
    11610/11610 [==============================] - 1s 95us/sample - loss: 0.4961 - val_loss: 0.5018
    Epoch 14/20
    11610/11610 [==============================] - 1s 96us/sample - loss: 0.4905 - val_loss: 0.4985
    Epoch 15/20
    11610/11610 [==============================] - 1s 98us/sample - loss: 0.4847 - val_loss: 0.4929
    Epoch 16/20
    11610/11610 [==============================] - 1s 96us/sample - loss: 0.4800 - val_loss: 0.4872
    Epoch 17/20
    11610/11610 [==============================] - 1s 95us/sample - loss: 0.4752 - val_loss: 0.4839
    Epoch 18/20
    11610/11610 [==============================] - 1s 100us/sample - loss: 0.4705 - val_loss: 0.4788
    Epoch 19/20
    11610/11610 [==============================] - 1s 96us/sample - loss: 0.4671 - val_loss: 0.4749
    Epoch 20/20
    11610/11610 [==============================] - 1s 111us/sample - loss: 0.4641 - val_loss: 0.4726



```python
model.evaluate(split_input_matrices(X_test), y_test, verbose=0)
```




    0.488090944243956




```python
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.show()
```


![png](homl_ch10_Introduction-to-artificial-neural-networks-with-keras_files/homl_ch10_Introduction-to-artificial-neural-networks-with-keras_64_0.png)


Adding outputs is just as easy in adding inputs.
The following code is the same structure as the *Wide & Deep* ANN except there is an additional output from the second hidden layer.


```python
# Two input layers.
input_A = keras.layers.Input(shape=[5], name='wide_input')
input_B = keras.layers.Input(shape=[6], name='deep_input')

# Hidden layers.
hidden1 = keras.layers.Dense(30, activation='relu')(input_B)
hidden2 = keras.layers.Dense(100, activation='relu')(hidden1)

# Concatenate.
concat = keras.layers.concatenate([input_A, hidden2])

# Output.
output = keras.layers.Dense(1, name='output')(concat)
aux_output = keras.layers.Dense(1, name='aux_output')(hidden2)

# Make the model object.
model = keras.Model(inputs=[input_A, input_B], outputs=[output, aux_output])
model.summary()
```

    Model: "model_6"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    deep_input (InputLayer)         [(None, 6)]          0                                            
    __________________________________________________________________________________________________
    dense_27 (Dense)                (None, 30)           210         deep_input[0][0]                 
    __________________________________________________________________________________________________
    wide_input (InputLayer)         [(None, 5)]          0                                            
    __________________________________________________________________________________________________
    dense_28 (Dense)                (None, 100)          3100        dense_27[0][0]                   
    __________________________________________________________________________________________________
    concatenate_6 (Concatenate)     (None, 105)          0           wide_input[0][0]                 
                                                                     dense_28[0][0]                   
    __________________________________________________________________________________________________
    output (Dense)                  (None, 1)            106         concatenate_6[0][0]              
    __________________________________________________________________________________________________
    aux_output (Dense)              (None, 1)            101         dense_28[0][0]                   
    ==================================================================================================
    Total params: 3,517
    Trainable params: 3,517
    Non-trainable params: 0
    __________________________________________________________________________________________________


Each output requires a loss function and we can weight the loss scores by which we value more.
These are stated at the compile step.


```python
model.compile(loss=['mse', 'mse'],
              loss_weights=[0.9, 0.1],
              optimizer='sgd')
```

For fitting, each output also needs a set of labels.


```python
history = model.fit(split_input_matrices(X_train), 
                    (y_train, y_train),
                    epochs=20,
                    validation_data=(
                        split_input_matrices(X_valid), 
                        (y_valid, y_valid)
                    ))
```

    Train on 11610 samples, validate on 3870 samples
    Epoch 1/20
    11610/11610 [==============================] - 2s 204us/sample - loss: 0.9768 - output_loss: 0.8486 - aux_output_loss: 2.1288 - val_loss: 0.6062 - val_output_loss: 0.5343 - val_aux_output_loss: 1.2543
    Epoch 2/20
    11610/11610 [==============================] - 1s 115us/sample - loss: 0.5879 - output_loss: 0.5199 - aux_output_loss: 1.1992 - val_loss: 0.5454 - val_output_loss: 0.4900 - val_aux_output_loss: 1.0455
    Epoch 3/20
    11610/11610 [==============================] - 1s 111us/sample - loss: 0.5299 - output_loss: 0.4790 - aux_output_loss: 0.9873 - val_loss: 0.5138 - val_output_loss: 0.4720 - val_aux_output_loss: 0.8902
    Epoch 4/20
    11610/11610 [==============================] - 1s 113us/sample - loss: 0.4945 - output_loss: 0.4561 - aux_output_loss: 0.8401 - val_loss: 0.4920 - val_output_loss: 0.4593 - val_aux_output_loss: 0.7860
    Epoch 5/20
    11610/11610 [==============================] - 1s 113us/sample - loss: 0.4687 - output_loss: 0.4383 - aux_output_loss: 0.7415 - val_loss: 0.4729 - val_output_loss: 0.4456 - val_aux_output_loss: 0.7189
    Epoch 6/20
    11610/11610 [==============================] - 1s 122us/sample - loss: 0.4548 - output_loss: 0.4294 - aux_output_loss: 0.6840 - val_loss: 0.4574 - val_output_loss: 0.4315 - val_aux_output_loss: 0.6917
    Epoch 7/20
    11610/11610 [==============================] - 1s 99us/sample - loss: 0.4424 - output_loss: 0.4198 - aux_output_loss: 0.6465 - val_loss: 0.4458 - val_output_loss: 0.4233 - val_aux_output_loss: 0.6483
    Epoch 8/20
    11610/11610 [==============================] - 1s 113us/sample - loss: 0.4287 - output_loss: 0.4078 - aux_output_loss: 0.6158 - val_loss: 0.4472 - val_output_loss: 0.4206 - val_aux_output_loss: 0.6872
    Epoch 9/20
    11610/11610 [==============================] - 1s 112us/sample - loss: 0.4231 - output_loss: 0.4036 - aux_output_loss: 0.5982 - val_loss: 0.4330 - val_output_loss: 0.4132 - val_aux_output_loss: 0.6114
    Epoch 10/20
    11610/11610 [==============================] - 1s 119us/sample - loss: 0.4153 - output_loss: 0.3969 - aux_output_loss: 0.5812 - val_loss: 0.4268 - val_output_loss: 0.4062 - val_aux_output_loss: 0.6125
    Epoch 11/20
    11610/11610 [==============================] - 1s 104us/sample - loss: 0.4265 - output_loss: 0.4104 - aux_output_loss: 0.5697 - val_loss: 0.4254 - val_output_loss: 0.4081 - val_aux_output_loss: 0.5815
    Epoch 12/20
    11610/11610 [==============================] - 1s 112us/sample - loss: 0.4127 - output_loss: 0.3970 - aux_output_loss: 0.5526 - val_loss: 0.4326 - val_output_loss: 0.4128 - val_aux_output_loss: 0.6108
    Epoch 13/20
    11610/11610 [==============================] - 1s 113us/sample - loss: 0.4014 - output_loss: 0.3852 - aux_output_loss: 0.5459 - val_loss: 0.4086 - val_output_loss: 0.3904 - val_aux_output_loss: 0.5726
    Epoch 14/20
    11610/11610 [==============================] - 1s 112us/sample - loss: 0.3892 - output_loss: 0.3728 - aux_output_loss: 0.5361 - val_loss: 0.3993 - val_output_loss: 0.3812 - val_aux_output_loss: 0.5628
    Epoch 15/20
    11610/11610 [==============================] - 1s 113us/sample - loss: 0.3861 - output_loss: 0.3702 - aux_output_loss: 0.5284 - val_loss: 0.3988 - val_output_loss: 0.3800 - val_aux_output_loss: 0.5677
    Epoch 16/20
    11610/11610 [==============================] - 1s 114us/sample - loss: 0.3777 - output_loss: 0.3621 - aux_output_loss: 0.5168 - val_loss: 0.3905 - val_output_loss: 0.3721 - val_aux_output_loss: 0.5558
    Epoch 17/20
    11610/11610 [==============================] - 1s 106us/sample - loss: 0.3786 - output_loss: 0.3639 - aux_output_loss: 0.5119 - val_loss: 0.3855 - val_output_loss: 0.3685 - val_aux_output_loss: 0.5381
    Epoch 18/20
    11610/11610 [==============================] - 1s 100us/sample - loss: 0.3731 - output_loss: 0.3586 - aux_output_loss: 0.5034 - val_loss: 0.3877 - val_output_loss: 0.3716 - val_aux_output_loss: 0.5324
    Epoch 19/20
    11610/11610 [==============================] - 1s 114us/sample - loss: 0.3664 - output_loss: 0.3521 - aux_output_loss: 0.4946 - val_loss: 0.3777 - val_output_loss: 0.3600 - val_aux_output_loss: 0.5377
    Epoch 20/20
    11610/11610 [==============================] - 1s 113us/sample - loss: 0.3611 - output_loss: 0.3470 - aux_output_loss: 0.4882 - val_loss: 0.3807 - val_output_loss: 0.3634 - val_aux_output_loss: 0.5371



```python
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.show()
```


![png](homl_ch10_Introduction-to-artificial-neural-networks-with-keras_files/homl_ch10_Introduction-to-artificial-neural-networks-with-keras_71_0.png)



```python
X_new = split_input_matrices(X_test)
y_new, y_new_aux = model.predict(X_new)
```


```python
plt.scatter(y_new, y_new_aux, c='b', s=20, alpha=0.2)
plt.plot(np.linspace(0, 5, 10), np.linspace(0, 9, 10), 'k--')
plt.xlabel('Main output prediction')
plt.ylabel('Aux. output prediction')
plt.show()
```


![png](homl_ch10_Introduction-to-artificial-neural-networks-with-keras_files/homl_ch10_Introduction-to-artificial-neural-networks-with-keras_73_0.png)



```python
from sklearn.neighbors import KNeighborsRegressor
y_order = np.argsort(y_new.ravel())
y_diff = y_new[y_order] - y_new_aux[y_order]
x = np.arange(len(y_diff)).reshape(-1, 1)

knn = KNeighborsRegressor(n_neighbors=100)
y_knn = knn.fit(x, y_diff).predict(x)

plt.figure(figsize=(8, 5))
plt.plot(x, y_diff, 'g.', alpha=0.15, label='diff.')
plt.plot(x, y_knn, 'k-', linewidth=2.5, label='running avg.')
plt.axis((0, np.max(x), -1.5, 2.5))
plt.xlabel('ordered output')
plt.ylabel('output - aux. output')
plt.title('Difference between main and aux. outputs of model.')
plt.legend(loc='best')
plt.show()
```


![png](homl_ch10_Introduction-to-artificial-neural-networks-with-keras_files/homl_ch10_Introduction-to-artificial-neural-networks-with-keras_74_0.png)



```python

```
