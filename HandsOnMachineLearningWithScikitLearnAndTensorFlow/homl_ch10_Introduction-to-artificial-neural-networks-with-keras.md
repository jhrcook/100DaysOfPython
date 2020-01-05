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

    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/IPython/config.py:13: ShimWarning: The `IPython.config` package has been deprecated since IPython 4.0. You should import from traitlets.config instead.
      "You should import from traitlets.config instead.", ShimWarning)
    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/ipycache.py:17: UserWarning: IPython.utils.traitlets has moved to a top-level traitlets package.
      from IPython.utils.traitlets import Unicode


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

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    flatten (Flatten)            (None, 784)               0         
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




    [<tensorflow.python.keras.layers.core.Flatten at 0x1a47852cd0>,
     <tensorflow.python.keras.layers.core.Dense at 0x1a4786b1d0>,
     <tensorflow.python.keras.layers.core.Dense at 0x1a1d650690>,
     <tensorflow.python.keras.layers.core.Dense at 0x1a1d80cdd0>]




```python
model.get_layer('dense_1')
```




    <tensorflow.python.keras.layers.core.Dense at 0x1a4786b1d0>




```python
weights, biases = model.get_layer('dense_1').get_weights()
```


```python
weights
```




    array([[ 4.4014677e-03, -4.7703154e-02, -5.6954548e-02, ...,
             1.3799518e-02,  6.8606809e-02,  4.4642702e-02],
           [ 1.1618771e-02,  6.5349728e-02, -1.9043323e-02, ...,
            -6.2322959e-02, -3.4375217e-02, -6.4794019e-02],
           [ 4.7343969e-04, -2.6652794e-02,  5.7301193e-02, ...,
            -7.3247261e-02,  3.8669445e-02,  6.1915472e-02],
           ...,
           [ 2.1660134e-02, -3.2525033e-02, -2.2409968e-02, ...,
            -1.6713545e-02, -4.5746118e-02, -3.6396444e-02],
           [ 2.5894038e-02,  4.4876792e-02, -3.1769171e-02, ...,
            -6.1228871e-05,  2.0354383e-02, -1.6598497e-02],
           [ 3.8472585e-02,  5.4374188e-02,  7.2529078e-02, ...,
            -4.8064545e-02,  3.8844623e-02, -2.2803590e-02]], dtype=float32)




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
    55000/55000 [==============================] - 26s 475us/sample - loss: 0.7192 - accuracy: 0.7617 - val_loss: 0.5081 - val_accuracy: 0.8246
    Epoch 2/30
    55000/55000 [==============================] - 9s 159us/sample - loss: 0.4897 - accuracy: 0.8295 - val_loss: 0.4663 - val_accuracy: 0.8366
    Epoch 3/30
    55000/55000 [==============================] - 8s 149us/sample - loss: 0.4448 - accuracy: 0.8448 - val_loss: 0.4331 - val_accuracy: 0.8514
    Epoch 4/30
    55000/55000 [==============================] - 8s 145us/sample - loss: 0.4185 - accuracy: 0.8535 - val_loss: 0.4125 - val_accuracy: 0.8560
    Epoch 5/30
    55000/55000 [==============================] - 8s 145us/sample - loss: 0.3976 - accuracy: 0.8608 - val_loss: 0.3831 - val_accuracy: 0.8666
    Epoch 6/30
    55000/55000 [==============================] - 8s 144us/sample - loss: 0.3837 - accuracy: 0.8657 - val_loss: 0.3930 - val_accuracy: 0.8628
    Epoch 7/30
    55000/55000 [==============================] - 8s 139us/sample - loss: 0.3699 - accuracy: 0.8691 - val_loss: 0.3921 - val_accuracy: 0.8614
    Epoch 8/30
    55000/55000 [==============================] - 8s 147us/sample - loss: 0.3578 - accuracy: 0.8733 - val_loss: 0.3557 - val_accuracy: 0.8758
    Epoch 9/30
    55000/55000 [==============================] - 8s 144us/sample - loss: 0.3474 - accuracy: 0.8768 - val_loss: 0.3638 - val_accuracy: 0.8714
    Epoch 10/30
    55000/55000 [==============================] - 8s 147us/sample - loss: 0.3386 - accuracy: 0.8802 - val_loss: 0.3451 - val_accuracy: 0.8804
    Epoch 11/30
    55000/55000 [==============================] - 8s 143us/sample - loss: 0.3294 - accuracy: 0.8820 - val_loss: 0.3807 - val_accuracy: 0.8636
    Epoch 12/30
    55000/55000 [==============================] - 8s 140us/sample - loss: 0.3211 - accuracy: 0.8856 - val_loss: 0.3555 - val_accuracy: 0.8730
    Epoch 13/30
    55000/55000 [==============================] - 8s 142us/sample - loss: 0.3143 - accuracy: 0.8871 - val_loss: 0.3371 - val_accuracy: 0.8798
    Epoch 14/30
    55000/55000 [==============================] - 8s 143us/sample - loss: 0.3062 - accuracy: 0.8897 - val_loss: 0.3291 - val_accuracy: 0.8800
    Epoch 15/30
    55000/55000 [==============================] - 8s 143us/sample - loss: 0.3006 - accuracy: 0.8910 - val_loss: 0.3274 - val_accuracy: 0.8838
    Epoch 16/30
    55000/55000 [==============================] - 8s 143us/sample - loss: 0.2940 - accuracy: 0.8938 - val_loss: 0.3307 - val_accuracy: 0.8836
    Epoch 17/30
    55000/55000 [==============================] - 8s 144us/sample - loss: 0.2881 - accuracy: 0.8961 - val_loss: 0.3357 - val_accuracy: 0.8822
    Epoch 18/30
    55000/55000 [==============================] - 8s 145us/sample - loss: 0.2821 - accuracy: 0.8973 - val_loss: 0.3273 - val_accuracy: 0.8848
    Epoch 19/30
    55000/55000 [==============================] - 8s 144us/sample - loss: 0.2768 - accuracy: 0.9012 - val_loss: 0.3170 - val_accuracy: 0.8880
    Epoch 20/30
    55000/55000 [==============================] - 8s 144us/sample - loss: 0.2728 - accuracy: 0.9008 - val_loss: 0.3115 - val_accuracy: 0.8910
    Epoch 21/30
    55000/55000 [==============================] - 8s 144us/sample - loss: 0.2669 - accuracy: 0.9037 - val_loss: 0.3095 - val_accuracy: 0.8900
    Epoch 22/30
    55000/55000 [==============================] - 8s 145us/sample - loss: 0.2628 - accuracy: 0.9056 - val_loss: 0.3262 - val_accuracy: 0.8804
    Epoch 23/30
    55000/55000 [==============================] - 9s 155us/sample - loss: 0.2581 - accuracy: 0.9071 - val_loss: 0.3065 - val_accuracy: 0.8904
    Epoch 24/30
    55000/55000 [==============================] - 8s 144us/sample - loss: 0.2536 - accuracy: 0.9089 - val_loss: 0.3045 - val_accuracy: 0.8934
    Epoch 25/30
    55000/55000 [==============================] - 8s 146us/sample - loss: 0.2496 - accuracy: 0.9101 - val_loss: 0.3159 - val_accuracy: 0.8874
    Epoch 26/30
    55000/55000 [==============================] - 8s 147us/sample - loss: 0.2455 - accuracy: 0.9119 - val_loss: 0.3052 - val_accuracy: 0.8926
    Epoch 27/30
    55000/55000 [==============================] - 8s 146us/sample - loss: 0.2415 - accuracy: 0.9133 - val_loss: 0.3003 - val_accuracy: 0.8966
    Epoch 28/30
    55000/55000 [==============================] - 9s 163us/sample - loss: 0.2377 - accuracy: 0.9142 - val_loss: 0.3035 - val_accuracy: 0.8940
    Epoch 29/30
    55000/55000 [==============================] - 8s 145us/sample - loss: 0.2330 - accuracy: 0.9161 - val_loss: 0.2989 - val_accuracy: 0.8950
    Epoch 30/30
    55000/55000 [==============================] - 8s 146us/sample - loss: 0.2303 - accuracy: 0.9181 - val_loss: 0.3084 - val_accuracy: 0.8916



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




    [0.33497632868289945, 0.8802]



We can also use the model to make predictions.
Using the `predict()` method, we get a probability per class.


```python
X_new = fashion_preprocesser.transform(X_test[:3])
np.round(model.predict(X_new), 2)
```




    array([[0.  , 0.  , 0.  , 0.  , 0.  , 0.03, 0.  , 0.01, 0.  , 0.95],
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
    11610/11610 [==============================] - 2s 141us/sample - loss: 0.7797 - val_loss: 0.5026
    Epoch 2/20
    11610/11610 [==============================] - 1s 101us/sample - loss: 0.4528 - val_loss: 0.4491
    Epoch 3/20
    11610/11610 [==============================] - 1s 87us/sample - loss: 0.4151 - val_loss: 0.4242
    Epoch 4/20
    11610/11610 [==============================] - 1s 87us/sample - loss: 0.3990 - val_loss: 0.4128
    Epoch 5/20
    11610/11610 [==============================] - 1s 87us/sample - loss: 0.3832 - val_loss: 0.4045
    Epoch 6/20
    11610/11610 [==============================] - 1s 86us/sample - loss: 0.3793 - val_loss: 0.3995
    Epoch 7/20
    11610/11610 [==============================] - 1s 85us/sample - loss: 0.3676 - val_loss: 0.3923
    Epoch 8/20
    11610/11610 [==============================] - 1s 89us/sample - loss: 0.3601 - val_loss: 0.3837
    Epoch 9/20
    11610/11610 [==============================] - 1s 86us/sample - loss: 0.3563 - val_loss: 0.3742
    Epoch 10/20
    11610/11610 [==============================] - 1s 91us/sample - loss: 0.3504 - val_loss: 0.3764
    Epoch 11/20
    11610/11610 [==============================] - 1s 85us/sample - loss: 0.3744 - val_loss: 0.3811
    Epoch 12/20
    11610/11610 [==============================] - 1s 84us/sample - loss: 0.3466 - val_loss: 0.3691
    Epoch 13/20
    11610/11610 [==============================] - 1s 85us/sample - loss: 0.3396 - val_loss: 0.3711
    Epoch 14/20
    11610/11610 [==============================] - 1s 85us/sample - loss: 0.3353 - val_loss: 0.3690
    Epoch 15/20
    11610/11610 [==============================] - 1s 85us/sample - loss: 0.3358 - val_loss: 0.3662
    Epoch 16/20
    11610/11610 [==============================] - 1s 85us/sample - loss: 0.3283 - val_loss: 0.3539
    Epoch 17/20
    11610/11610 [==============================] - 1s 84us/sample - loss: 0.3276 - val_loss: 0.3865
    Epoch 18/20
    11610/11610 [==============================] - 1s 83us/sample - loss: 0.3215 - val_loss: 0.3634
    Epoch 19/20
    11610/11610 [==============================] - 1s 82us/sample - loss: 0.3206 - val_loss: 0.3467
    Epoch 20/20
    11610/11610 [==============================] - 1s 84us/sample - loss: 0.3158 - val_loss: 0.3442



```python
model.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense (Dense)                (None, 30)                270       
    _________________________________________________________________
    dense_1 (Dense)              (None, 100)               3100      
    _________________________________________________________________
    dense_2 (Dense)              (None, 1)                 101       
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




    0.32532836919607117




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
    11610/11610 [==============================] - 2s 156us/sample - loss: 0.7767 - val_loss: 0.4869
    Epoch 2/20
    11610/11610 [==============================] - 1s 86us/sample - loss: 0.4573 - val_loss: 0.4619
    Epoch 3/20
    11610/11610 [==============================] - 1s 95us/sample - loss: 0.4768 - val_loss: 0.4714
    Epoch 4/20
    11610/11610 [==============================] - 1s 105us/sample - loss: 0.4138 - val_loss: 0.4226
    Epoch 5/20
    11610/11610 [==============================] - 1s 98us/sample - loss: 0.3877 - val_loss: 0.4010
    Epoch 6/20
    11610/11610 [==============================] - 1s 114us/sample - loss: 0.3811 - val_loss: 0.3984
    Epoch 7/20
    11610/11610 [==============================] - 2s 168us/sample - loss: 0.3746 - val_loss: 0.4009
    Epoch 8/20
    11610/11610 [==============================] - 2s 168us/sample - loss: 0.3716 - val_loss: 0.3903
    Epoch 9/20
    11610/11610 [==============================] - 1s 127us/sample - loss: 0.3674 - val_loss: 0.3920
    Epoch 10/20
    11610/11610 [==============================] - 1s 98us/sample - loss: 0.3634 - val_loss: 0.3894
    Epoch 11/20
    11610/11610 [==============================] - 1s 96us/sample - loss: 0.3762 - val_loss: 0.4032
    Epoch 12/20
    11610/11610 [==============================] - 1s 96us/sample - loss: 0.3597 - val_loss: 0.3815
    Epoch 13/20
    11610/11610 [==============================] - 1s 95us/sample - loss: 0.3569 - val_loss: 0.3878
    Epoch 14/20
    11610/11610 [==============================] - 1s 96us/sample - loss: 0.3540 - val_loss: 0.3790
    Epoch 15/20
    11610/11610 [==============================] - 1s 95us/sample - loss: 0.3523 - val_loss: 0.3745
    Epoch 16/20
    11610/11610 [==============================] - 1s 101us/sample - loss: 0.3478 - val_loss: 0.3718
    Epoch 17/20
    11610/11610 [==============================] - 1s 96us/sample - loss: 0.3464 - val_loss: 0.3732
    Epoch 18/20
    11610/11610 [==============================] - 1s 95us/sample - loss: 0.3423 - val_loss: 0.3759
    Epoch 19/20
    11610/11610 [==============================] - 1s 96us/sample - loss: 0.3418 - val_loss: 0.3669
    Epoch 20/20
    11610/11610 [==============================] - 1s 96us/sample - loss: 0.3380 - val_loss: 0.3669



![png](homl_ch10_Introduction-to-artificial-neural-networks-with-keras_files/homl_ch10_Introduction-to-artificial-neural-networks-with-keras_53_1.png)



```python
model.evaluate(X_test, y_test, verbose=0)
```




    0.35652519011682315




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

    Model: "model_1"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    deep_input (InputLayer)         [(None, 6)]          0                                            
    __________________________________________________________________________________________________
    dense_6 (Dense)                 (None, 30)           210         deep_input[0][0]                 
    __________________________________________________________________________________________________
    wide_input (InputLayer)         [(None, 5)]          0                                            
    __________________________________________________________________________________________________
    dense_7 (Dense)                 (None, 100)          3100        dense_6[0][0]                    
    __________________________________________________________________________________________________
    concatenate_1 (Concatenate)     (None, 105)          0           wide_input[0][0]                 
                                                                     dense_7[0][0]                    
    __________________________________________________________________________________________________
    output (Dense)                  (None, 1)            106         concatenate_1[0][0]              
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
    11610/11610 [==============================] - 2s 151us/sample - loss: 2.1714 - val_loss: 0.9608
    Epoch 2/20
    11610/11610 [==============================] - 1s 95us/sample - loss: 0.7916 - val_loss: 0.7177
    Epoch 3/20
    11610/11610 [==============================] - 1s 100us/sample - loss: 0.6790 - val_loss: 0.6621
    Epoch 4/20
    11610/11610 [==============================] - 1s 95us/sample - loss: 0.6309 - val_loss: 0.6262
    Epoch 5/20
    11610/11610 [==============================] - 1s 95us/sample - loss: 0.5986 - val_loss: 0.5993
    Epoch 6/20
    11610/11610 [==============================] - 1s 95us/sample - loss: 0.5740 - val_loss: 0.5806
    Epoch 7/20
    11610/11610 [==============================] - 1s 94us/sample - loss: 0.5541 - val_loss: 0.5651
    Epoch 8/20
    11610/11610 [==============================] - 1s 95us/sample - loss: 0.5379 - val_loss: 0.5440
    Epoch 9/20
    11610/11610 [==============================] - 1s 94us/sample - loss: 0.5250 - val_loss: 0.5348
    Epoch 10/20
    11610/11610 [==============================] - 1s 94us/sample - loss: 0.5135 - val_loss: 0.5195
    Epoch 11/20
    11610/11610 [==============================] - 1s 94us/sample - loss: 0.5049 - val_loss: 0.5128
    Epoch 12/20
    11610/11610 [==============================] - 1s 100us/sample - loss: 0.4970 - val_loss: 0.5059
    Epoch 13/20
    11610/11610 [==============================] - 1s 97us/sample - loss: 0.4903 - val_loss: 0.5125
    Epoch 14/20
    11610/11610 [==============================] - 1s 95us/sample - loss: 0.4844 - val_loss: 0.4932
    Epoch 15/20
    11610/11610 [==============================] - 1s 95us/sample - loss: 0.4792 - val_loss: 0.4877
    Epoch 16/20
    11610/11610 [==============================] - 1s 93us/sample - loss: 0.4740 - val_loss: 0.4954
    Epoch 17/20
    11610/11610 [==============================] - 1s 95us/sample - loss: 0.4710 - val_loss: 0.4982
    Epoch 18/20
    11610/11610 [==============================] - 1s 97us/sample - loss: 0.4673 - val_loss: 0.4795
    Epoch 19/20
    11610/11610 [==============================] - 1s 94us/sample - loss: 0.4627 - val_loss: 0.4783
    Epoch 20/20
    11610/11610 [==============================] - 1s 98us/sample - loss: 0.4592 - val_loss: 0.4700



```python
model.evaluate(split_input_matrices(X_test), y_test, verbose=0)
```




    0.4853714690189953




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

    Model: "model_2"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    deep_input (InputLayer)         [(None, 6)]          0                                            
    __________________________________________________________________________________________________
    dense_8 (Dense)                 (None, 30)           210         deep_input[0][0]                 
    __________________________________________________________________________________________________
    wide_input (InputLayer)         [(None, 5)]          0                                            
    __________________________________________________________________________________________________
    dense_9 (Dense)                 (None, 100)          3100        dense_8[0][0]                    
    __________________________________________________________________________________________________
    concatenate_2 (Concatenate)     (None, 105)          0           wide_input[0][0]                 
                                                                     dense_9[0][0]                    
    __________________________________________________________________________________________________
    output (Dense)                  (None, 1)            106         concatenate_2[0][0]              
    __________________________________________________________________________________________________
    aux_output (Dense)              (None, 1)            101         dense_9[0][0]                    
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
    11610/11610 [==============================] - 2s 210us/sample - loss: 1.0211 - output_loss: 0.9023 - aux_output_loss: 2.0904 - val_loss: 0.5812 - val_output_loss: 0.5272 - val_aux_output_loss: 1.0675
    Epoch 2/20
    11610/11610 [==============================] - 1s 112us/sample - loss: 0.5320 - output_loss: 0.4827 - aux_output_loss: 0.9762 - val_loss: 0.5017 - val_output_loss: 0.4623 - val_aux_output_loss: 0.8570
    Epoch 3/20
    11610/11610 [==============================] - 1s 110us/sample - loss: 0.4772 - output_loss: 0.4409 - aux_output_loss: 0.8032 - val_loss: 0.4799 - val_output_loss: 0.4500 - val_aux_output_loss: 0.7493
    Epoch 4/20
    11610/11610 [==============================] - 1s 110us/sample - loss: 0.4675 - output_loss: 0.4403 - aux_output_loss: 0.7140 - val_loss: 0.4518 - val_output_loss: 0.4264 - val_aux_output_loss: 0.6814
    Epoch 5/20
    11610/11610 [==============================] - 1s 111us/sample - loss: 0.4396 - output_loss: 0.4155 - aux_output_loss: 0.6577 - val_loss: 0.4585 - val_output_loss: 0.4374 - val_aux_output_loss: 0.6487
    Epoch 6/20
    11610/11610 [==============================] - 1s 113us/sample - loss: 0.4346 - output_loss: 0.4137 - aux_output_loss: 0.6231 - val_loss: 0.4353 - val_output_loss: 0.4134 - val_aux_output_loss: 0.6326
    Epoch 7/20
    11610/11610 [==============================] - 1s 122us/sample - loss: 0.4178 - output_loss: 0.3973 - aux_output_loss: 0.6025 - val_loss: 0.4266 - val_output_loss: 0.4057 - val_aux_output_loss: 0.6153
    Epoch 8/20
    11610/11610 [==============================] - 1s 112us/sample - loss: 0.4202 - output_loss: 0.4018 - aux_output_loss: 0.5869 - val_loss: 0.4160 - val_output_loss: 0.3964 - val_aux_output_loss: 0.5930
    Epoch 9/20
    11610/11610 [==============================] - 1s 116us/sample - loss: 0.4069 - output_loss: 0.3883 - aux_output_loss: 0.5743 - val_loss: 0.4139 - val_output_loss: 0.3955 - val_aux_output_loss: 0.5798
    Epoch 10/20
    11610/11610 [==============================] - 1s 113us/sample - loss: 0.4005 - output_loss: 0.3826 - aux_output_loss: 0.5617 - val_loss: 0.4031 - val_output_loss: 0.3847 - val_aux_output_loss: 0.5689
    Epoch 11/20
    11610/11610 [==============================] - 1s 112us/sample - loss: 0.4023 - output_loss: 0.3855 - aux_output_loss: 0.5530 - val_loss: 0.4005 - val_output_loss: 0.3828 - val_aux_output_loss: 0.5605
    Epoch 12/20
    11610/11610 [==============================] - 1s 115us/sample - loss: 0.3907 - output_loss: 0.3739 - aux_output_loss: 0.5414 - val_loss: 0.4007 - val_output_loss: 0.3815 - val_aux_output_loss: 0.5741
    Epoch 13/20
    11610/11610 [==============================] - 1s 113us/sample - loss: 0.3824 - output_loss: 0.3654 - aux_output_loss: 0.5357 - val_loss: 0.3891 - val_output_loss: 0.3714 - val_aux_output_loss: 0.5494
    Epoch 14/20
    11610/11610 [==============================] - 1s 112us/sample - loss: 0.3818 - output_loss: 0.3656 - aux_output_loss: 0.5270 - val_loss: 0.3914 - val_output_loss: 0.3747 - val_aux_output_loss: 0.5422
    Epoch 15/20
    11610/11610 [==============================] - 1s 117us/sample - loss: 0.3958 - output_loss: 0.3817 - aux_output_loss: 0.5229 - val_loss: 0.3930 - val_output_loss: 0.3764 - val_aux_output_loss: 0.5426
    Epoch 16/20
    11610/11610 [==============================] - 1s 111us/sample - loss: 0.3693 - output_loss: 0.3530 - aux_output_loss: 0.5153 - val_loss: 0.3771 - val_output_loss: 0.3602 - val_aux_output_loss: 0.5303
    Epoch 17/20
    11610/11610 [==============================] - 1s 113us/sample - loss: 0.3679 - output_loss: 0.3525 - aux_output_loss: 0.5063 - val_loss: 0.3747 - val_output_loss: 0.3580 - val_aux_output_loss: 0.5259
    Epoch 18/20
    11610/11610 [==============================] - 1s 113us/sample - loss: 0.3652 - output_loss: 0.3502 - aux_output_loss: 0.5003 - val_loss: 0.3698 - val_output_loss: 0.3539 - val_aux_output_loss: 0.5133
    Epoch 19/20
    11610/11610 [==============================] - 1s 107us/sample - loss: 0.3661 - output_loss: 0.3518 - aux_output_loss: 0.4942 - val_loss: 0.4021 - val_output_loss: 0.3900 - val_aux_output_loss: 0.5107
    Epoch 20/20
    11610/11610 [==============================] - 1s 106us/sample - loss: 0.3603 - output_loss: 0.3460 - aux_output_loss: 0.4880 - val_loss: 0.3746 - val_output_loss: 0.3578 - val_aux_output_loss: 0.5253



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


### Using the subclassing API to build dynamic models

The Sequential and Functional APIs are declariative and, thus, have a static model graph.
This has many advantages because TF can display and analyze the graph.
However, this means it has limited flebibility.

The Subclassing API adds limitless flexibility.
The main steps are to subclass the `Model` class, create the layers in the constructor, and use them to perform any computations in the `call()` method.
The `call()` method can do anything the programming likes, including using for-loops or logical branching.

Below is an implementation of the *Wide & Deep* model from above using the Subclassing API.


```python
class WideAndDeep(keras.Model):
    """
    The 'Wide & Deep' ANN using the subclassing Keras API.
    """

    def __init__(self, units=30, activation='relu', **kwargs):
        super().__init__(**kwargs)  # handles standard argumentas such as `name`
        self.hidden1 = keras.layers.Dense(units, activation=activation)
        self.hidden2 = keras.layers.Dense(units, activation=activation)
        self.main_output = keras.layers.Dense(1)
        self.aut_output = keras.layers.Dense(1)

    def call(self, inputs):
        input_A, input_B = inputs
        hidden1 = self.hidden1(input_B)
        hidden2 = self.hidden2(hidden1)
        concat = keras.layers.concatenate([input_A, hidden2])
        main_output = self.main_output(concat)
        aux_output = self.aut_output(hidden2)
        return main_output, aux_output


# Create the model object, compile, and fit.
model = WideAndDeep()
model.compile(loss=['mse', 'mse'],
              loss_weights=[0.9, 0.1],
              optimizer='sgd')

history = model.fit(split_input_matrices(X_train),
                    (y_train, y_train),
                    epochs=20,
                    validation_data=(
                        split_input_matrices(X_valid),
                        (y_valid, y_valid)
))

# Plot the progress of the training.
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.show()
```

    Train on 11610 samples, validate on 3870 samples
    Epoch 1/20
    11610/11610 [==============================] - 2s 211us/sample - loss: 0.9067 - output_1_loss: 0.7862 - output_2_loss: 1.9906 - val_loss: 0.6329 - val_output_1_loss: 0.5529 - val_output_2_loss: 1.3543
    Epoch 2/20
    11610/11610 [==============================] - 1s 106us/sample - loss: 0.6313 - output_1_loss: 0.5556 - output_2_loss: 1.3108 - val_loss: 0.5660 - val_output_1_loss: 0.4984 - val_output_2_loss: 1.1756
    Epoch 3/20
    11610/11610 [==============================] - 2s 133us/sample - loss: 0.5602 - output_1_loss: 0.4951 - output_2_loss: 1.1465 - val_loss: 0.5236 - val_output_1_loss: 0.4713 - val_output_2_loss: 0.9946
    Epoch 4/20
    11610/11610 [==============================] - 2s 135us/sample - loss: 0.5182 - output_1_loss: 0.4662 - output_2_loss: 0.9852 - val_loss: 0.4949 - val_output_1_loss: 0.4528 - val_output_2_loss: 0.8743
    Epoch 5/20
    11610/11610 [==============================] - 1s 129us/sample - loss: 0.4791 - output_1_loss: 0.4364 - output_2_loss: 0.8616 - val_loss: 0.4767 - val_output_1_loss: 0.4427 - val_output_2_loss: 0.7835
    Epoch 6/20
    11610/11610 [==============================] - 2s 133us/sample - loss: 0.4640 - output_1_loss: 0.4286 - output_2_loss: 0.7812 - val_loss: 0.4519 - val_output_1_loss: 0.4215 - val_output_2_loss: 0.7257
    Epoch 7/20
    11610/11610 [==============================] - 1s 128us/sample - loss: 0.4529 - output_1_loss: 0.4228 - output_2_loss: 0.7237 - val_loss: 0.4552 - val_output_1_loss: 0.4297 - val_output_2_loss: 0.6856
    Epoch 8/20
    11610/11610 [==============================] - 1s 128us/sample - loss: 0.4381 - output_1_loss: 0.4109 - output_2_loss: 0.6821 - val_loss: 0.4426 - val_output_1_loss: 0.4179 - val_output_2_loss: 0.6658
    Epoch 9/20
    11610/11610 [==============================] - 1s 128us/sample - loss: 0.4285 - output_1_loss: 0.4037 - output_2_loss: 0.6524 - val_loss: 0.4415 - val_output_1_loss: 0.4196 - val_output_2_loss: 0.6388
    Epoch 10/20
    11610/11610 [==============================] - 2s 129us/sample - loss: 0.4241 - output_1_loss: 0.4015 - output_2_loss: 0.6292 - val_loss: 0.4566 - val_output_1_loss: 0.4338 - val_output_2_loss: 0.6622
    Epoch 11/20
    11610/11610 [==============================] - 1s 118us/sample - loss: 0.4250 - output_1_loss: 0.4040 - output_2_loss: 0.6129 - val_loss: 0.4507 - val_output_1_loss: 0.4298 - val_output_2_loss: 0.6392
    Epoch 12/20
    11610/11610 [==============================] - 1s 128us/sample - loss: 0.4104 - output_1_loss: 0.3899 - output_2_loss: 0.5939 - val_loss: 0.4107 - val_output_1_loss: 0.3905 - val_output_2_loss: 0.5929
    Epoch 13/20
    11610/11610 [==============================] - 1s 128us/sample - loss: 0.4109 - output_1_loss: 0.3919 - output_2_loss: 0.5832 - val_loss: 0.4630 - val_output_1_loss: 0.4459 - val_output_2_loss: 0.6171
    Epoch 14/20
    11610/11610 [==============================] - 1s 128us/sample - loss: 0.4038 - output_1_loss: 0.3856 - output_2_loss: 0.5671 - val_loss: 0.4024 - val_output_1_loss: 0.3835 - val_output_2_loss: 0.5724
    Epoch 15/20
    11610/11610 [==============================] - 2s 134us/sample - loss: 0.3979 - output_1_loss: 0.3802 - output_2_loss: 0.5570 - val_loss: 0.3957 - val_output_1_loss: 0.3774 - val_output_2_loss: 0.5607
    Epoch 16/20
    11610/11610 [==============================] - 2s 133us/sample - loss: 0.3864 - output_1_loss: 0.3693 - output_2_loss: 0.5405 - val_loss: 0.3931 - val_output_1_loss: 0.3756 - val_output_2_loss: 0.5517
    Epoch 17/20
    11610/11610 [==============================] - 1s 115us/sample - loss: 0.3859 - output_1_loss: 0.3701 - output_2_loss: 0.5279 - val_loss: 0.3857 - val_output_1_loss: 0.3690 - val_output_2_loss: 0.5367
    Epoch 18/20
    11610/11610 [==============================] - 2s 132us/sample - loss: 0.3885 - output_1_loss: 0.3736 - output_2_loss: 0.5216 - val_loss: 0.3881 - val_output_1_loss: 0.3693 - val_output_2_loss: 0.5573
    Epoch 19/20
    11610/11610 [==============================] - 1s 124us/sample - loss: 0.3696 - output_1_loss: 0.3542 - output_2_loss: 0.5089 - val_loss: 0.3776 - val_output_1_loss: 0.3609 - val_output_2_loss: 0.5281
    Epoch 20/20
    11610/11610 [==============================] - 1s 104us/sample - loss: 0.3653 - output_1_loss: 0.3501 - output_2_loss: 0.5002 - val_loss: 0.3928 - val_output_1_loss: 0.3750 - val_output_2_loss: 0.5528



![png](homl_ch10_Introduction-to-artificial-neural-networks-with-keras_files/homl_ch10_Introduction-to-artificial-neural-networks-with-keras_76_1.png)


In general, it is safer and easier to use the Sequential and Functional Keras APIs.

### Saving and Restoring a Model

Saving a model created using the Sequential or Functional APIs is simple and shown below.


```python
# ** An example model built using the functional API ** #
input_ = keras.layers.Input(shape=X_train.shape[1:])
hidden1 = keras.layers.Dense(30, activation='relu')(input_)
hidden2 = keras.layers.Dense(100, activation='relu')(hidden1)
concat = keras.layers.Concatenate()([input_, hidden2])
output = keras.layers.Dense(1, activation='relu')(concat)
model = keras.Model(inputs=[input_], outputs=[output])

model.compile(loss='mean_squared_error', optimizer='sgd')

history = model.fit(X_train, y_train,
                    epochs=20,
                    validation_data=(X_valid, y_valid))

model.summary()
```

    Train on 11610 samples, validate on 3870 samples
    Epoch 1/20
    11610/11610 [==============================] - 2s 171us/sample - loss: 0.8478 - val_loss: 0.4696
    Epoch 2/20
    11610/11610 [==============================] - 1s 96us/sample - loss: 0.4339 - val_loss: 0.4445
    Epoch 3/20
    11610/11610 [==============================] - 1s 96us/sample - loss: 0.4091 - val_loss: 0.4279
    Epoch 4/20
    11610/11610 [==============================] - 1s 96us/sample - loss: 0.4000 - val_loss: 0.4103
    Epoch 5/20
    11610/11610 [==============================] - 1s 96us/sample - loss: 0.3953 - val_loss: 0.4073
    Epoch 6/20
    11610/11610 [==============================] - 1s 97us/sample - loss: 0.3847 - val_loss: 0.4041
    Epoch 7/20
    11610/11610 [==============================] - 1s 97us/sample - loss: 0.3765 - val_loss: 0.3991
    Epoch 8/20
    11610/11610 [==============================] - 1s 96us/sample - loss: 0.3720 - val_loss: 0.3991
    Epoch 9/20
    11610/11610 [==============================] - 1s 96us/sample - loss: 0.3826 - val_loss: 0.4135
    Epoch 10/20
    11610/11610 [==============================] - 1s 100us/sample - loss: 0.3698 - val_loss: 0.3955
    Epoch 11/20
    11610/11610 [==============================] - 1s 98us/sample - loss: 0.3653 - val_loss: 0.3919
    Epoch 12/20
    11610/11610 [==============================] - 1s 99us/sample - loss: 0.3592 - val_loss: 0.3952
    Epoch 13/20
    11610/11610 [==============================] - 1s 110us/sample - loss: 0.3580 - val_loss: 0.3869
    Epoch 14/20
    11610/11610 [==============================] - 1s 108us/sample - loss: 0.3539 - val_loss: 0.3834
    Epoch 15/20
    11610/11610 [==============================] - 1s 107us/sample - loss: 0.3499 - val_loss: 0.3953
    Epoch 16/20
    11610/11610 [==============================] - 1s 101us/sample - loss: 0.3481 - val_loss: 0.3786
    Epoch 17/20
    11610/11610 [==============================] - 1s 102us/sample - loss: 0.3797 - val_loss: 0.4192
    Epoch 18/20
    11610/11610 [==============================] - 1s 101us/sample - loss: 0.3603 - val_loss: 0.3847
    Epoch 19/20
    11610/11610 [==============================] - 1s 107us/sample - loss: 0.3452 - val_loss: 0.3742
    Epoch 20/20
    11610/11610 [==============================] - 1s 102us/sample - loss: 0.3436 - val_loss: 0.3681
    Model: "model_3"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_2 (InputLayer)            [(None, 8)]          0                                            
    __________________________________________________________________________________________________
    dense_14 (Dense)                (None, 30)           270         input_2[0][0]                    
    __________________________________________________________________________________________________
    dense_15 (Dense)                (None, 100)          3100        dense_14[0][0]                   
    __________________________________________________________________________________________________
    concatenate_3 (Concatenate)     (None, 108)          0           input_2[0][0]                    
                                                                     dense_15[0][0]                   
    __________________________________________________________________________________________________
    dense_16 (Dense)                (None, 1)            109         concatenate_3[0][0]              
    ==================================================================================================
    Total params: 3,479
    Trainable params: 3,479
    Non-trainable params: 0
    __________________________________________________________________________________________________



```python
import pathlib

# Save the model.
assets_path = pathlib.Path('assets/ch10')
model.save(assets_path.joinpath('my_keras_model.h5'))
```

Then, the model can be loaded using the `load()` method.


```python
model = keras.models.load_model(assets_path.joinpath('my_keras_model.h5'))
```

The process is not very easy if the model was built with the Subclassing API.
The programmer would have to save and load the model parameters with `save_weights()` and `load_weights()` explcitly, and then recreate the model manually.

### Using Callbacks

The `fit()` method accepts a `callback` argument that is a list of `callback` objects that get called during each round (the point at which they get called can be specified, too).
They are useful for when a model may take a long time to train and you want to save intermediates such that they can be re-loaded if the computer crashes midway.
Another useful case is to set `save_bast_only=True` for a `ModelCheckpoint` callback object to that only the best model on the validation data is saved.
This can help prevent overfitting by essentially implementing early-stopping.


```python
# Create the model.
model = keras.Model(inputs=[input_], outputs=[output])

# Compile the model.
model.compile(loss='mean_squared_error', optimizer='sgd')

# Add a checkpoint to save the model on each round of training.
# The model is only saved if it improves upon the previous on the
#   validation data set.
checkpoint_path = assets_path.joinpath('my_keras_model_chkpt.h5')
checkpoint_cb = keras.callbacks.ModelCheckpoint(checkpoint_path.as_posix(),
                                                save_best_only=True)

# Fit the model.
history = model.fit(X_train, y_train,
                    epochs=20,
                    validation_data=(X_valid, y_valid),
                    callbacks=[checkpoint_cb])
```

    Train on 11610 samples, validate on 3870 samples
    Epoch 1/20
    11610/11610 [==============================] - 2s 162us/sample - loss: 0.3388 - val_loss: 0.3708
    Epoch 2/20
    11610/11610 [==============================] - 1s 102us/sample - loss: 0.3364 - val_loss: 0.3672
    Epoch 3/20
    11610/11610 [==============================] - 1s 102us/sample - loss: 0.3345 - val_loss: 0.3708
    Epoch 4/20
    11610/11610 [==============================] - 1s 102us/sample - loss: 0.3321 - val_loss: 0.3617
    Epoch 5/20
    11610/11610 [==============================] - 1s 101us/sample - loss: 0.3291 - val_loss: 0.3620
    Epoch 6/20
    11610/11610 [==============================] - 1s 109us/sample - loss: 0.3280 - val_loss: 0.3672
    Epoch 7/20
    11610/11610 [==============================] - 1s 107us/sample - loss: 0.3258 - val_loss: 0.3624
    Epoch 8/20
    11610/11610 [==============================] - 1s 102us/sample - loss: 0.3254 - val_loss: 0.3632
    Epoch 9/20
    11610/11610 [==============================] - 1s 103us/sample - loss: 0.3225 - val_loss: 0.3508
    Epoch 10/20
    11610/11610 [==============================] - 1s 102us/sample - loss: 0.3210 - val_loss: 0.3513
    Epoch 11/20
    11610/11610 [==============================] - 1s 101us/sample - loss: 0.3189 - val_loss: 0.3733
    Epoch 12/20
    11610/11610 [==============================] - 1s 103us/sample - loss: 0.3166 - val_loss: 0.3469
    Epoch 13/20
    11610/11610 [==============================] - 1s 104us/sample - loss: 0.3157 - val_loss: 0.3592
    Epoch 14/20
    11610/11610 [==============================] - 1s 101us/sample - loss: 0.3144 - val_loss: 0.3485
    Epoch 15/20
    11610/11610 [==============================] - 1s 106us/sample - loss: 0.3105 - val_loss: 0.3541
    Epoch 16/20
    11610/11610 [==============================] - 1s 107us/sample - loss: 0.3088 - val_loss: 0.3446
    Epoch 17/20
    11610/11610 [==============================] - 1s 107us/sample - loss: 0.3087 - val_loss: 0.3432
    Epoch 18/20
    11610/11610 [==============================] - 1s 114us/sample - loss: 0.3067 - val_loss: 0.3455
    Epoch 19/20
    11610/11610 [==============================] - 1s 109us/sample - loss: 0.3072 - val_loss: 0.3398
    Epoch 20/20
    11610/11610 [==============================] - 1s 103us/sample - loss: 0.3036 - val_loss: 0.3373


Alternatively, the `EarlyStopping` callback can be used.
A common implementation is to use both `ModelCheckpoint` and `EarlyStopping` together, the first to save intermediate models in case of a crash, the latter to prevent unnecessary time spent on training.
The number of epochs can also be increased because early-stopping will prevent overfitting and excessive training..


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


# Add a checkpoint to save the model on each round of training.
# The model is only saved if it improves upon the previous on the
#   validation data set.
checkpoint_path = assets_path.joinpath('my_keras_model_chkpt.h5')
checkpoint_cb = keras.callbacks.ModelCheckpoint(checkpoint_path.as_posix(),
                                                save_best_only=True)

# Implement easly stopping after no improvement for 5 epochs.
early_stopping_cb = keras.callbacks.EarlyStopping(patience=5,
                                                  restore_best_weights=True)

model.compile(loss=['mse', 'mse'],
              loss_weights=[0.9, 0.1],
              optimizer='sgd')

history = model.fit(split_input_matrices(X_train),
                    (y_train, y_train),
                    epochs=100,
                    validation_data=(
                        split_input_matrices(X_valid),
                        (y_valid, y_valid)
                    ),
                    callbacks=[checkpoint_cb, early_stopping_cb])
```

    Train on 11610 samples, validate on 3870 samples
    Epoch 1/100
    11610/11610 [==============================] - 3s 238us/sample - loss: 1.0470 - output_loss: 0.9302 - aux_output_loss: 2.0948 - val_loss: 0.6145 - val_output_loss: 0.5463 - val_aux_output_loss: 1.2302
    Epoch 2/100
    11610/11610 [==============================] - 1s 125us/sample - loss: 0.6306 - output_loss: 0.5695 - aux_output_loss: 1.1828 - val_loss: 0.5389 - val_output_loss: 0.4841 - val_aux_output_loss: 1.0337
    Epoch 3/100
    11610/11610 [==============================] - 1s 120us/sample - loss: 0.5229 - output_loss: 0.4710 - aux_output_loss: 0.9895 - val_loss: 0.4987 - val_output_loss: 0.4552 - val_aux_output_loss: 0.8904
    Epoch 4/100
    11610/11610 [==============================] - 1s 120us/sample - loss: 0.4771 - output_loss: 0.4362 - aux_output_loss: 0.8451 - val_loss: 0.4718 - val_output_loss: 0.4372 - val_aux_output_loss: 0.7841
    Epoch 5/100
    11610/11610 [==============================] - 2s 167us/sample - loss: 0.4556 - output_loss: 0.4236 - aux_output_loss: 0.7424 - val_loss: 0.4523 - val_output_loss: 0.4229 - val_aux_output_loss: 0.7168
    Epoch 6/100
    11610/11610 [==============================] - 2s 145us/sample - loss: 0.4396 - output_loss: 0.4135 - aux_output_loss: 0.6762 - val_loss: 0.4535 - val_output_loss: 0.4294 - val_aux_output_loss: 0.6714
    Epoch 7/100
    11610/11610 [==============================] - 1s 96us/sample - loss: 0.4311 - output_loss: 0.4083 - aux_output_loss: 0.6370 - val_loss: 0.4381 - val_output_loss: 0.4153 - val_aux_output_loss: 0.6434
    Epoch 8/100
    11610/11610 [==============================] - 1s 101us/sample - loss: 0.4195 - output_loss: 0.3986 - aux_output_loss: 0.6068 - val_loss: 0.4314 - val_output_loss: 0.4104 - val_aux_output_loss: 0.6212
    Epoch 9/100
    11610/11610 [==============================] - 1s 119us/sample - loss: 0.4153 - output_loss: 0.3961 - aux_output_loss: 0.5877 - val_loss: 0.4165 - val_output_loss: 0.3950 - val_aux_output_loss: 0.6104
    Epoch 10/100
    11610/11610 [==============================] - 1s 99us/sample - loss: 0.4076 - output_loss: 0.3895 - aux_output_loss: 0.5707 - val_loss: 0.4107 - val_output_loss: 0.3909 - val_aux_output_loss: 0.5894
    Epoch 11/100
    11610/11610 [==============================] - 1s 101us/sample - loss: 0.3992 - output_loss: 0.3817 - aux_output_loss: 0.5561 - val_loss: 0.4078 - val_output_loss: 0.3886 - val_aux_output_loss: 0.5810
    Epoch 12/100
    11610/11610 [==============================] - 1s 97us/sample - loss: 0.3960 - output_loss: 0.3795 - aux_output_loss: 0.5448 - val_loss: 0.4021 - val_output_loss: 0.3840 - val_aux_output_loss: 0.5651
    Epoch 13/100
    11610/11610 [==============================] - 1s 104us/sample - loss: 0.3886 - output_loss: 0.3726 - aux_output_loss: 0.5335 - val_loss: 0.3924 - val_output_loss: 0.3757 - val_aux_output_loss: 0.5425
    Epoch 14/100
    11610/11610 [==============================] - 1s 98us/sample - loss: 0.3820 - output_loss: 0.3669 - aux_output_loss: 0.5198 - val_loss: 0.4103 - val_output_loss: 0.3957 - val_aux_output_loss: 0.5416
    Epoch 15/100
    11610/11610 [==============================] - 1s 98us/sample - loss: 0.3782 - output_loss: 0.3635 - aux_output_loss: 0.5105 - val_loss: 0.3867 - val_output_loss: 0.3695 - val_aux_output_loss: 0.5422
    Epoch 16/100
    11610/11610 [==============================] - 1s 100us/sample - loss: 0.3744 - output_loss: 0.3603 - aux_output_loss: 0.5011 - val_loss: 0.3788 - val_output_loss: 0.3629 - val_aux_output_loss: 0.5218
    Epoch 17/100
    11610/11610 [==============================] - 1s 102us/sample - loss: 0.3678 - output_loss: 0.3539 - aux_output_loss: 0.4925 - val_loss: 0.3775 - val_output_loss: 0.3619 - val_aux_output_loss: 0.5181
    Epoch 18/100
    11610/11610 [==============================] - 1s 122us/sample - loss: 0.3641 - output_loss: 0.3508 - aux_output_loss: 0.4842 - val_loss: 0.3764 - val_output_loss: 0.3607 - val_aux_output_loss: 0.5183
    Epoch 19/100
    11610/11610 [==============================] - 2s 159us/sample - loss: 0.3603 - output_loss: 0.3472 - aux_output_loss: 0.4780 - val_loss: 0.3712 - val_output_loss: 0.3570 - val_aux_output_loss: 0.4993
    Epoch 20/100
    11610/11610 [==============================] - 1s 103us/sample - loss: 0.3568 - output_loss: 0.3441 - aux_output_loss: 0.4708 - val_loss: 0.3670 - val_output_loss: 0.3522 - val_aux_output_loss: 0.5005
    Epoch 21/100
    11610/11610 [==============================] - 2s 137us/sample - loss: 0.3575 - output_loss: 0.3454 - aux_output_loss: 0.4656 - val_loss: 0.3630 - val_output_loss: 0.3493 - val_aux_output_loss: 0.4862
    Epoch 22/100
    11610/11610 [==============================] - 3s 235us/sample - loss: 0.3538 - output_loss: 0.3420 - aux_output_loss: 0.4604 - val_loss: 0.3598 - val_output_loss: 0.3471 - val_aux_output_loss: 0.4753
    Epoch 23/100
    11610/11610 [==============================] - 1s 125us/sample - loss: 0.3473 - output_loss: 0.3355 - aux_output_loss: 0.4536 - val_loss: 0.3561 - val_output_loss: 0.3432 - val_aux_output_loss: 0.4731
    Epoch 24/100
    11610/11610 [==============================] - 1s 129us/sample - loss: 0.3482 - output_loss: 0.3369 - aux_output_loss: 0.4497 - val_loss: 0.3553 - val_output_loss: 0.3422 - val_aux_output_loss: 0.4743
    Epoch 25/100
    11610/11610 [==============================] - 2s 142us/sample - loss: 0.3445 - output_loss: 0.3333 - aux_output_loss: 0.4453 - val_loss: 0.3572 - val_output_loss: 0.3455 - val_aux_output_loss: 0.4637
    Epoch 26/100
    11610/11610 [==============================] - 2s 161us/sample - loss: 0.3448 - output_loss: 0.3341 - aux_output_loss: 0.4414 - val_loss: 0.3642 - val_output_loss: 0.3526 - val_aux_output_loss: 0.4691
    Epoch 27/100
    11610/11610 [==============================] - 2s 174us/sample - loss: 0.3426 - output_loss: 0.3322 - aux_output_loss: 0.4364 - val_loss: 0.3664 - val_output_loss: 0.3553 - val_aux_output_loss: 0.4667
    Epoch 28/100
    11610/11610 [==============================] - 2s 154us/sample - loss: 0.3391 - output_loss: 0.3286 - aux_output_loss: 0.4338 - val_loss: 0.3482 - val_output_loss: 0.3362 - val_aux_output_loss: 0.4564
    Epoch 29/100
    11610/11610 [==============================] - 2s 131us/sample - loss: 0.3398 - output_loss: 0.3296 - aux_output_loss: 0.4306 - val_loss: 0.3518 - val_output_loss: 0.3399 - val_aux_output_loss: 0.4592
    Epoch 30/100
    11610/11610 [==============================] - 2s 160us/sample - loss: 0.3417 - output_loss: 0.3323 - aux_output_loss: 0.4259 - val_loss: 0.3531 - val_output_loss: 0.3418 - val_aux_output_loss: 0.4555
    Epoch 31/100
    11610/11610 [==============================] - 2s 173us/sample - loss: 0.3340 - output_loss: 0.3240 - aux_output_loss: 0.4233 - val_loss: 0.3479 - val_output_loss: 0.3367 - val_aux_output_loss: 0.4494
    Epoch 32/100
    11610/11610 [==============================] - 2s 172us/sample - loss: 0.3348 - output_loss: 0.3254 - aux_output_loss: 0.4207 - val_loss: 0.3452 - val_output_loss: 0.3341 - val_aux_output_loss: 0.4455
    Epoch 33/100
    11610/11610 [==============================] - 2s 134us/sample - loss: 0.3349 - output_loss: 0.3257 - aux_output_loss: 0.4184 - val_loss: 0.3848 - val_output_loss: 0.3767 - val_aux_output_loss: 0.4583
    Epoch 34/100
    11610/11610 [==============================] - 1s 117us/sample - loss: 0.3335 - output_loss: 0.3244 - aux_output_loss: 0.4165 - val_loss: 0.3474 - val_output_loss: 0.3371 - val_aux_output_loss: 0.4411
    Epoch 35/100
    11610/11610 [==============================] - 1s 100us/sample - loss: 0.3312 - output_loss: 0.3222 - aux_output_loss: 0.4134 - val_loss: 0.3419 - val_output_loss: 0.3304 - val_aux_output_loss: 0.4457
    Epoch 36/100
    11610/11610 [==============================] - 1s 112us/sample - loss: 0.3279 - output_loss: 0.3188 - aux_output_loss: 0.4108 - val_loss: 0.3460 - val_output_loss: 0.3344 - val_aux_output_loss: 0.4513
    Epoch 37/100
    11610/11610 [==============================] - 1s 102us/sample - loss: 0.3318 - output_loss: 0.3229 - aux_output_loss: 0.4115 - val_loss: 0.3428 - val_output_loss: 0.3328 - val_aux_output_loss: 0.4332
    Epoch 38/100
    11610/11610 [==============================] - 1s 114us/sample - loss: 0.3291 - output_loss: 0.3201 - aux_output_loss: 0.4088 - val_loss: 0.3399 - val_output_loss: 0.3297 - val_aux_output_loss: 0.4313
    Epoch 39/100
    11610/11610 [==============================] - 1s 103us/sample - loss: 0.3266 - output_loss: 0.3175 - aux_output_loss: 0.4072 - val_loss: 0.3415 - val_output_loss: 0.3315 - val_aux_output_loss: 0.4317
    Epoch 40/100
    11610/11610 [==============================] - 1s 95us/sample - loss: 0.3289 - output_loss: 0.3204 - aux_output_loss: 0.4053 - val_loss: 0.3386 - val_output_loss: 0.3283 - val_aux_output_loss: 0.4319
    Epoch 41/100
    11610/11610 [==============================] - 1s 97us/sample - loss: 0.3251 - output_loss: 0.3165 - aux_output_loss: 0.4031 - val_loss: 0.3461 - val_output_loss: 0.3352 - val_aux_output_loss: 0.4443
    Epoch 42/100
    11610/11610 [==============================] - 1s 102us/sample - loss: 0.3288 - output_loss: 0.3206 - aux_output_loss: 0.4033 - val_loss: 0.3443 - val_output_loss: 0.3343 - val_aux_output_loss: 0.4354
    Epoch 43/100
    11610/11610 [==============================] - 1s 92us/sample - loss: 0.3245 - output_loss: 0.3161 - aux_output_loss: 0.3996 - val_loss: 0.3355 - val_output_loss: 0.3257 - val_aux_output_loss: 0.4240
    Epoch 44/100
    11610/11610 [==============================] - 1s 92us/sample - loss: 0.3227 - output_loss: 0.3144 - aux_output_loss: 0.3984 - val_loss: 0.3355 - val_output_loss: 0.3258 - val_aux_output_loss: 0.4225
    Epoch 45/100
    11610/11610 [==============================] - 1s 91us/sample - loss: 0.3224 - output_loss: 0.3139 - aux_output_loss: 0.3970 - val_loss: 0.3374 - val_output_loss: 0.3278 - val_aux_output_loss: 0.4248
    Epoch 46/100
    11610/11610 [==============================] - 1s 101us/sample - loss: 0.3215 - output_loss: 0.3132 - aux_output_loss: 0.3961 - val_loss: 0.3313 - val_output_loss: 0.3220 - val_aux_output_loss: 0.4161
    Epoch 47/100
    11610/11610 [==============================] - 1s 103us/sample - loss: 0.3212 - output_loss: 0.3132 - aux_output_loss: 0.3940 - val_loss: 0.3537 - val_output_loss: 0.3445 - val_aux_output_loss: 0.4369
    Epoch 48/100
    11610/11610 [==============================] - 1s 114us/sample - loss: 0.3234 - output_loss: 0.3154 - aux_output_loss: 0.3947 - val_loss: 0.3294 - val_output_loss: 0.3202 - val_aux_output_loss: 0.4134
    Epoch 49/100
    11610/11610 [==============================] - 1s 112us/sample - loss: 0.3203 - output_loss: 0.3123 - aux_output_loss: 0.3914 - val_loss: 0.3335 - val_output_loss: 0.3236 - val_aux_output_loss: 0.4229
    Epoch 50/100
    11610/11610 [==============================] - 1s 104us/sample - loss: 0.3206 - output_loss: 0.3127 - aux_output_loss: 0.3912 - val_loss: 0.3347 - val_output_loss: 0.3253 - val_aux_output_loss: 0.4202
    Epoch 51/100
    11610/11610 [==============================] - 2s 169us/sample - loss: 0.3180 - output_loss: 0.3100 - aux_output_loss: 0.3894 - val_loss: 0.3361 - val_output_loss: 0.3272 - val_aux_output_loss: 0.4167
    Epoch 52/100
    11610/11610 [==============================] - 1s 129us/sample - loss: 0.3187 - output_loss: 0.3109 - aux_output_loss: 0.3882 - val_loss: 0.3487 - val_output_loss: 0.3404 - val_aux_output_loss: 0.4235
    Epoch 53/100
    11610/11610 [==============================] - 1s 92us/sample - loss: 0.3178 - output_loss: 0.3101 - aux_output_loss: 0.3867 - val_loss: 0.3310 - val_output_loss: 0.3218 - val_aux_output_loss: 0.4139



```python
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.show()
```


![png](homl_ch10_Introduction-to-artificial-neural-networks-with-keras_files/homl_ch10_Introduction-to-artificial-neural-networks-with-keras_86_0.png)


It is also possible to implement custom callbacks by subclassing `Callback` and defining methods with names such as `on_epoch_begin()`, `on_train_begin()`, and more.
Below is an example that logs the validation over training error on each epoch to detect overfitting.


```python
class PrintValidationTrainingRatioCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        print(f'\nval/train: {logs["val_loss"] / logs["loss"]}')
```

### Using Tensorboard for Visualization

Tensorboard is a powerful visualization tool for Tensorflow models.
To run it, we must save special binary log files called *event files* to a subdirectory in a root log folder.
It is best to save each record (called a *summary*) to s unique subdirectory so it can be inspected in the future.

Below is a function that creates a new subdirectory based on the date and time.


```python
from pathlib import Path
root_logdir = Path('tf_logs_ch10')

# Remove old examples.
!rm -rf $root_logdir/*

def get_run_logdir():
    import time
    run_id = time.strftime('run_%Y_%m_%d_%H_%M_%S')
    return root_logdir.joinpath(run_id).as_posix()
```


```python
get_run_logdir()
```




    'tf_logs_ch10/run_2020_01_04_17_20_42'



Below, I train and run two ANNs on the blobs artificial data set.


```python
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA

X_blob, y_blob = make_blobs(n_samples=2000,
                            n_features=5,
                            centers=7,
                            cluster_std=3,
                            random_state=0)

X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob,
                                                                        y_blob,
                                                                        shuffle=True,
                                                                        random_state=0)

pca = PCA(n_components=2)
blob_reduced = pca.fit_transform(X_blob)

plt.scatter(blob_reduced[:, 0], blob_reduced[:, 1],
            c=y_blob, cmap='Set2', alpha=0.5)
plt.xlabel('PC1', fontsize=12)
plt.ylabel('PC2', fontsize=12)
plt.title('5D Blobs in 2D', fontsize=14)
plt.show()
```


![png](homl_ch10_Introduction-to-artificial-neural-networks-with-keras_files/homl_ch10_Introduction-to-artificial-neural-networks-with-keras_93_0.png)



```python
from sklearn.manifold import TSNE

tsne = TSNE(random_state=0)
blob_reduced = tsne.fit_transform(X_blob)

plt.scatter(blob_reduced[:, 0], blob_reduced[:, 1],
            c=y_blob, cmap='Set2', alpha=0.5)
plt.xlabel('z1', fontsize=12)
plt.ylabel('z2', fontsize=12)
plt.title('5D Blobs in 2D (t-SNE)', fontsize=14)
plt.show()
```


![png](homl_ch10_Introduction-to-artificial-neural-networks-with-keras_files/homl_ch10_Introduction-to-artificial-neural-networks-with-keras_94_0.png)



```python
# Construct.
ann_1 = keras.models.Sequential([
    keras.layers.Input(shape=X_blob.shape[1]),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(7, activation='softmax')
])

# Compile.
ann_1.compile(loss='sparse_categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

# Callback for Tensorboard.
tensorboard_cb = keras.callbacks.TensorBoard(get_run_logdir())

# Train.
history = ann_1.fit(X_blob_train, 
          y_blob_train, 
          epochs=30, 
          validation_split=0.2, 
          callbacks=[tensorboard_cb])
```

    Train on 1200 samples, validate on 300 samples
    Epoch 1/30
    1200/1200 [==============================] - 1s 609us/sample - loss: 0.9922 - accuracy: 0.7092 - val_loss: 0.5765 - val_accuracy: 0.8200
    Epoch 2/30
    1200/1200 [==============================] - 0s 131us/sample - loss: 0.4867 - accuracy: 0.8567 - val_loss: 0.4400 - val_accuracy: 0.8700
    Epoch 3/30
    1200/1200 [==============================] - 0s 126us/sample - loss: 0.4094 - accuracy: 0.8667 - val_loss: 0.3908 - val_accuracy: 0.8867
    Epoch 4/30
    1200/1200 [==============================] - 0s 127us/sample - loss: 0.3753 - accuracy: 0.8742 - val_loss: 0.3650 - val_accuracy: 0.8800
    Epoch 5/30
    1200/1200 [==============================] - 0s 126us/sample - loss: 0.3569 - accuracy: 0.8800 - val_loss: 0.3574 - val_accuracy: 0.8900
    Epoch 6/30
    1200/1200 [==============================] - 0s 125us/sample - loss: 0.3408 - accuracy: 0.8917 - val_loss: 0.3475 - val_accuracy: 0.8833
    Epoch 7/30
    1200/1200 [==============================] - 0s 122us/sample - loss: 0.3327 - accuracy: 0.8833 - val_loss: 0.3333 - val_accuracy: 0.8700
    Epoch 8/30
    1200/1200 [==============================] - 0s 136us/sample - loss: 0.3231 - accuracy: 0.8875 - val_loss: 0.3314 - val_accuracy: 0.8933
    Epoch 9/30
    1200/1200 [==============================] - 0s 140us/sample - loss: 0.3172 - accuracy: 0.8858 - val_loss: 0.3334 - val_accuracy: 0.8967
    Epoch 10/30
    1200/1200 [==============================] - 0s 125us/sample - loss: 0.3114 - accuracy: 0.8908 - val_loss: 0.3228 - val_accuracy: 0.8867
    Epoch 11/30
    1200/1200 [==============================] - 0s 124us/sample - loss: 0.3074 - accuracy: 0.8925 - val_loss: 0.3279 - val_accuracy: 0.8933
    Epoch 12/30
    1200/1200 [==============================] - 0s 128us/sample - loss: 0.3001 - accuracy: 0.8925 - val_loss: 0.3292 - val_accuracy: 0.8733
    Epoch 13/30
    1200/1200 [==============================] - 0s 124us/sample - loss: 0.2995 - accuracy: 0.8917 - val_loss: 0.3178 - val_accuracy: 0.8867
    Epoch 14/30
    1200/1200 [==============================] - 0s 178us/sample - loss: 0.2953 - accuracy: 0.8917 - val_loss: 0.3162 - val_accuracy: 0.8833
    Epoch 15/30
    1200/1200 [==============================] - 0s 128us/sample - loss: 0.2937 - accuracy: 0.8933 - val_loss: 0.3143 - val_accuracy: 0.8867
    Epoch 16/30
    1200/1200 [==============================] - 0s 123us/sample - loss: 0.2884 - accuracy: 0.8933 - val_loss: 0.3115 - val_accuracy: 0.8833
    Epoch 17/30
    1200/1200 [==============================] - 0s 127us/sample - loss: 0.2861 - accuracy: 0.8925 - val_loss: 0.3151 - val_accuracy: 0.8933
    Epoch 18/30
    1200/1200 [==============================] - 0s 123us/sample - loss: 0.2841 - accuracy: 0.8967 - val_loss: 0.3123 - val_accuracy: 0.8867
    Epoch 19/30
    1200/1200 [==============================] - 0s 129us/sample - loss: 0.2816 - accuracy: 0.8967 - val_loss: 0.3148 - val_accuracy: 0.8833
    Epoch 20/30
    1200/1200 [==============================] - 0s 128us/sample - loss: 0.2820 - accuracy: 0.8917 - val_loss: 0.3084 - val_accuracy: 0.8867
    Epoch 21/30
    1200/1200 [==============================] - 0s 130us/sample - loss: 0.2802 - accuracy: 0.8950 - val_loss: 0.3093 - val_accuracy: 0.8867
    Epoch 22/30
    1200/1200 [==============================] - 0s 161us/sample - loss: 0.2794 - accuracy: 0.8933 - val_loss: 0.3131 - val_accuracy: 0.9000
    Epoch 23/30
    1200/1200 [==============================] - 0s 170us/sample - loss: 0.2760 - accuracy: 0.8950 - val_loss: 0.3093 - val_accuracy: 0.8867
    Epoch 24/30
    1200/1200 [==============================] - 0s 124us/sample - loss: 0.2734 - accuracy: 0.8925 - val_loss: 0.3111 - val_accuracy: 0.8800
    Epoch 25/30
    1200/1200 [==============================] - 0s 151us/sample - loss: 0.2726 - accuracy: 0.8950 - val_loss: 0.3074 - val_accuracy: 0.8833
    Epoch 26/30
    1200/1200 [==============================] - 0s 130us/sample - loss: 0.2720 - accuracy: 0.8925 - val_loss: 0.3082 - val_accuracy: 0.8833
    Epoch 27/30
    1200/1200 [==============================] - 0s 127us/sample - loss: 0.2700 - accuracy: 0.8983 - val_loss: 0.3064 - val_accuracy: 0.8900
    Epoch 28/30
    1200/1200 [==============================] - 0s 125us/sample - loss: 0.2697 - accuracy: 0.8983 - val_loss: 0.3110 - val_accuracy: 0.8933
    Epoch 29/30
    1200/1200 [==============================] - 0s 134us/sample - loss: 0.2675 - accuracy: 0.8992 - val_loss: 0.3094 - val_accuracy: 0.8933
    Epoch 30/30
    1200/1200 [==============================] - 0s 122us/sample - loss: 0.2663 - accuracy: 0.8917 - val_loss: 0.3082 - val_accuracy: 0.8867



```python
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.title('Wide ANN')
plt.show()
```


![png](homl_ch10_Introduction-to-artificial-neural-networks-with-keras_files/homl_ch10_Introduction-to-artificial-neural-networks-with-keras_96_0.png)



```python
# Construct.
ann_2 = keras.models.Sequential([
    keras.layers.Input(shape=X_blob.shape[1]),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(7, activation='softmax')
])

# Compile.
ann_2.compile(loss='sparse_categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

# Callback for Tensorboard.
tensorboard_cb = keras.callbacks.TensorBoard(get_run_logdir())

# Train.
history = ann_2.fit(X_blob_train, 
          y_blob_train, 
          epochs=30, 
          validation_split=0.2, 
          callbacks=[tensorboard_cb])
```

    Train on 1200 samples, validate on 300 samples
    Epoch 1/30
    1200/1200 [==============================] - 1s 898us/sample - loss: 1.8715 - accuracy: 0.2492 - val_loss: 1.5670 - val_accuracy: 0.3733
    Epoch 2/30
    1200/1200 [==============================] - 0s 141us/sample - loss: 1.4984 - accuracy: 0.4492 - val_loss: 1.3758 - val_accuracy: 0.5233
    Epoch 3/30
    1200/1200 [==============================] - 0s 136us/sample - loss: 1.3493 - accuracy: 0.5350 - val_loss: 1.2634 - val_accuracy: 0.5800
    Epoch 4/30
    1200/1200 [==============================] - 0s 136us/sample - loss: 1.2427 - accuracy: 0.5900 - val_loss: 1.1741 - val_accuracy: 0.5933
    Epoch 5/30
    1200/1200 [==============================] - 0s 135us/sample - loss: 1.1400 - accuracy: 0.6225 - val_loss: 1.0824 - val_accuracy: 0.6433
    Epoch 6/30
    1200/1200 [==============================] - 0s 142us/sample - loss: 1.0364 - accuracy: 0.6667 - val_loss: 0.9943 - val_accuracy: 0.6833
    Epoch 7/30
    1200/1200 [==============================] - 0s 135us/sample - loss: 0.9418 - accuracy: 0.7058 - val_loss: 0.9205 - val_accuracy: 0.6767
    Epoch 8/30
    1200/1200 [==============================] - 0s 136us/sample - loss: 0.8555 - accuracy: 0.7350 - val_loss: 0.8515 - val_accuracy: 0.7567
    Epoch 9/30
    1200/1200 [==============================] - 0s 136us/sample - loss: 0.7761 - accuracy: 0.7858 - val_loss: 0.7687 - val_accuracy: 0.7700
    Epoch 10/30
    1200/1200 [==============================] - 0s 135us/sample - loss: 0.6936 - accuracy: 0.8150 - val_loss: 0.7049 - val_accuracy: 0.8000
    Epoch 11/30
    1200/1200 [==============================] - 0s 137us/sample - loss: 0.6193 - accuracy: 0.8342 - val_loss: 0.6254 - val_accuracy: 0.8167
    Epoch 12/30
    1200/1200 [==============================] - 0s 136us/sample - loss: 0.5565 - accuracy: 0.8367 - val_loss: 0.5786 - val_accuracy: 0.8167
    Epoch 13/30
    1200/1200 [==============================] - 0s 138us/sample - loss: 0.5108 - accuracy: 0.8325 - val_loss: 0.5372 - val_accuracy: 0.8333
    Epoch 14/30
    1200/1200 [==============================] - 0s 133us/sample - loss: 0.4780 - accuracy: 0.8408 - val_loss: 0.5162 - val_accuracy: 0.8267
    Epoch 15/30
    1200/1200 [==============================] - 0s 135us/sample - loss: 0.4557 - accuracy: 0.8442 - val_loss: 0.4910 - val_accuracy: 0.8367
    Epoch 16/30
    1200/1200 [==============================] - 0s 134us/sample - loss: 0.4336 - accuracy: 0.8542 - val_loss: 0.4803 - val_accuracy: 0.8300
    Epoch 17/30
    1200/1200 [==============================] - 0s 131us/sample - loss: 0.4225 - accuracy: 0.8558 - val_loss: 0.4617 - val_accuracy: 0.8367
    Epoch 18/30
    1200/1200 [==============================] - 0s 134us/sample - loss: 0.4133 - accuracy: 0.8533 - val_loss: 0.4496 - val_accuracy: 0.8433
    Epoch 19/30
    1200/1200 [==============================] - 0s 131us/sample - loss: 0.4027 - accuracy: 0.8583 - val_loss: 0.4494 - val_accuracy: 0.8500
    Epoch 20/30
    1200/1200 [==============================] - 0s 134us/sample - loss: 0.3934 - accuracy: 0.8583 - val_loss: 0.4393 - val_accuracy: 0.8433
    Epoch 21/30
    1200/1200 [==============================] - 0s 133us/sample - loss: 0.3851 - accuracy: 0.8650 - val_loss: 0.4336 - val_accuracy: 0.8567
    Epoch 22/30
    1200/1200 [==============================] - 0s 137us/sample - loss: 0.3785 - accuracy: 0.8658 - val_loss: 0.4205 - val_accuracy: 0.8533
    Epoch 23/30
    1200/1200 [==============================] - 0s 137us/sample - loss: 0.3721 - accuracy: 0.8642 - val_loss: 0.4221 - val_accuracy: 0.8467
    Epoch 24/30
    1200/1200 [==============================] - 0s 134us/sample - loss: 0.3666 - accuracy: 0.8675 - val_loss: 0.4129 - val_accuracy: 0.8500
    Epoch 25/30
    1200/1200 [==============================] - 0s 134us/sample - loss: 0.3630 - accuracy: 0.8692 - val_loss: 0.4132 - val_accuracy: 0.8467
    Epoch 26/30
    1200/1200 [==============================] - 0s 132us/sample - loss: 0.3582 - accuracy: 0.8742 - val_loss: 0.4073 - val_accuracy: 0.8567
    Epoch 27/30
    1200/1200 [==============================] - 0s 132us/sample - loss: 0.3516 - accuracy: 0.8683 - val_loss: 0.3994 - val_accuracy: 0.8533
    Epoch 28/30
    1200/1200 [==============================] - 0s 134us/sample - loss: 0.3477 - accuracy: 0.8767 - val_loss: 0.3917 - val_accuracy: 0.8500
    Epoch 29/30
    1200/1200 [==============================] - 0s 134us/sample - loss: 0.3430 - accuracy: 0.8775 - val_loss: 0.4274 - val_accuracy: 0.8467
    Epoch 30/30
    1200/1200 [==============================] - 0s 133us/sample - loss: 0.3413 - accuracy: 0.8758 - val_loss: 0.3988 - val_accuracy: 0.8567



```python
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.title('Deep ANN')
plt.show()
```


![png](homl_ch10_Introduction-to-artificial-neural-networks-with-keras_files/homl_ch10_Introduction-to-artificial-neural-networks-with-keras_98_0.png)


Tensorboard can be started either from the CLI or through magic commands in a Jupyter Notebook.
The latter is used to activate Tesnorboard below.
For the CLI method, just use the second command without the leading `%`.


```python
%reload_ext tensorboard
%tensorboard --logdir=./tf_logs_ch10 --port=6006
```



<iframe id="tensorboard-frame-9328e93cd894d2bc" width="100%" height="800" frameborder="0">
</iframe>
<script>
  (function() {
    const frame = document.getElementById("tensorboard-frame-9328e93cd894d2bc");
    const url = new URL("/", window.location);
    url.port = 6006;
    frame.src = url;
  })();
</script>



## Fine-tuning neural network hyperparameters

The simplest, yet naive, way of tuning the hyperparamters is to use Grid Search or Randomized Search from Chapter 2.
To use the `GridSearchCV` and `RandomizedSearchCV` classes from Scikit-Learn, we need to make one function to construct and compile the ANN, then wrap it with a Scikit-Learn API using `keras.wrappers.scikit_learn.KerasRegressor` or `KerasClassifier`.

Below is an example of creating a regressor to predict the values for a very high-dimensional linear model.


```python
from sklearn.datasets import make_regression

Xr, yr = make_regression(n_samples=4000,
                         n_features=100,
                         n_informative=90,
                         n_targets=1,
                         bias=0.0,
                         noise=20,
                         shuffle=True,
                         coef=False,
                         random_state=0)

Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, 
                                                        random_state=0, 
                                                        shuffle=True)
```

The first step if to create a single function that constructs and compiles the ANN given a set of the hyperparamters to tune.
Only the hyperparamters made available as arguments can be tuned.


```python
def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-4, input_shape=100):
    """Construct and compile a Keras Sequential ANN."""
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    for i in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation='relu'))
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(loss='mse', optimizer=optimizer)
    return model
```

The second step it to wrap this function with the Scikit-Learn API.


```python
keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)
```

Below is how an example if how to train a single model using the wrapped regression ANN.
For now, it has all of the default hyperparamters.


```python

history = keras_reg.fit(Xr_train,
                        yr_train,
                        epochs=100,
                        validation_split=0.2,
                        callbacks=[keras.callbacks.EarlyStopping(patience=5)],
                        verbose=1)
```

    Train on 2400 samples, validate on 600 samples
    Epoch 1/100
    2400/2400 [==============================] - 2s 697us/sample - loss: 80842.7652 - val_loss: 5614.9643
    Epoch 2/100
    2400/2400 [==============================] - 0s 171us/sample - loss: 3843.9761 - val_loss: 2835.9806
    Epoch 3/100
    2400/2400 [==============================] - 0s 90us/sample - loss: 1603.5196 - val_loss: 1486.6631
    Epoch 4/100
    2400/2400 [==============================] - 0s 124us/sample - loss: 910.3000 - val_loss: 1020.2191
    Epoch 5/100
    2400/2400 [==============================] - 0s 112us/sample - loss: 655.0098 - val_loss: 1012.5459
    Epoch 6/100
    2400/2400 [==============================] - 0s 88us/sample - loss: 555.6573 - val_loss: 919.6081
    Epoch 7/100
    2400/2400 [==============================] - ETA: 0s - loss: 477.036 - 0s 106us/sample - loss: 483.7755 - val_loss: 922.0758
    Epoch 8/100
    2400/2400 [==============================] - 0s 145us/sample - loss: 458.3321 - val_loss: 864.5639
    Epoch 9/100
    2400/2400 [==============================] - 0s 194us/sample - loss: 431.9420 - val_loss: 895.0225
    Epoch 10/100
    2400/2400 [==============================] - 0s 112us/sample - loss: 401.4542 - val_loss: 877.8324
    Epoch 11/100
    2400/2400 [==============================] - 0s 114us/sample - loss: 400.3141 - val_loss: 861.1881
    Epoch 12/100
    2400/2400 [==============================] - 0s 109us/sample - loss: 368.4688 - val_loss: 873.9443
    Epoch 13/100
    2400/2400 [==============================] - 0s 139us/sample - loss: 359.2944 - val_loss: 820.9361
    Epoch 14/100
    2400/2400 [==============================] - 0s 113us/sample - loss: 355.4209 - val_loss: 789.4344
    Epoch 15/100
    2400/2400 [==============================] - 0s 120us/sample - loss: 331.5245 - val_loss: 840.0759
    Epoch 16/100
    2400/2400 [==============================] - 0s 129us/sample - loss: 335.0913 - val_loss: 860.6806
    Epoch 17/100
    2400/2400 [==============================] - 0s 107us/sample - loss: 317.9181 - val_loss: 873.6323
    Epoch 18/100
    2400/2400 [==============================] - 0s 99us/sample - loss: 314.6124 - val_loss: 806.9757
    Epoch 19/100
    2400/2400 [==============================] - 0s 110us/sample - loss: 291.1224 - val_loss: 866.2093



```python
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.show()
```


![png](homl_ch10_Introduction-to-artificial-neural-networks-with-keras_files/homl_ch10_Introduction-to-artificial-neural-networks-with-keras_109_0.png)


Finally, below is an small example of using Randomized Search with 3-fold CV to tune three of the models hyperparamters.


```python
%%capture

from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV

param_distribs = {
    'n_hidden': [0, 1, 2, 3],
    'n_neurons': np.arange(1, 100),
    'learning_rate': reciprocal(3e-6, 3e-3)
}

# Wrap the Keras regressor and set the number of input neurons.
keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)

# Create the randomized search object and train.
rnd_search_cv = RandomizedSearchCV(keras_reg,
                                   param_distributions=param_distribs,
                                   n_iter=5,
                                   cv=2,
                                   verbose=0)
rnd_search_cv.fit(Xr_train, yr_train,
                  epochs=100,
                  validation_split=0.2,
                  callbacks=[keras.callbacks.EarlyStopping(patience=5)],
                  verbose=0)
```


```python
rnd_search_cv.best_params_
```




    {'learning_rate': 1.876245776062917e-05, 'n_hidden': 3, 'n_neurons': 15}




```python

```
