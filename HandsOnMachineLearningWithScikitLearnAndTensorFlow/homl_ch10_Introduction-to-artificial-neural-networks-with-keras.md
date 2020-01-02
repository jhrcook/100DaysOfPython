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

    Model: "sequential_2"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    flatten_1 (Flatten)          (None, 784)               0         
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




    [<tensorflow.python.keras.layers.core.Flatten at 0x1a5cacf550>,
     <tensorflow.python.keras.layers.core.Dense at 0x1a5c0a7790>,
     <tensorflow.python.keras.layers.core.Dense at 0x1a5cad6e50>,
     <tensorflow.python.keras.layers.core.Dense at 0x1a5cad6890>]




```python
model.get_layer('dense_1')
```




    <tensorflow.python.keras.layers.core.Dense at 0x1a5c0a7790>




```python
weights, biases = model.get_layer('dense_1').get_weights()
```


```python
weights
```




    array([[ 0.03080886, -0.05333123, -0.05565906, ..., -0.02042927,
             0.06373829, -0.04611184],
           [-0.00015379, -0.0357279 ,  0.04347041, ...,  0.05682653,
             0.03356642, -0.06567048],
           [ 0.02849189,  0.06317663, -0.04996761, ..., -0.07197773,
            -0.05356319,  0.03411862],
           ...,
           [-0.02418977,  0.03684312, -0.05788426, ..., -0.02884386,
            -0.03504391,  0.00044907],
           [ 0.03757305,  0.01313937, -0.05386346, ...,  0.06802045,
             0.06104603,  0.02287725],
           [-0.07304647, -0.05812673, -0.03914445, ...,  0.04360647,
             0.01374307, -0.0349627 ]], dtype=float32)




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
    55000/55000 [==============================] - 9s 156us/sample - loss: 0.7248 - accuracy: 0.7689 - val_loss: 0.5255 - val_accuracy: 0.8208
    Epoch 2/30
    55000/55000 [==============================] - 8s 146us/sample - loss: 0.4870 - accuracy: 0.8310 - val_loss: 0.4507 - val_accuracy: 0.8460
    Epoch 3/30
    55000/55000 [==============================] - 8s 146us/sample - loss: 0.4415 - accuracy: 0.8463 - val_loss: 0.4204 - val_accuracy: 0.8572
    Epoch 4/30
    55000/55000 [==============================] - 8s 145us/sample - loss: 0.4141 - accuracy: 0.8552 - val_loss: 0.4106 - val_accuracy: 0.8532
    Epoch 5/30
    55000/55000 [==============================] - 8s 145us/sample - loss: 0.3944 - accuracy: 0.8615 - val_loss: 0.4445 - val_accuracy: 0.8454
    Epoch 6/30
    55000/55000 [==============================] - 8s 145us/sample - loss: 0.3794 - accuracy: 0.8660 - val_loss: 0.3852 - val_accuracy: 0.8672
    Epoch 7/30
    55000/55000 [==============================] - 10s 175us/sample - loss: 0.3658 - accuracy: 0.8720 - val_loss: 0.4213 - val_accuracy: 0.8442
    Epoch 8/30
    55000/55000 [==============================] - 10s 184us/sample - loss: 0.3552 - accuracy: 0.8742 - val_loss: 0.3804 - val_accuracy: 0.8640
    Epoch 9/30
    55000/55000 [==============================] - 9s 164us/sample - loss: 0.3442 - accuracy: 0.8776 - val_loss: 0.3554 - val_accuracy: 0.8698
    Epoch 10/30
    55000/55000 [==============================] - 8s 144us/sample - loss: 0.3347 - accuracy: 0.8813 - val_loss: 0.3455 - val_accuracy: 0.8776
    Epoch 11/30
    55000/55000 [==============================] - 8s 147us/sample - loss: 0.3252 - accuracy: 0.8831 - val_loss: 0.3448 - val_accuracy: 0.8722
    Epoch 12/30
    55000/55000 [==============================] - 8s 146us/sample - loss: 0.3178 - accuracy: 0.8871 - val_loss: 0.3654 - val_accuracy: 0.8674
    Epoch 13/30
    55000/55000 [==============================] - 8s 150us/sample - loss: 0.3113 - accuracy: 0.8891 - val_loss: 0.3944 - val_accuracy: 0.8586
    Epoch 14/30
    55000/55000 [==============================] - 9s 162us/sample - loss: 0.3039 - accuracy: 0.8902 - val_loss: 0.3275 - val_accuracy: 0.8826
    Epoch 15/30
    55000/55000 [==============================] - 8s 149us/sample - loss: 0.2979 - accuracy: 0.8925 - val_loss: 0.3279 - val_accuracy: 0.8844
    Epoch 16/30
    55000/55000 [==============================] - 8s 151us/sample - loss: 0.2918 - accuracy: 0.8954 - val_loss: 0.3278 - val_accuracy: 0.8818
    Epoch 17/30
    55000/55000 [==============================] - 8s 149us/sample - loss: 0.2855 - accuracy: 0.8967 - val_loss: 0.3227 - val_accuracy: 0.8866
    Epoch 18/30
    55000/55000 [==============================] - 8s 151us/sample - loss: 0.2802 - accuracy: 0.8975 - val_loss: 0.3209 - val_accuracy: 0.8830
    Epoch 19/30
    55000/55000 [==============================] - 8s 148us/sample - loss: 0.2752 - accuracy: 0.9007 - val_loss: 0.3117 - val_accuracy: 0.8864
    Epoch 20/30
    55000/55000 [==============================] - 8s 147us/sample - loss: 0.2701 - accuracy: 0.9032 - val_loss: 0.3188 - val_accuracy: 0.8844
    Epoch 21/30
    55000/55000 [==============================] - 8s 149us/sample - loss: 0.2648 - accuracy: 0.9040 - val_loss: 0.3263 - val_accuracy: 0.8782
    Epoch 22/30
    55000/55000 [==============================] - 8s 149us/sample - loss: 0.2599 - accuracy: 0.9056 - val_loss: 0.3154 - val_accuracy: 0.8872
    Epoch 23/30
    55000/55000 [==============================] - 8s 149us/sample - loss: 0.2555 - accuracy: 0.9080 - val_loss: 0.3128 - val_accuracy: 0.8904
    Epoch 24/30
    55000/55000 [==============================] - 8s 150us/sample - loss: 0.2520 - accuracy: 0.9087 - val_loss: 0.3251 - val_accuracy: 0.8830
    Epoch 25/30
    55000/55000 [==============================] - 8s 150us/sample - loss: 0.2472 - accuracy: 0.9102 - val_loss: 0.3101 - val_accuracy: 0.8912
    Epoch 26/30
    55000/55000 [==============================] - 8s 150us/sample - loss: 0.2437 - accuracy: 0.9122 - val_loss: 0.3004 - val_accuracy: 0.8880
    Epoch 27/30
    55000/55000 [==============================] - 8s 150us/sample - loss: 0.2388 - accuracy: 0.9131 - val_loss: 0.2948 - val_accuracy: 0.8924
    Epoch 28/30
    55000/55000 [==============================] - 8s 150us/sample - loss: 0.2354 - accuracy: 0.9150 - val_loss: 0.3074 - val_accuracy: 0.8894
    Epoch 29/30
    55000/55000 [==============================] - 8s 145us/sample - loss: 0.2317 - accuracy: 0.9168 - val_loss: 0.3061 - val_accuracy: 0.8934
    Epoch 30/30
    55000/55000 [==============================] - 8s 142us/sample - loss: 0.2279 - accuracy: 0.9174 - val_loss: 0.3017 - val_accuracy: 0.8942



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




    [0.33415108824968337, 0.881]



We can also use the model to make predictions.
Using the `predict()` method, we get a probability per class.


```python
X_new = fashion_preprocesser.transform(X_test[:3])
np.round(model.predict(X_new), 2)
```




    array([[0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.01, 0.  , 0.99],
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
    11610/11610 [==============================] - 2s 133us/sample - loss: 0.8786 - val_loss: 0.5121
    Epoch 2/20
    11610/11610 [==============================] - 1s 89us/sample - loss: 0.4524 - val_loss: 0.4363
    Epoch 3/20
    11610/11610 [==============================] - 1s 77us/sample - loss: 0.4168 - val_loss: 0.4273
    Epoch 4/20
    11610/11610 [==============================] - 1s 80us/sample - loss: 0.5157 - val_loss: 0.4958
    Epoch 5/20
    11610/11610 [==============================] - 1s 91us/sample - loss: 0.4154 - val_loss: 0.4108
    Epoch 6/20
    11610/11610 [==============================] - 1s 90us/sample - loss: 0.3859 - val_loss: 0.4093
    Epoch 7/20
    11610/11610 [==============================] - 1s 89us/sample - loss: 0.3768 - val_loss: 0.3988
    Epoch 8/20
    11610/11610 [==============================] - 1s 90us/sample - loss: 0.3691 - val_loss: 0.3957
    Epoch 9/20
    11610/11610 [==============================] - 1s 90us/sample - loss: 0.3629 - val_loss: 0.3889
    Epoch 10/20
    11610/11610 [==============================] - 1s 89us/sample - loss: 0.3592 - val_loss: 0.3824
    Epoch 11/20
    11610/11610 [==============================] - 1s 85us/sample - loss: 0.3549 - val_loss: 0.3761
    Epoch 12/20
    11610/11610 [==============================] - 1s 89us/sample - loss: 0.3506 - val_loss: 0.3757
    Epoch 13/20
    11610/11610 [==============================] - 1s 84us/sample - loss: 0.3467 - val_loss: 0.3792
    Epoch 14/20
    11610/11610 [==============================] - 1s 85us/sample - loss: 0.3455 - val_loss: 0.3826
    Epoch 15/20
    11610/11610 [==============================] - 1s 85us/sample - loss: 0.3428 - val_loss: 0.3676
    Epoch 16/20
    11610/11610 [==============================] - 1s 85us/sample - loss: 0.3435 - val_loss: 0.3603
    Epoch 17/20
    11610/11610 [==============================] - 1s 84us/sample - loss: 0.3358 - val_loss: 0.3690
    Epoch 18/20
    11610/11610 [==============================] - 1s 89us/sample - loss: 0.3342 - val_loss: 0.3621
    Epoch 19/20
    11610/11610 [==============================] - 1s 77us/sample - loss: 0.3299 - val_loss: 0.3607
    Epoch 20/20
    11610/11610 [==============================] - 1s 84us/sample - loss: 0.3309 - val_loss: 0.3592



```python
model.summary()
```

    Model: "sequential_3"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_27 (Dense)             (None, 30)                270       
    _________________________________________________________________
    dense_28 (Dense)             (None, 100)               3100      
    _________________________________________________________________
    dense_29 (Dense)             (None, 1)                 101       
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




    0.3401941142571989




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
    11610/11610 [==============================] - 2s 159us/sample - loss: 0.9070 - val_loss: 0.4935
    Epoch 2/20
    11610/11610 [==============================] - 1s 99us/sample - loss: 0.4466 - val_loss: 0.4392
    Epoch 3/20
    11610/11610 [==============================] - 1s 97us/sample - loss: 0.4163 - val_loss: 0.4266
    Epoch 4/20
    11610/11610 [==============================] - 1s 107us/sample - loss: 0.4034 - val_loss: 0.4161
    Epoch 5/20
    11610/11610 [==============================] - 1s 100us/sample - loss: 0.3941 - val_loss: 0.4122
    Epoch 6/20
    11610/11610 [==============================] - 1s 97us/sample - loss: 0.3898 - val_loss: 0.4042
    Epoch 7/20
    11610/11610 [==============================] - 1s 99us/sample - loss: 0.3831 - val_loss: 0.4079
    Epoch 8/20
    11610/11610 [==============================] - 1s 99us/sample - loss: 0.3794 - val_loss: 0.4012
    Epoch 9/20
    11610/11610 [==============================] - 1s 103us/sample - loss: 0.3747 - val_loss: 0.3934
    Epoch 10/20
    11610/11610 [==============================] - 1s 104us/sample - loss: 0.3714 - val_loss: 0.3980
    Epoch 11/20
    11610/11610 [==============================] - 1s 99us/sample - loss: 0.3695 - val_loss: 0.3872
    Epoch 12/20
    11610/11610 [==============================] - 1s 97us/sample - loss: 0.3667 - val_loss: 0.3859
    Epoch 13/20
    11610/11610 [==============================] - 1s 99us/sample - loss: 0.3610 - val_loss: 0.4051
    Epoch 14/20
    11610/11610 [==============================] - 1s 100us/sample - loss: 0.3587 - val_loss: 0.3918
    Epoch 15/20
    11610/11610 [==============================] - 1s 100us/sample - loss: 0.3660 - val_loss: 0.4032
    Epoch 16/20
    11610/11610 [==============================] - 1s 99us/sample - loss: 0.3592 - val_loss: 0.3847
    Epoch 17/20
    11610/11610 [==============================] - 1s 99us/sample - loss: 0.3507 - val_loss: 0.3772
    Epoch 18/20
    11610/11610 [==============================] - 1s 99us/sample - loss: 0.3471 - val_loss: 0.3688
    Epoch 19/20
    11610/11610 [==============================] - 1s 104us/sample - loss: 0.3441 - val_loss: 0.3655
    Epoch 20/20
    11610/11610 [==============================] - 1s 100us/sample - loss: 0.3410 - val_loss: 0.3771



![png](homl_ch10_Introduction-to-artificial-neural-networks-with-keras_files/homl_ch10_Introduction-to-artificial-neural-networks-with-keras_53_1.png)



```python
model.evaluate(X_test, y_test, verbose=0)
```




    0.3660215558931809




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

    Model: "model_13"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    deep_input (InputLayer)         [(None, 6)]          0                                            
    __________________________________________________________________________________________________
    dense_33 (Dense)                (None, 30)           210         deep_input[0][0]                 
    __________________________________________________________________________________________________
    wide_input (InputLayer)         [(None, 5)]          0                                            
    __________________________________________________________________________________________________
    dense_34 (Dense)                (None, 100)          3100        dense_33[0][0]                   
    __________________________________________________________________________________________________
    concatenate_8 (Concatenate)     (None, 105)          0           wide_input[0][0]                 
                                                                     dense_34[0][0]                   
    __________________________________________________________________________________________________
    output (Dense)                  (None, 1)            106         concatenate_8[0][0]              
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
    11610/11610 [==============================] - 2s 150us/sample - loss: 2.3573 - val_loss: 1.0254
    Epoch 2/20
    11610/11610 [==============================] - 2s 142us/sample - loss: 0.8709 - val_loss: 0.7388
    Epoch 3/20
    11610/11610 [==============================] - 2s 133us/sample - loss: 0.7052 - val_loss: 0.6821
    Epoch 4/20
    11610/11610 [==============================] - 1s 109us/sample - loss: 0.6591 - val_loss: 0.6504
    Epoch 5/20
    11610/11610 [==============================] - 1s 118us/sample - loss: 0.6303 - val_loss: 0.6262
    Epoch 6/20
    11610/11610 [==============================] - 1s 103us/sample - loss: 0.6062 - val_loss: 0.6063
    Epoch 7/20
    11610/11610 [==============================] - 1s 111us/sample - loss: 0.5848 - val_loss: 0.5863
    Epoch 8/20
    11610/11610 [==============================] - 1s 105us/sample - loss: 0.5671 - val_loss: 0.5689
    Epoch 9/20
    11610/11610 [==============================] - 1s 106us/sample - loss: 0.5520 - val_loss: 0.5589
    Epoch 10/20
    11610/11610 [==============================] - 1s 122us/sample - loss: 0.5396 - val_loss: 0.5550
    Epoch 11/20
    11610/11610 [==============================] - 1s 119us/sample - loss: 0.5279 - val_loss: 0.5381
    Epoch 12/20
    11610/11610 [==============================] - 1s 103us/sample - loss: 0.5170 - val_loss: 0.5254
    Epoch 13/20
    11610/11610 [==============================] - 1s 97us/sample - loss: 0.5086 - val_loss: 0.5167
    Epoch 14/20
    11610/11610 [==============================] - 1s 97us/sample - loss: 0.4994 - val_loss: 0.5067
    Epoch 15/20
    11610/11610 [==============================] - 1s 96us/sample - loss: 0.4916 - val_loss: 0.4993
    Epoch 16/20
    11610/11610 [==============================] - 1s 96us/sample - loss: 0.4851 - val_loss: 0.4917
    Epoch 17/20
    11610/11610 [==============================] - 1s 97us/sample - loss: 0.4798 - val_loss: 0.4868
    Epoch 18/20
    11610/11610 [==============================] - 1s 106us/sample - loss: 0.4732 - val_loss: 0.4803
    Epoch 19/20
    11610/11610 [==============================] - 1s 85us/sample - loss: 0.4683 - val_loss: 0.4977
    Epoch 20/20
    11610/11610 [==============================] - 1s 90us/sample - loss: 0.4642 - val_loss: 0.4800



```python
model.evaluate(split_input_matrices(X_test), y_test, verbose=0)
```




    0.49082014066304347




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

    Model: "model_14"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    deep_input (InputLayer)         [(None, 6)]          0                                            
    __________________________________________________________________________________________________
    dense_35 (Dense)                (None, 30)           210         deep_input[0][0]                 
    __________________________________________________________________________________________________
    wide_input (InputLayer)         [(None, 5)]          0                                            
    __________________________________________________________________________________________________
    dense_36 (Dense)                (None, 100)          3100        dense_35[0][0]                   
    __________________________________________________________________________________________________
    concatenate_9 (Concatenate)     (None, 105)          0           wide_input[0][0]                 
                                                                     dense_36[0][0]                   
    __________________________________________________________________________________________________
    output (Dense)                  (None, 1)            106         concatenate_9[0][0]              
    __________________________________________________________________________________________________
    aux_output (Dense)              (None, 1)            101         dense_36[0][0]                   
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
    11610/11610 [==============================] - 2s 201us/sample - loss: 0.9165 - output_loss: 0.7785 - aux_output_loss: 2.1560 - val_loss: 0.5947 - val_output_loss: 0.5234 - val_aux_output_loss: 1.2376
    Epoch 2/20
    11610/11610 [==============================] - 1s 113us/sample - loss: 0.6496 - output_loss: 0.5853 - aux_output_loss: 1.2267 - val_loss: 0.5385 - val_output_loss: 0.4791 - val_aux_output_loss: 1.0730
    Epoch 3/20
    11610/11610 [==============================] - 1s 114us/sample - loss: 0.5199 - output_loss: 0.4639 - aux_output_loss: 1.0257 - val_loss: 0.4970 - val_output_loss: 0.4492 - val_aux_output_loss: 0.9279
    Epoch 4/20
    11610/11610 [==============================] - 1s 113us/sample - loss: 0.4937 - output_loss: 0.4495 - aux_output_loss: 0.8898 - val_loss: 0.4763 - val_output_loss: 0.4374 - val_aux_output_loss: 0.8272
    Epoch 5/20
    11610/11610 [==============================] - 1s 118us/sample - loss: 0.4624 - output_loss: 0.4260 - aux_output_loss: 0.7886 - val_loss: 0.5046 - val_output_loss: 0.4789 - val_aux_output_loss: 0.7362
    Epoch 6/20
    11610/11610 [==============================] - 1s 113us/sample - loss: 0.4469 - output_loss: 0.4172 - aux_output_loss: 0.7146 - val_loss: 0.4433 - val_output_loss: 0.4152 - val_aux_output_loss: 0.6966
    Epoch 7/20
    11610/11610 [==============================] - 1s 113us/sample - loss: 0.4317 - output_loss: 0.4061 - aux_output_loss: 0.6638 - val_loss: 0.4351 - val_output_loss: 0.4109 - val_aux_output_loss: 0.6529
    Epoch 8/20
    11610/11610 [==============================] - 1s 113us/sample - loss: 0.4202 - output_loss: 0.3973 - aux_output_loss: 0.6256 - val_loss: 0.4242 - val_output_loss: 0.4014 - val_aux_output_loss: 0.6305
    Epoch 9/20
    11610/11610 [==============================] - 1s 114us/sample - loss: 0.4117 - output_loss: 0.3908 - aux_output_loss: 0.5998 - val_loss: 0.4210 - val_output_loss: 0.4007 - val_aux_output_loss: 0.6039
    Epoch 10/20
    11610/11610 [==============================] - 1s 114us/sample - loss: 0.4016 - output_loss: 0.3820 - aux_output_loss: 0.5777 - val_loss: 0.4065 - val_output_loss: 0.3860 - val_aux_output_loss: 0.5913
    Epoch 11/20
    11610/11610 [==============================] - 1s 115us/sample - loss: 0.3996 - output_loss: 0.3818 - aux_output_loss: 0.5612 - val_loss: 0.3993 - val_output_loss: 0.3805 - val_aux_output_loss: 0.5681
    Epoch 12/20
    11610/11610 [==============================] - 1s 115us/sample - loss: 0.3873 - output_loss: 0.3697 - aux_output_loss: 0.5458 - val_loss: 0.4182 - val_output_loss: 0.4014 - val_aux_output_loss: 0.5695
    Epoch 13/20
    11610/11610 [==============================] - 1s 120us/sample - loss: 0.3824 - output_loss: 0.3660 - aux_output_loss: 0.5327 - val_loss: 0.3899 - val_output_loss: 0.3725 - val_aux_output_loss: 0.5461
    Epoch 14/20
    11610/11610 [==============================] - 1s 113us/sample - loss: 0.3775 - output_loss: 0.3614 - aux_output_loss: 0.5225 - val_loss: 0.3846 - val_output_loss: 0.3670 - val_aux_output_loss: 0.5428
    Epoch 15/20
    11610/11610 [==============================] - 1s 115us/sample - loss: 0.3705 - output_loss: 0.3545 - aux_output_loss: 0.5130 - val_loss: 0.3763 - val_output_loss: 0.3599 - val_aux_output_loss: 0.5240
    Epoch 16/20
    11610/11610 [==============================] - 1s 114us/sample - loss: 0.3676 - output_loss: 0.3523 - aux_output_loss: 0.5047 - val_loss: 0.3716 - val_output_loss: 0.3553 - val_aux_output_loss: 0.5181
    Epoch 17/20
    11610/11610 [==============================] - 1s 114us/sample - loss: 0.3643 - output_loss: 0.3496 - aux_output_loss: 0.4958 - val_loss: 0.3692 - val_output_loss: 0.3529 - val_aux_output_loss: 0.5157
    Epoch 18/20
    11610/11610 [==============================] - 1s 115us/sample - loss: 0.3589 - output_loss: 0.3445 - aux_output_loss: 0.4888 - val_loss: 0.3648 - val_output_loss: 0.3497 - val_aux_output_loss: 0.5016
    Epoch 19/20
    11610/11610 [==============================] - 1s 115us/sample - loss: 0.3541 - output_loss: 0.3402 - aux_output_loss: 0.4786 - val_loss: 0.3841 - val_output_loss: 0.3697 - val_aux_output_loss: 0.5138
    Epoch 20/20
    11610/11610 [==============================] - 1s 115us/sample - loss: 0.3525 - output_loss: 0.3392 - aux_output_loss: 0.4734 - val_loss: 0.3690 - val_output_loss: 0.3541 - val_aux_output_loss: 0.5036



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
    11610/11610 [==============================] - 2s 197us/sample - loss: 0.9392 - output_1_loss: 0.8232 - output_2_loss: 1.9822 - val_loss: 0.6092 - val_output_1_loss: 0.5484 - val_output_2_loss: 1.1576
    Epoch 2/20
    11610/11610 [==============================] - 1s 117us/sample - loss: 0.5874 - output_1_loss: 0.5349 - output_2_loss: 1.0604 - val_loss: 0.5272 - val_output_1_loss: 0.4832 - val_output_2_loss: 0.9241
    Epoch 3/20
    11610/11610 [==============================] - 1s 111us/sample - loss: 0.5023 - output_1_loss: 0.4608 - output_2_loss: 0.8755 - val_loss: 0.4959 - val_output_1_loss: 0.4630 - val_output_2_loss: 0.7931
    Epoch 4/20
    11610/11610 [==============================] - 1s 110us/sample - loss: 0.4804 - output_1_loss: 0.4479 - output_2_loss: 0.7726 - val_loss: 0.4725 - val_output_1_loss: 0.4459 - val_output_2_loss: 0.7131
    Epoch 5/20
    11610/11610 [==============================] - 1s 111us/sample - loss: 0.4684 - output_1_loss: 0.4420 - output_2_loss: 0.7062 - val_loss: 0.4592 - val_output_1_loss: 0.4358 - val_output_2_loss: 0.6710
    Epoch 6/20
    11610/11610 [==============================] - 1s 115us/sample - loss: 0.4482 - output_1_loss: 0.4242 - output_2_loss: 0.6638 - val_loss: 0.4496 - val_output_1_loss: 0.4277 - val_output_2_loss: 0.6471
    Epoch 7/20
    11610/11610 [==============================] - 1s 108us/sample - loss: 0.4505 - output_1_loss: 0.4290 - output_2_loss: 0.6431 - val_loss: 0.4381 - val_output_1_loss: 0.4174 - val_output_2_loss: 0.6252
    Epoch 8/20
    11610/11610 [==============================] - 1s 111us/sample - loss: 0.4313 - output_1_loss: 0.4104 - output_2_loss: 0.6185 - val_loss: 0.4397 - val_output_1_loss: 0.4166 - val_output_2_loss: 0.6469
    Epoch 9/20
    11610/11610 [==============================] - 1s 110us/sample - loss: 0.4236 - output_1_loss: 0.4032 - output_2_loss: 0.6071 - val_loss: 0.4316 - val_output_1_loss: 0.4117 - val_output_2_loss: 0.6109
    Epoch 10/20
    11610/11610 [==============================] - 1s 111us/sample - loss: 0.4177 - output_1_loss: 0.3982 - output_2_loss: 0.5940 - val_loss: 0.4244 - val_output_1_loss: 0.4049 - val_output_2_loss: 0.6005
    Epoch 11/20
    11610/11610 [==============================] - 1s 110us/sample - loss: 0.4145 - output_1_loss: 0.3955 - output_2_loss: 0.5849 - val_loss: 0.4248 - val_output_1_loss: 0.4026 - val_output_2_loss: 0.6250
    Epoch 12/20
    11610/11610 [==============================] - 1s 114us/sample - loss: 0.4089 - output_1_loss: 0.3900 - output_2_loss: 0.5783 - val_loss: 0.4216 - val_output_1_loss: 0.4017 - val_output_2_loss: 0.6007
    Epoch 13/20
    11610/11610 [==============================] - 1s 110us/sample - loss: 0.4455 - output_1_loss: 0.4292 - output_2_loss: 0.5933 - val_loss: 0.4223 - val_output_1_loss: 0.4035 - val_output_2_loss: 0.5917
    Epoch 14/20
    11610/11610 [==============================] - 2s 144us/sample - loss: 0.4026 - output_1_loss: 0.3830 - output_2_loss: 0.5787 - val_loss: 0.4220 - val_output_1_loss: 0.3942 - val_output_2_loss: 0.6726
    Epoch 15/20
    11610/11610 [==============================] - 1s 112us/sample - loss: 0.4269 - output_1_loss: 0.4113 - output_2_loss: 0.5688 - val_loss: 0.4233 - val_output_1_loss: 0.4039 - val_output_2_loss: 0.5987
    Epoch 16/20
    11610/11610 [==============================] - 1s 114us/sample - loss: 0.3920 - output_1_loss: 0.3738 - output_2_loss: 0.5544 - val_loss: 0.4117 - val_output_1_loss: 0.3942 - val_output_2_loss: 0.5699
    Epoch 17/20
    11610/11610 [==============================] - 1s 112us/sample - loss: 0.3885 - output_1_loss: 0.3712 - output_2_loss: 0.5451 - val_loss: 0.3994 - val_output_1_loss: 0.3814 - val_output_2_loss: 0.5614
    Epoch 18/20
    11610/11610 [==============================] - 1s 111us/sample - loss: 0.3919 - output_1_loss: 0.3751 - output_2_loss: 0.5420 - val_loss: 0.4093 - val_output_1_loss: 0.3932 - val_output_2_loss: 0.5542
    Epoch 19/20
    11610/11610 [==============================] - 1s 112us/sample - loss: 0.3809 - output_1_loss: 0.3642 - output_2_loss: 0.5316 - val_loss: 0.3902 - val_output_1_loss: 0.3726 - val_output_2_loss: 0.5485
    Epoch 20/20
    11610/11610 [==============================] - 1s 112us/sample - loss: 0.3761 - output_1_loss: 0.3596 - output_2_loss: 0.5248 - val_loss: 0.3962 - val_output_1_loss: 0.3790 - val_output_2_loss: 0.5510



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
    11610/11610 [==============================] - 2s 179us/sample - loss: 0.7129 - val_loss: 0.5697
    Epoch 2/20
    11610/11610 [==============================] - 1s 119us/sample - loss: 0.4762 - val_loss: 0.4574
    Epoch 3/20
    11610/11610 [==============================] - 1s 112us/sample - loss: 0.4357 - val_loss: 0.4820
    Epoch 4/20
    11610/11610 [==============================] - 1s 112us/sample - loss: 0.4173 - val_loss: 0.4295
    Epoch 5/20
    11610/11610 [==============================] - 1s 104us/sample - loss: 0.3996 - val_loss: 0.4175
    Epoch 6/20
    11610/11610 [==============================] - 1s 112us/sample - loss: 0.3865 - val_loss: 0.4050
    Epoch 7/20
    11610/11610 [==============================] - 1s 101us/sample - loss: 0.3827 - val_loss: 0.4087
    Epoch 8/20
    11610/11610 [==============================] - 1s 106us/sample - loss: 0.3761 - val_loss: 0.4142
    Epoch 9/20
    11610/11610 [==============================] - 1s 99us/sample - loss: 0.3713 - val_loss: 0.4009
    Epoch 10/20
    11610/11610 [==============================] - 1s 99us/sample - loss: 0.3653 - val_loss: 0.3849
    Epoch 11/20
    11610/11610 [==============================] - 1s 99us/sample - loss: 0.3584 - val_loss: 0.3861
    Epoch 12/20
    11610/11610 [==============================] - 1s 99us/sample - loss: 0.3648 - val_loss: 0.3963
    Epoch 13/20
    11610/11610 [==============================] - 1s 100us/sample - loss: 0.3699 - val_loss: 0.4188
    Epoch 14/20
    11610/11610 [==============================] - 1s 114us/sample - loss: 0.3651 - val_loss: 0.3818
    Epoch 15/20
    11610/11610 [==============================] - 1s 114us/sample - loss: 0.4463 - val_loss: 0.5439
    Epoch 16/20
    11610/11610 [==============================] - 1s 98us/sample - loss: 0.4228 - val_loss: 0.4195
    Epoch 17/20
    11610/11610 [==============================] - 1s 100us/sample - loss: 0.3691 - val_loss: 0.3968
    Epoch 18/20
    11610/11610 [==============================] - 1s 89us/sample - loss: 0.3492 - val_loss: 0.3799
    Epoch 19/20
    11610/11610 [==============================] - 1s 89us/sample - loss: 0.3449 - val_loss: 0.3670
    Epoch 20/20
    11610/11610 [==============================] - 1s 88us/sample - loss: 0.3383 - val_loss: 0.3690
    Model: "model_15"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_4 (InputLayer)            [(None, 8)]          0                                            
    __________________________________________________________________________________________________
    dense_41 (Dense)                (None, 30)           270         input_4[0][0]                    
    __________________________________________________________________________________________________
    dense_42 (Dense)                (None, 100)          3100        dense_41[0][0]                   
    __________________________________________________________________________________________________
    concatenate_10 (Concatenate)    (None, 108)          0           input_4[0][0]                    
                                                                     dense_42[0][0]                   
    __________________________________________________________________________________________________
    dense_43 (Dense)                (None, 1)            109         concatenate_10[0][0]             
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
    11610/11610 [==============================] - 2s 155us/sample - loss: 0.3366 - val_loss: 0.3632
    Epoch 2/20
    11610/11610 [==============================] - 1s 100us/sample - loss: 0.3319 - val_loss: 0.3582
    Epoch 3/20
    11610/11610 [==============================] - 1s 101us/sample - loss: 0.3273 - val_loss: 0.3570
    Epoch 4/20
    11610/11610 [==============================] - 1s 101us/sample - loss: 0.3247 - val_loss: 0.3569
    Epoch 5/20
    11610/11610 [==============================] - 1s 104us/sample - loss: 0.3242 - val_loss: 0.3771
    Epoch 6/20
    11610/11610 [==============================] - 1s 103us/sample - loss: 0.3314 - val_loss: 0.3570
    Epoch 7/20
    11610/11610 [==============================] - 1s 109us/sample - loss: 0.3191 - val_loss: 0.3445
    Epoch 8/20
    11610/11610 [==============================] - 1s 111us/sample - loss: 0.3149 - val_loss: 0.3478
    Epoch 9/20
    11610/11610 [==============================] - 1s 119us/sample - loss: 0.3135 - val_loss: 0.3384
    Epoch 10/20
    11610/11610 [==============================] - 1s 119us/sample - loss: 0.3126 - val_loss: 0.3379
    Epoch 11/20
    11610/11610 [==============================] - 1s 101us/sample - loss: 0.3089 - val_loss: 0.3372
    Epoch 12/20
    11610/11610 [==============================] - 1s 111us/sample - loss: 0.3061 - val_loss: 0.3575
    Epoch 13/20
    11610/11610 [==============================] - 1s 103us/sample - loss: 0.3046 - val_loss: 0.3361
    Epoch 14/20
    11610/11610 [==============================] - 1s 89us/sample - loss: 0.3042 - val_loss: 0.3424
    Epoch 15/20
    11610/11610 [==============================] - 1s 102us/sample - loss: 0.3013 - val_loss: 0.3534
    Epoch 16/20
    11610/11610 [==============================] - 1s 110us/sample - loss: 0.3013 - val_loss: 0.3327
    Epoch 17/20
    11610/11610 [==============================] - 1s 109us/sample - loss: 0.3008 - val_loss: 0.3287
    Epoch 18/20
    11610/11610 [==============================] - 1s 111us/sample - loss: 0.2990 - val_loss: 0.3294
    Epoch 19/20
    11610/11610 [==============================] - 1s 111us/sample - loss: 0.2962 - val_loss: 0.3294
    Epoch 20/20
    11610/11610 [==============================] - 1s 111us/sample - loss: 0.2946 - val_loss: 0.3841


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
    11610/11610 [==============================] - 5s 402us/sample - loss: 0.9559 - output_loss: 0.8157 - aux_output_loss: 2.2145 - val_loss: 0.6139 - val_output_loss: 0.5482 - val_aux_output_loss: 1.2056
    Epoch 2/100
    11610/11610 [==============================] - 2s 132us/sample - loss: 0.5532 - output_loss: 0.4910 - aux_output_loss: 1.1149 - val_loss: 0.5143 - val_output_loss: 0.4654 - val_aux_output_loss: 0.9546
    Epoch 3/100
    11610/11610 [==============================] - 2s 136us/sample - loss: 0.5072 - output_loss: 0.4632 - aux_output_loss: 0.9021 - val_loss: 0.4834 - val_output_loss: 0.4475 - val_aux_output_loss: 0.8071
    Epoch 4/100
    11610/11610 [==============================] - 2s 149us/sample - loss: 0.4784 - output_loss: 0.4453 - aux_output_loss: 0.7757 - val_loss: 0.4641 - val_output_loss: 0.4355 - val_aux_output_loss: 0.7218
    Epoch 5/100
    11610/11610 [==============================] - 2s 146us/sample - loss: 0.5238 - output_loss: 0.5008 - aux_output_loss: 0.7312 - val_loss: 0.4528 - val_output_loss: 0.4250 - val_aux_output_loss: 0.7039
    Epoch 6/100
    11610/11610 [==============================] - 2s 138us/sample - loss: 0.4447 - output_loss: 0.4208 - aux_output_loss: 0.6598 - val_loss: 0.4436 - val_output_loss: 0.4207 - val_aux_output_loss: 0.6502
    Epoch 7/100
    11610/11610 [==============================] - 2s 138us/sample - loss: 0.4277 - output_loss: 0.4055 - aux_output_loss: 0.6272 - val_loss: 0.4435 - val_output_loss: 0.4222 - val_aux_output_loss: 0.6352
    Epoch 8/100
    11610/11610 [==============================] - 2s 140us/sample - loss: 0.4207 - output_loss: 0.4000 - aux_output_loss: 0.6088 - val_loss: 0.4234 - val_output_loss: 0.4019 - val_aux_output_loss: 0.6181
    Epoch 9/100
    11610/11610 [==============================] - 2s 138us/sample - loss: 0.4103 - output_loss: 0.3904 - aux_output_loss: 0.5891 - val_loss: 0.4479 - val_output_loss: 0.4303 - val_aux_output_loss: 0.6064
    Epoch 10/100
    11610/11610 [==============================] - 2s 140us/sample - loss: 0.4062 - output_loss: 0.3871 - aux_output_loss: 0.5796 - val_loss: 0.4048 - val_output_loss: 0.3850 - val_aux_output_loss: 0.5832
    Epoch 11/100
    11610/11610 [==============================] - 2s 141us/sample - loss: 0.3977 - output_loss: 0.3794 - aux_output_loss: 0.5624 - val_loss: 0.4050 - val_output_loss: 0.3865 - val_aux_output_loss: 0.5725
    Epoch 12/100
    11610/11610 [==============================] - 2s 142us/sample - loss: 0.3896 - output_loss: 0.3716 - aux_output_loss: 0.5522 - val_loss: 0.4009 - val_output_loss: 0.3830 - val_aux_output_loss: 0.5626
    Epoch 13/100
    11610/11610 [==============================] - 2s 141us/sample - loss: 0.3829 - output_loss: 0.3654 - aux_output_loss: 0.5397 - val_loss: 0.4045 - val_output_loss: 0.3861 - val_aux_output_loss: 0.5708
    Epoch 14/100
    11610/11610 [==============================] - 2s 147us/sample - loss: 0.3910 - output_loss: 0.3752 - aux_output_loss: 0.5334 - val_loss: 0.3879 - val_output_loss: 0.3688 - val_aux_output_loss: 0.5600
    Epoch 15/100
    11610/11610 [==============================] - 2s 171us/sample - loss: 0.3840 - output_loss: 0.3681 - aux_output_loss: 0.5272 - val_loss: 0.3887 - val_output_loss: 0.3722 - val_aux_output_loss: 0.5375
    Epoch 16/100
    11610/11610 [==============================] - 2s 138us/sample - loss: 0.3827 - output_loss: 0.3679 - aux_output_loss: 0.5150 - val_loss: 0.3823 - val_output_loss: 0.3656 - val_aux_output_loss: 0.5321
    Epoch 17/100
    11610/11610 [==============================] - 2s 138us/sample - loss: 0.3674 - output_loss: 0.3517 - aux_output_loss: 0.5077 - val_loss: 0.3852 - val_output_loss: 0.3696 - val_aux_output_loss: 0.5262
    Epoch 18/100
    11610/11610 [==============================] - 2s 143us/sample - loss: 0.3638 - output_loss: 0.3490 - aux_output_loss: 0.4987 - val_loss: 0.3719 - val_output_loss: 0.3560 - val_aux_output_loss: 0.5150
    Epoch 19/100
    11610/11610 [==============================] - 2s 139us/sample - loss: 0.3597 - output_loss: 0.3452 - aux_output_loss: 0.4910 - val_loss: 0.3663 - val_output_loss: 0.3500 - val_aux_output_loss: 0.5126
    Epoch 20/100
    11610/11610 [==============================] - 2s 140us/sample - loss: 0.3556 - output_loss: 0.3415 - aux_output_loss: 0.4832 - val_loss: 0.3667 - val_output_loss: 0.3517 - val_aux_output_loss: 0.5018
    Epoch 21/100
    11610/11610 [==============================] - 2s 142us/sample - loss: 0.3541 - output_loss: 0.3403 - aux_output_loss: 0.4781 - val_loss: 0.3734 - val_output_loss: 0.3579 - val_aux_output_loss: 0.5134
    Epoch 22/100
    11610/11610 [==============================] - 2s 143us/sample - loss: 0.3502 - output_loss: 0.3367 - aux_output_loss: 0.4719 - val_loss: 0.3616 - val_output_loss: 0.3471 - val_aux_output_loss: 0.4920
    Epoch 23/100
    11610/11610 [==============================] - 2s 160us/sample - loss: 0.3498 - output_loss: 0.3366 - aux_output_loss: 0.4678 - val_loss: 0.3595 - val_output_loss: 0.3451 - val_aux_output_loss: 0.4890
    Epoch 24/100
    11610/11610 [==============================] - 2s 175us/sample - loss: 0.3472 - output_loss: 0.3344 - aux_output_loss: 0.4629 - val_loss: 0.3855 - val_output_loss: 0.3719 - val_aux_output_loss: 0.5080
    Epoch 25/100
    11610/11610 [==============================] - 2s 170us/sample - loss: 0.3460 - output_loss: 0.3335 - aux_output_loss: 0.4590 - val_loss: 0.3805 - val_output_loss: 0.3663 - val_aux_output_loss: 0.5085
    Epoch 26/100
    11610/11610 [==============================] - 2s 153us/sample - loss: 0.3441 - output_loss: 0.3318 - aux_output_loss: 0.4555 - val_loss: 0.3575 - val_output_loss: 0.3443 - val_aux_output_loss: 0.4768
    Epoch 27/100
    11610/11610 [==============================] - 2s 143us/sample - loss: 0.3413 - output_loss: 0.3292 - aux_output_loss: 0.4498 - val_loss: 0.3588 - val_output_loss: 0.3453 - val_aux_output_loss: 0.4808
    Epoch 28/100
    11610/11610 [==============================] - 2s 149us/sample - loss: 0.3422 - output_loss: 0.3304 - aux_output_loss: 0.4478 - val_loss: 0.3571 - val_output_loss: 0.3439 - val_aux_output_loss: 0.4763
    Epoch 29/100
    11610/11610 [==============================] - 2s 141us/sample - loss: 0.3383 - output_loss: 0.3266 - aux_output_loss: 0.4426 - val_loss: 0.3750 - val_output_loss: 0.3625 - val_aux_output_loss: 0.4873
    Epoch 30/100
    11610/11610 [==============================] - 2s 144us/sample - loss: 0.3364 - output_loss: 0.3250 - aux_output_loss: 0.4394 - val_loss: 0.3540 - val_output_loss: 0.3412 - val_aux_output_loss: 0.4693
    Epoch 31/100
    11610/11610 [==============================] - 2s 143us/sample - loss: 0.3376 - output_loss: 0.3266 - aux_output_loss: 0.4384 - val_loss: 0.3497 - val_output_loss: 0.3372 - val_aux_output_loss: 0.4628
    Epoch 32/100
    11610/11610 [==============================] - 2s 141us/sample - loss: 0.3337 - output_loss: 0.3225 - aux_output_loss: 0.4344 - val_loss: 0.3639 - val_output_loss: 0.3520 - val_aux_output_loss: 0.4709
    Epoch 33/100
    11610/11610 [==============================] - 2s 147us/sample - loss: 0.3331 - output_loss: 0.3222 - aux_output_loss: 0.4317 - val_loss: 0.3498 - val_output_loss: 0.3374 - val_aux_output_loss: 0.4624
    Epoch 34/100
    11610/11610 [==============================] - 2s 144us/sample - loss: 0.3341 - output_loss: 0.3235 - aux_output_loss: 0.4294 - val_loss: 0.3496 - val_output_loss: 0.3373 - val_aux_output_loss: 0.4599
    Epoch 35/100
    11610/11610 [==============================] - 2s 143us/sample - loss: 0.3308 - output_loss: 0.3200 - aux_output_loss: 0.4269 - val_loss: 0.3436 - val_output_loss: 0.3316 - val_aux_output_loss: 0.4513
    Epoch 36/100
    11610/11610 [==============================] - 2s 137us/sample - loss: 0.3316 - output_loss: 0.3215 - aux_output_loss: 0.4252 - val_loss: 0.3600 - val_output_loss: 0.3486 - val_aux_output_loss: 0.4626
    Epoch 37/100
    11610/11610 [==============================] - 1s 122us/sample - loss: 0.3305 - output_loss: 0.3203 - aux_output_loss: 0.4218 - val_loss: 0.3542 - val_output_loss: 0.3419 - val_aux_output_loss: 0.4645
    Epoch 38/100
    11610/11610 [==============================] - 1s 107us/sample - loss: 0.3315 - output_loss: 0.3213 - aux_output_loss: 0.4236 - val_loss: 0.3548 - val_output_loss: 0.3431 - val_aux_output_loss: 0.4604
    Epoch 39/100
    11610/11610 [==============================] - 1s 102us/sample - loss: 0.3297 - output_loss: 0.3198 - aux_output_loss: 0.4186 - val_loss: 0.3467 - val_output_loss: 0.3350 - val_aux_output_loss: 0.4524
    Epoch 40/100
    11610/11610 [==============================] - 1s 99us/sample - loss: 0.3256 - output_loss: 0.3155 - aux_output_loss: 0.4159 - val_loss: 0.3436 - val_output_loss: 0.3324 - val_aux_output_loss: 0.4451



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


```python

```
