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




    [<tensorflow.python.keras.layers.core.Flatten at 0x1a5d790a10>,
     <tensorflow.python.keras.layers.core.Dense at 0x1a1b660590>,
     <tensorflow.python.keras.layers.core.Dense at 0x1a422316d0>,
     <tensorflow.python.keras.layers.core.Dense at 0x1a42231950>]




```python
model.get_layer('dense_1')
```




    <tensorflow.python.keras.layers.core.Dense at 0x1a1b660590>




```python
weights, biases = model.get_layer('dense_1').get_weights()
```


```python
weights
```




    array([[ 0.01909669, -0.03947309, -0.03040729, ..., -0.03438038,
             0.04921912,  0.00550243],
           [-0.03329127, -0.01047307, -0.07190025, ..., -0.05564579,
            -0.04258295, -0.05159908],
           [ 0.02722082,  0.03788579,  0.0499421 , ..., -0.05883027,
             0.03593911,  0.07339516],
           ...,
           [ 0.02745932,  0.01938465, -0.05722888, ...,  0.00050315,
            -0.01725952, -0.0181507 ],
           [ 0.05467865,  0.06947832,  0.05840707, ...,  0.04485394,
            -0.00939389,  0.06902052],
           [-0.06370141,  0.02759115,  0.0312682 , ...,  0.02090842,
             0.04941054,  0.03282518]], dtype=float32)




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
    55000/55000 [==============================] - 9s 170us/sample - loss: 0.7055 - accuracy: 0.7665 - val_loss: 0.5268 - val_accuracy: 0.8188
    Epoch 2/30
    55000/55000 [==============================] - 8s 145us/sample - loss: 0.4885 - accuracy: 0.8307 - val_loss: 0.4418 - val_accuracy: 0.8524
    Epoch 3/30
    55000/55000 [==============================] - 8s 148us/sample - loss: 0.4439 - accuracy: 0.8460 - val_loss: 0.4289 - val_accuracy: 0.8510
    Epoch 4/30
    55000/55000 [==============================] - 8s 145us/sample - loss: 0.4146 - accuracy: 0.8558 - val_loss: 0.3939 - val_accuracy: 0.8644
    Epoch 5/30
    55000/55000 [==============================] - 8s 142us/sample - loss: 0.3946 - accuracy: 0.8628 - val_loss: 0.3874 - val_accuracy: 0.8658
    Epoch 6/30
    55000/55000 [==============================] - 8s 141us/sample - loss: 0.3775 - accuracy: 0.8677 - val_loss: 0.3698 - val_accuracy: 0.8696
    Epoch 7/30
    55000/55000 [==============================] - 8s 143us/sample - loss: 0.3656 - accuracy: 0.8713 - val_loss: 0.3725 - val_accuracy: 0.8686
    Epoch 8/30
    55000/55000 [==============================] - 8s 140us/sample - loss: 0.3520 - accuracy: 0.8754 - val_loss: 0.3832 - val_accuracy: 0.8570
    Epoch 9/30
    55000/55000 [==============================] - 8s 145us/sample - loss: 0.3426 - accuracy: 0.8783 - val_loss: 0.3568 - val_accuracy: 0.8714
    Epoch 10/30
    55000/55000 [==============================] - 8s 144us/sample - loss: 0.3332 - accuracy: 0.8809 - val_loss: 0.3469 - val_accuracy: 0.8736
    Epoch 11/30
    55000/55000 [==============================] - 8s 146us/sample - loss: 0.3237 - accuracy: 0.8844 - val_loss: 0.3707 - val_accuracy: 0.8656
    Epoch 12/30
    55000/55000 [==============================] - 10s 181us/sample - loss: 0.3168 - accuracy: 0.8873 - val_loss: 0.3350 - val_accuracy: 0.8772
    Epoch 13/30
    55000/55000 [==============================] - 8s 139us/sample - loss: 0.3093 - accuracy: 0.8890 - val_loss: 0.3418 - val_accuracy: 0.8792
    Epoch 14/30
    55000/55000 [==============================] - 8s 137us/sample - loss: 0.3018 - accuracy: 0.8904 - val_loss: 0.3307 - val_accuracy: 0.8808
    Epoch 15/30
    55000/55000 [==============================] - 7s 128us/sample - loss: 0.2959 - accuracy: 0.8931 - val_loss: 0.3269 - val_accuracy: 0.8830
    Epoch 16/30
    55000/55000 [==============================] - 8s 144us/sample - loss: 0.2883 - accuracy: 0.8962 - val_loss: 0.3130 - val_accuracy: 0.8886
    Epoch 17/30
    55000/55000 [==============================] - 8s 138us/sample - loss: 0.2835 - accuracy: 0.8978 - val_loss: 0.3169 - val_accuracy: 0.8860
    Epoch 18/30
    55000/55000 [==============================] - 7s 135us/sample - loss: 0.2770 - accuracy: 0.8998 - val_loss: 0.3436 - val_accuracy: 0.8744
    Epoch 19/30
    55000/55000 [==============================] - 7s 134us/sample - loss: 0.2722 - accuracy: 0.9011 - val_loss: 0.3319 - val_accuracy: 0.8826
    Epoch 20/30
    55000/55000 [==============================] - 7s 136us/sample - loss: 0.2665 - accuracy: 0.9034 - val_loss: 0.3249 - val_accuracy: 0.8836
    Epoch 21/30
    55000/55000 [==============================] - 7s 134us/sample - loss: 0.2610 - accuracy: 0.9044 - val_loss: 0.3196 - val_accuracy: 0.8846
    Epoch 22/30
    55000/55000 [==============================] - 7s 135us/sample - loss: 0.2564 - accuracy: 0.9067 - val_loss: 0.3068 - val_accuracy: 0.8868
    Epoch 23/30
    55000/55000 [==============================] - 7s 134us/sample - loss: 0.2517 - accuracy: 0.9091 - val_loss: 0.3141 - val_accuracy: 0.8838
    Epoch 24/30
    55000/55000 [==============================] - 8s 138us/sample - loss: 0.2473 - accuracy: 0.9096 - val_loss: 0.3209 - val_accuracy: 0.8838
    Epoch 25/30
    55000/55000 [==============================] - 8s 140us/sample - loss: 0.2430 - accuracy: 0.9120 - val_loss: 0.3305 - val_accuracy: 0.8836
    Epoch 26/30
    55000/55000 [==============================] - 13s 242us/sample - loss: 0.2395 - accuracy: 0.9134 - val_loss: 0.3034 - val_accuracy: 0.8932
    Epoch 27/30
    55000/55000 [==============================] - 10s 176us/sample - loss: 0.2351 - accuracy: 0.9152 - val_loss: 0.3025 - val_accuracy: 0.8912
    Epoch 28/30
    55000/55000 [==============================] - 8s 146us/sample - loss: 0.2306 - accuracy: 0.9169 - val_loss: 0.3621 - val_accuracy: 0.8690
    Epoch 29/30
    55000/55000 [==============================] - 8s 148us/sample - loss: 0.2277 - accuracy: 0.9177 - val_loss: 0.3083 - val_accuracy: 0.8888
    Epoch 30/30
    55000/55000 [==============================] - 8s 150us/sample - loss: 0.2224 - accuracy: 0.9191 - val_loss: 0.2963 - val_accuracy: 0.8982



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




    [0.32204199167490005, 0.8867]



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
    11610/11610 [==============================] - 2s 150us/sample - loss: 0.7240 - val_loss: 0.5115
    Epoch 2/20
    11610/11610 [==============================] - 1s 89us/sample - loss: 0.4576 - val_loss: 0.4544
    Epoch 3/20
    11610/11610 [==============================] - 1s 91us/sample - loss: 0.4151 - val_loss: 0.4270
    Epoch 4/20
    11610/11610 [==============================] - 1s 88us/sample - loss: 0.3956 - val_loss: 0.4151
    Epoch 5/20
    11610/11610 [==============================] - 1s 89us/sample - loss: 0.3836 - val_loss: 0.4002
    Epoch 6/20
    11610/11610 [==============================] - 1s 89us/sample - loss: 0.3748 - val_loss: 0.4114
    Epoch 7/20
    11610/11610 [==============================] - 1s 89us/sample - loss: 0.3694 - val_loss: 0.3898
    Epoch 8/20
    11610/11610 [==============================] - 1s 89us/sample - loss: 0.3620 - val_loss: 0.3912
    Epoch 9/20
    11610/11610 [==============================] - 1s 89us/sample - loss: 0.3568 - val_loss: 0.3874
    Epoch 10/20
    11610/11610 [==============================] - 1s 95us/sample - loss: 0.3503 - val_loss: 0.3833
    Epoch 11/20
    11610/11610 [==============================] - 1s 95us/sample - loss: 0.3492 - val_loss: 0.3767
    Epoch 12/20
    11610/11610 [==============================] - 1s 85us/sample - loss: 0.4252 - val_loss: 0.3970
    Epoch 13/20
    11610/11610 [==============================] - 1s 85us/sample - loss: 0.3571 - val_loss: 0.3842
    Epoch 14/20
    11610/11610 [==============================] - 1s 85us/sample - loss: 0.3420 - val_loss: 0.3744
    Epoch 15/20
    11610/11610 [==============================] - 1s 84us/sample - loss: 0.3366 - val_loss: 0.3659
    Epoch 16/20
    11610/11610 [==============================] - 1s 84us/sample - loss: 0.3365 - val_loss: 0.3589
    Epoch 17/20
    11610/11610 [==============================] - 1s 84us/sample - loss: 0.3335 - val_loss: 0.3658
    Epoch 18/20
    11610/11610 [==============================] - 1s 84us/sample - loss: 0.3274 - val_loss: 0.3688
    Epoch 19/20
    11610/11610 [==============================] - 1s 84us/sample - loss: 0.3235 - val_loss: 0.3480
    Epoch 20/20
    11610/11610 [==============================] - 1s 89us/sample - loss: 0.3195 - val_loss: 0.3547



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




    0.337194950913274




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
    11610/11610 [==============================] - 2s 170us/sample - loss: 1.0967 - val_loss: 0.6579
    Epoch 2/20
    11610/11610 [==============================] - 1s 111us/sample - loss: 0.5468 - val_loss: 0.5027
    Epoch 3/20
    11610/11610 [==============================] - 1s 111us/sample - loss: 0.5267 - val_loss: 0.4958
    Epoch 4/20
    11610/11610 [==============================] - 1s 111us/sample - loss: 0.4359 - val_loss: 0.4386
    Epoch 5/20
    11610/11610 [==============================] - 1s 101us/sample - loss: 0.3999 - val_loss: 0.4272
    Epoch 6/20
    11610/11610 [==============================] - 1s 101us/sample - loss: 0.3880 - val_loss: 0.4071
    Epoch 7/20
    11610/11610 [==============================] - 1s 104us/sample - loss: 0.3795 - val_loss: 0.4873
    Epoch 8/20
    11610/11610 [==============================] - 1s 99us/sample - loss: 0.3769 - val_loss: 0.4008
    Epoch 9/20
    11610/11610 [==============================] - 1s 99us/sample - loss: 0.3692 - val_loss: 0.3937
    Epoch 10/20
    11610/11610 [==============================] - 1s 99us/sample - loss: 0.3646 - val_loss: 0.3928
    Epoch 11/20
    11610/11610 [==============================] - 1s 99us/sample - loss: 0.3622 - val_loss: 0.3847
    Epoch 12/20
    11610/11610 [==============================] - 1s 99us/sample - loss: 0.3587 - val_loss: 0.3842
    Epoch 13/20
    11610/11610 [==============================] - 1s 97us/sample - loss: 0.3555 - val_loss: 0.3882
    Epoch 14/20
    11610/11610 [==============================] - 1s 94us/sample - loss: 0.3528 - val_loss: 0.3827
    Epoch 15/20
    11610/11610 [==============================] - 1s 89us/sample - loss: 0.3507 - val_loss: 0.3800
    Epoch 16/20
    11610/11610 [==============================] - 1s 91us/sample - loss: 0.3475 - val_loss: 0.3959
    Epoch 17/20
    11610/11610 [==============================] - 1s 104us/sample - loss: 0.3457 - val_loss: 0.3760
    Epoch 18/20
    11610/11610 [==============================] - 1s 99us/sample - loss: 0.3437 - val_loss: 0.3718
    Epoch 19/20
    11610/11610 [==============================] - 1s 99us/sample - loss: 0.3422 - val_loss: 0.3701
    Epoch 20/20
    11610/11610 [==============================] - 1s 99us/sample - loss: 0.3404 - val_loss: 0.3710



![png](homl_ch10_Introduction-to-artificial-neural-networks-with-keras_files/homl_ch10_Introduction-to-artificial-neural-networks-with-keras_53_1.png)



```python
model.evaluate(X_test, y_test, verbose=0)
```




    0.35631471438001294




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
    11610/11610 [==============================] - 2s 168us/sample - loss: 1.9254 - val_loss: 0.9484
    Epoch 2/20
    11610/11610 [==============================] - 1s 96us/sample - loss: 0.8495 - val_loss: 0.7634
    Epoch 3/20
    11610/11610 [==============================] - 1s 102us/sample - loss: 0.7184 - val_loss: 0.6942
    Epoch 4/20
    11610/11610 [==============================] - 1s 97us/sample - loss: 0.6604 - val_loss: 0.6476
    Epoch 5/20
    11610/11610 [==============================] - 1s 97us/sample - loss: 0.6198 - val_loss: 0.6137
    Epoch 6/20
    11610/11610 [==============================] - 1s 97us/sample - loss: 0.5892 - val_loss: 0.5878
    Epoch 7/20
    11610/11610 [==============================] - 1s 96us/sample - loss: 0.5653 - val_loss: 0.5673
    Epoch 8/20
    11610/11610 [==============================] - 1s 98us/sample - loss: 0.5467 - val_loss: 0.5500
    Epoch 9/20
    11610/11610 [==============================] - 1s 97us/sample - loss: 0.5327 - val_loss: 0.5373
    Epoch 10/20
    11610/11610 [==============================] - 1s 108us/sample - loss: 0.5205 - val_loss: 0.5262
    Epoch 11/20
    11610/11610 [==============================] - 1s 98us/sample - loss: 0.5095 - val_loss: 0.5166
    Epoch 12/20
    11610/11610 [==============================] - 1s 98us/sample - loss: 0.5006 - val_loss: 0.5090
    Epoch 13/20
    11610/11610 [==============================] - 1s 96us/sample - loss: 0.4943 - val_loss: 0.5039
    Epoch 14/20
    11610/11610 [==============================] - 1s 98us/sample - loss: 0.4868 - val_loss: 0.4957
    Epoch 15/20
    11610/11610 [==============================] - 1s 98us/sample - loss: 0.4812 - val_loss: 0.4902
    Epoch 16/20
    11610/11610 [==============================] - 1s 98us/sample - loss: 0.4755 - val_loss: 0.4887
    Epoch 17/20
    11610/11610 [==============================] - 1s 97us/sample - loss: 0.4712 - val_loss: 0.4833
    Epoch 18/20
    11610/11610 [==============================] - 1s 97us/sample - loss: 0.4671 - val_loss: 0.4751
    Epoch 19/20
    11610/11610 [==============================] - 1s 105us/sample - loss: 0.4635 - val_loss: 0.4704
    Epoch 20/20
    11610/11610 [==============================] - 1s 108us/sample - loss: 0.4601 - val_loss: 0.4701



```python
model.evaluate(split_input_matrices(X_test), y_test, verbose=0)
```




    0.48695583951103594




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
    11610/11610 [==============================] - 3s 246us/sample - loss: 1.0505 - output_loss: 0.9667 - aux_output_loss: 1.8011 - val_loss: 0.6166 - val_output_loss: 0.5594 - val_aux_output_loss: 1.1317
    Epoch 2/20
    11610/11610 [==============================] - 2s 140us/sample - loss: 0.6090 - output_loss: 0.5627 - aux_output_loss: 1.0263 - val_loss: 0.5358 - val_output_loss: 0.4978 - val_aux_output_loss: 0.8783
    Epoch 3/20
    11610/11610 [==============================] - 2s 133us/sample - loss: 0.4955 - output_loss: 0.4578 - aux_output_loss: 0.8345 - val_loss: 0.4886 - val_output_loss: 0.4574 - val_aux_output_loss: 0.7707
    Epoch 4/20
    11610/11610 [==============================] - 1s 126us/sample - loss: 0.4661 - output_loss: 0.4349 - aux_output_loss: 0.7469 - val_loss: 0.4844 - val_output_loss: 0.4605 - val_aux_output_loss: 0.6997
    Epoch 5/20
    11610/11610 [==============================] - 1s 126us/sample - loss: 0.4503 - output_loss: 0.4236 - aux_output_loss: 0.6917 - val_loss: 0.4531 - val_output_loss: 0.4296 - val_aux_output_loss: 0.6648
    Epoch 6/20
    11610/11610 [==============================] - 1s 126us/sample - loss: 0.4371 - output_loss: 0.4133 - aux_output_loss: 0.6513 - val_loss: 0.4379 - val_output_loss: 0.4166 - val_aux_output_loss: 0.6311
    Epoch 7/20
    11610/11610 [==============================] - 2s 131us/sample - loss: 0.4286 - output_loss: 0.4080 - aux_output_loss: 0.6131 - val_loss: 0.4338 - val_output_loss: 0.4136 - val_aux_output_loss: 0.6159
    Epoch 8/20
    11610/11610 [==============================] - 2s 138us/sample - loss: 0.4139 - output_loss: 0.3945 - aux_output_loss: 0.5884 - val_loss: 0.4306 - val_output_loss: 0.4132 - val_aux_output_loss: 0.5872
    Epoch 9/20
    11610/11610 [==============================] - 2s 133us/sample - loss: 0.4043 - output_loss: 0.3864 - aux_output_loss: 0.5668 - val_loss: 0.4094 - val_output_loss: 0.3912 - val_aux_output_loss: 0.5740
    Epoch 10/20
    11610/11610 [==============================] - 2s 137us/sample - loss: 0.3988 - output_loss: 0.3818 - aux_output_loss: 0.5523 - val_loss: 0.4075 - val_output_loss: 0.3900 - val_aux_output_loss: 0.5645
    Epoch 11/20
    11610/11610 [==============================] - 1s 125us/sample - loss: 0.3921 - output_loss: 0.3762 - aux_output_loss: 0.5353 - val_loss: 0.4011 - val_output_loss: 0.3850 - val_aux_output_loss: 0.5464
    Epoch 12/20
    11610/11610 [==============================] - 2s 140us/sample - loss: 0.3838 - output_loss: 0.3685 - aux_output_loss: 0.5214 - val_loss: 0.4226 - val_output_loss: 0.4061 - val_aux_output_loss: 0.5709
    Epoch 13/20
    11610/11610 [==============================] - 2s 135us/sample - loss: 0.3858 - output_loss: 0.3719 - aux_output_loss: 0.5106 - val_loss: 0.3974 - val_output_loss: 0.3830 - val_aux_output_loss: 0.5276
    Epoch 14/20
    11610/11610 [==============================] - 1s 119us/sample - loss: 0.3762 - output_loss: 0.3622 - aux_output_loss: 0.5022 - val_loss: 0.3908 - val_output_loss: 0.3761 - val_aux_output_loss: 0.5231
    Epoch 15/20
    11610/11610 [==============================] - 1s 117us/sample - loss: 0.3673 - output_loss: 0.3531 - aux_output_loss: 0.4934 - val_loss: 0.3875 - val_output_loss: 0.3739 - val_aux_output_loss: 0.5100
    Epoch 16/20
    11610/11610 [==============================] - 1s 119us/sample - loss: 0.3643 - output_loss: 0.3510 - aux_output_loss: 0.4845 - val_loss: 0.4093 - val_output_loss: 0.3959 - val_aux_output_loss: 0.5305
    Epoch 17/20
    11610/11610 [==============================] - 1s 119us/sample - loss: 0.3648 - output_loss: 0.3518 - aux_output_loss: 0.4808 - val_loss: 0.3880 - val_output_loss: 0.3740 - val_aux_output_loss: 0.5145
    Epoch 18/20
    11610/11610 [==============================] - 1s 120us/sample - loss: 0.3559 - output_loss: 0.3432 - aux_output_loss: 0.4701 - val_loss: 0.3893 - val_output_loss: 0.3745 - val_aux_output_loss: 0.5235
    Epoch 19/20
    11610/11610 [==============================] - 1s 128us/sample - loss: 0.3540 - output_loss: 0.3413 - aux_output_loss: 0.4674 - val_loss: 0.3699 - val_output_loss: 0.3564 - val_aux_output_loss: 0.4919
    Epoch 20/20
    11610/11610 [==============================] - 1s 119us/sample - loss: 0.3512 - output_loss: 0.3390 - aux_output_loss: 0.4604 - val_loss: 0.3718 - val_output_loss: 0.3570 - val_aux_output_loss: 0.5047



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
    11610/11610 [==============================] - 2s 188us/sample - loss: 1.0742 - output_1_loss: 0.9809 - output_2_loss: 1.9124 - val_loss: 0.5835 - val_output_1_loss: 0.5319 - val_output_2_loss: 1.0490
    Epoch 2/20
    11610/11610 [==============================] - 1s 99us/sample - loss: 0.5414 - output_1_loss: 0.4912 - output_2_loss: 0.9942 - val_loss: 0.5239 - val_output_1_loss: 0.4776 - val_output_2_loss: 0.9414
    Epoch 3/20
    11610/11610 [==============================] - 1s 98us/sample - loss: 0.4949 - output_1_loss: 0.4534 - output_2_loss: 0.8679 - val_loss: 0.4820 - val_output_1_loss: 0.4463 - val_output_2_loss: 0.8033
    Epoch 4/20
    11610/11610 [==============================] - 1s 100us/sample - loss: 0.4795 - output_1_loss: 0.4470 - output_2_loss: 0.7711 - val_loss: 0.4765 - val_output_1_loss: 0.4509 - val_output_2_loss: 0.7075
    Epoch 5/20
    11610/11610 [==============================] - 1s 117us/sample - loss: 0.4503 - output_1_loss: 0.4234 - output_2_loss: 0.6938 - val_loss: 0.4556 - val_output_1_loss: 0.4331 - val_output_2_loss: 0.6589
    Epoch 6/20
    11610/11610 [==============================] - 1s 99us/sample - loss: 0.4297 - output_1_loss: 0.4062 - output_2_loss: 0.6406 - val_loss: 0.4343 - val_output_1_loss: 0.4110 - val_output_2_loss: 0.6441
    Epoch 7/20
    11610/11610 [==============================] - 1s 99us/sample - loss: 0.4202 - output_1_loss: 0.3997 - output_2_loss: 0.6046 - val_loss: 0.4240 - val_output_1_loss: 0.4043 - val_output_2_loss: 0.6018
    Epoch 8/20
    11610/11610 [==============================] - 1s 111us/sample - loss: 0.4095 - output_1_loss: 0.3906 - output_2_loss: 0.5798 - val_loss: 0.4154 - val_output_1_loss: 0.3970 - val_output_2_loss: 0.5809
    Epoch 9/20
    11610/11610 [==============================] - 1s 117us/sample - loss: 0.4074 - output_1_loss: 0.3903 - output_2_loss: 0.5605 - val_loss: 0.4131 - val_output_1_loss: 0.3957 - val_output_2_loss: 0.5700
    Epoch 10/20
    11610/11610 [==============================] - 1s 117us/sample - loss: 0.3999 - output_1_loss: 0.3836 - output_2_loss: 0.5467 - val_loss: 0.4066 - val_output_1_loss: 0.3887 - val_output_2_loss: 0.5682
    Epoch 11/20
    11610/11610 [==============================] - 1s 115us/sample - loss: 0.3910 - output_1_loss: 0.3751 - output_2_loss: 0.5329 - val_loss: 0.3943 - val_output_1_loss: 0.3776 - val_output_2_loss: 0.5444
    Epoch 12/20
    11610/11610 [==============================] - 1s 115us/sample - loss: 0.3844 - output_1_loss: 0.3696 - output_2_loss: 0.5179 - val_loss: 0.3912 - val_output_1_loss: 0.3755 - val_output_2_loss: 0.5334
    Epoch 13/20
    11610/11610 [==============================] - 1s 127us/sample - loss: 0.3783 - output_1_loss: 0.3640 - output_2_loss: 0.5077 - val_loss: 0.3866 - val_output_1_loss: 0.3710 - val_output_2_loss: 0.5272
    Epoch 14/20
    11610/11610 [==============================] - 1s 117us/sample - loss: 0.3737 - output_1_loss: 0.3601 - output_2_loss: 0.4955 - val_loss: 0.3888 - val_output_1_loss: 0.3750 - val_output_2_loss: 0.5131
    Epoch 15/20
    11610/11610 [==============================] - 1s 125us/sample - loss: 0.3724 - output_1_loss: 0.3598 - output_2_loss: 0.4861 - val_loss: 0.3786 - val_output_1_loss: 0.3618 - val_output_2_loss: 0.5301
    Epoch 16/20
    11610/11610 [==============================] - 1s 113us/sample - loss: 0.3657 - output_1_loss: 0.3533 - output_2_loss: 0.4762 - val_loss: 0.3670 - val_output_1_loss: 0.3531 - val_output_2_loss: 0.4930
    Epoch 17/20
    11610/11610 [==============================] - 1s 114us/sample - loss: 0.3606 - output_1_loss: 0.3484 - output_2_loss: 0.4697 - val_loss: 0.3692 - val_output_1_loss: 0.3560 - val_output_2_loss: 0.4882
    Epoch 18/20
    11610/11610 [==============================] - 1s 113us/sample - loss: 0.3610 - output_1_loss: 0.3497 - output_2_loss: 0.4631 - val_loss: 0.3873 - val_output_1_loss: 0.3770 - val_output_2_loss: 0.4798
    Epoch 19/20
    11610/11610 [==============================] - 1s 116us/sample - loss: 0.3599 - output_1_loss: 0.3495 - output_2_loss: 0.4535 - val_loss: 0.4469 - val_output_1_loss: 0.4429 - val_output_2_loss: 0.4825
    Epoch 20/20
    11610/11610 [==============================] - 1s 115us/sample - loss: 0.3561 - output_1_loss: 0.3458 - output_2_loss: 0.4495 - val_loss: 0.3650 - val_output_1_loss: 0.3518 - val_output_2_loss: 0.4841



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
    11610/11610 [==============================] - 2s 177us/sample - loss: 0.8410 - val_loss: 0.5378
    Epoch 2/20
    11610/11610 [==============================] - 1s 109us/sample - loss: 0.5082 - val_loss: 0.5506
    Epoch 3/20
    11610/11610 [==============================] - 1s 105us/sample - loss: 0.4695 - val_loss: 0.4615
    Epoch 4/20
    11610/11610 [==============================] - 1s 104us/sample - loss: 0.4172 - val_loss: 0.4949
    Epoch 5/20
    11610/11610 [==============================] - 1s 101us/sample - loss: 0.4006 - val_loss: 0.4349
    Epoch 6/20
    11610/11610 [==============================] - 1s 102us/sample - loss: 0.3883 - val_loss: 0.3973
    Epoch 7/20
    11610/11610 [==============================] - 1s 103us/sample - loss: 0.3745 - val_loss: 0.3925
    Epoch 8/20
    11610/11610 [==============================] - 1s 103us/sample - loss: 0.3690 - val_loss: 0.3871
    Epoch 9/20
    11610/11610 [==============================] - 1s 104us/sample - loss: 0.3584 - val_loss: 0.4044
    Epoch 10/20
    11610/11610 [==============================] - 1s 118us/sample - loss: 0.3554 - val_loss: 0.3882
    Epoch 11/20
    11610/11610 [==============================] - 1s 101us/sample - loss: 0.3518 - val_loss: 0.3743
    Epoch 12/20
    11610/11610 [==============================] - 1s 102us/sample - loss: 0.3486 - val_loss: 0.3767
    Epoch 13/20
    11610/11610 [==============================] - 1s 103us/sample - loss: 0.3448 - val_loss: 0.3708
    Epoch 14/20
    11610/11610 [==============================] - 1s 102us/sample - loss: 0.3419 - val_loss: 0.3792
    Epoch 15/20
    11610/11610 [==============================] - 1s 104us/sample - loss: 0.3392 - val_loss: 0.3633
    Epoch 16/20
    11610/11610 [==============================] - 1s 105us/sample - loss: 0.3351 - val_loss: 0.3667
    Epoch 17/20
    11610/11610 [==============================] - 1s 107us/sample - loss: 0.3339 - val_loss: 0.3721
    Epoch 18/20
    11610/11610 [==============================] - 1s 101us/sample - loss: 0.3313 - val_loss: 0.3572
    Epoch 19/20
    11610/11610 [==============================] - 1s 102us/sample - loss: 0.3277 - val_loss: 0.3570
    Epoch 20/20
    11610/11610 [==============================] - 1s 103us/sample - loss: 0.3258 - val_loss: 0.3541
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
    11610/11610 [==============================] - 2s 161us/sample - loss: 0.3252 - val_loss: 0.3546
    Epoch 2/20
    11610/11610 [==============================] - 1s 110us/sample - loss: 0.3211 - val_loss: 0.3474
    Epoch 3/20
    11610/11610 [==============================] - 1s 100us/sample - loss: 0.3187 - val_loss: 0.3470
    Epoch 4/20
    11610/11610 [==============================] - 1s 111us/sample - loss: 0.3161 - val_loss: 0.3574
    Epoch 5/20
    11610/11610 [==============================] - 2s 163us/sample - loss: 0.3129 - val_loss: 0.3438
    Epoch 6/20
    11610/11610 [==============================] - 2s 131us/sample - loss: 0.3113 - val_loss: 0.3433
    Epoch 7/20
    11610/11610 [==============================] - 2s 138us/sample - loss: 0.3100 - val_loss: 0.3403
    Epoch 8/20
    11610/11610 [==============================] - 1s 103us/sample - loss: 0.3087 - val_loss: 0.3375
    Epoch 9/20
    11610/11610 [==============================] - 1s 102us/sample - loss: 0.3046 - val_loss: 0.3411
    Epoch 10/20
    11610/11610 [==============================] - 1s 105us/sample - loss: 0.3045 - val_loss: 0.3320
    Epoch 11/20
    11610/11610 [==============================] - 1s 106us/sample - loss: 0.3019 - val_loss: 0.3315
    Epoch 12/20
    11610/11610 [==============================] - 1s 102us/sample - loss: 0.3005 - val_loss: 0.3337
    Epoch 13/20
    11610/11610 [==============================] - 1s 109us/sample - loss: 0.2995 - val_loss: 0.3322
    Epoch 14/20
    11610/11610 [==============================] - 1s 102us/sample - loss: 0.2983 - val_loss: 0.3268
    Epoch 15/20
    11610/11610 [==============================] - 1s 90us/sample - loss: 0.2975 - val_loss: 0.3267
    Epoch 16/20
    11610/11610 [==============================] - 1s 93us/sample - loss: 0.2957 - val_loss: 0.3270
    Epoch 17/20
    11610/11610 [==============================] - 1s 104us/sample - loss: 0.2952 - val_loss: 0.3211
    Epoch 18/20
    11610/11610 [==============================] - 1s 105us/sample - loss: 0.2928 - val_loss: 0.3174
    Epoch 19/20
    11610/11610 [==============================] - 1s 104us/sample - loss: 0.3011 - val_loss: 0.3311
    Epoch 20/20
    11610/11610 [==============================] - 1s 106us/sample - loss: 0.2919 - val_loss: 0.3219


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
    11610/11610 [==============================] - 3s 226us/sample - loss: 0.8896 - output_loss: 0.7873 - aux_output_loss: 1.8080 - val_loss: 0.9425 - val_output_loss: 0.9372 - val_aux_output_loss: 0.9891
    Epoch 2/100
    11610/11610 [==============================] - 1s 119us/sample - loss: 0.5127 - output_loss: 0.4671 - aux_output_loss: 0.9238 - val_loss: 0.4873 - val_output_loss: 0.4512 - val_aux_output_loss: 0.8126
    Epoch 3/100
    11610/11610 [==============================] - 1s 120us/sample - loss: 0.4654 - output_loss: 0.4318 - aux_output_loss: 0.7680 - val_loss: 0.4618 - val_output_loss: 0.4323 - val_aux_output_loss: 0.7279
    Epoch 4/100
    11610/11610 [==============================] - 1s 120us/sample - loss: 0.4463 - output_loss: 0.4193 - aux_output_loss: 0.6892 - val_loss: 0.4491 - val_output_loss: 0.4244 - val_aux_output_loss: 0.6712
    Epoch 5/100
    11610/11610 [==============================] - 1s 121us/sample - loss: 0.4345 - output_loss: 0.4112 - aux_output_loss: 0.6445 - val_loss: 0.4390 - val_output_loss: 0.4155 - val_aux_output_loss: 0.6506
    Epoch 6/100
    11610/11610 [==============================] - 1s 121us/sample - loss: 0.4251 - output_loss: 0.4036 - aux_output_loss: 0.6188 - val_loss: 0.4337 - val_output_loss: 0.4111 - val_aux_output_loss: 0.6380
    Epoch 7/100
    11610/11610 [==============================] - 1s 120us/sample - loss: 0.4171 - output_loss: 0.3967 - aux_output_loss: 0.5995 - val_loss: 0.4300 - val_output_loss: 0.4089 - val_aux_output_loss: 0.6203
    Epoch 8/100
    11610/11610 [==============================] - 1s 129us/sample - loss: 0.4107 - output_loss: 0.3911 - aux_output_loss: 0.5870 - val_loss: 0.4203 - val_output_loss: 0.3977 - val_aux_output_loss: 0.6237
    Epoch 9/100
    11610/11610 [==============================] - 1s 121us/sample - loss: 0.4037 - output_loss: 0.3845 - aux_output_loss: 0.5761 - val_loss: 0.4176 - val_output_loss: 0.3979 - val_aux_output_loss: 0.5956
    Epoch 10/100
    11610/11610 [==============================] - 1s 119us/sample - loss: 0.4064 - output_loss: 0.3880 - aux_output_loss: 0.5713 - val_loss: 0.4072 - val_output_loss: 0.3879 - val_aux_output_loss: 0.5814
    Epoch 11/100
    11610/11610 [==============================] - 1s 122us/sample - loss: 0.3921 - output_loss: 0.3734 - aux_output_loss: 0.5593 - val_loss: 0.4000 - val_output_loss: 0.3804 - val_aux_output_loss: 0.5768
    Epoch 12/100
    11610/11610 [==============================] - 1s 128us/sample - loss: 0.3928 - output_loss: 0.3752 - aux_output_loss: 0.5519 - val_loss: 0.4034 - val_output_loss: 0.3826 - val_aux_output_loss: 0.5909
    Epoch 13/100
    11610/11610 [==============================] - 1s 120us/sample - loss: 0.3917 - output_loss: 0.3746 - aux_output_loss: 0.5461 - val_loss: 0.3964 - val_output_loss: 0.3783 - val_aux_output_loss: 0.5588
    Epoch 14/100
    11610/11610 [==============================] - 1s 120us/sample - loss: 0.3823 - output_loss: 0.3654 - aux_output_loss: 0.5353 - val_loss: 0.4030 - val_output_loss: 0.3851 - val_aux_output_loss: 0.5645
    Epoch 15/100
    11610/11610 [==============================] - 1s 123us/sample - loss: 0.3780 - output_loss: 0.3611 - aux_output_loss: 0.5295 - val_loss: 0.3767 - val_output_loss: 0.3587 - val_aux_output_loss: 0.5392
    Epoch 16/100
    11610/11610 [==============================] - 1s 128us/sample - loss: 0.3675 - output_loss: 0.3509 - aux_output_loss: 0.5164 - val_loss: 0.3746 - val_output_loss: 0.3567 - val_aux_output_loss: 0.5365
    Epoch 17/100
    11610/11610 [==============================] - 1s 122us/sample - loss: 0.3627 - output_loss: 0.3466 - aux_output_loss: 0.5085 - val_loss: 0.3752 - val_output_loss: 0.3585 - val_aux_output_loss: 0.5251
    Epoch 18/100
    11610/11610 [==============================] - 1s 123us/sample - loss: 0.3628 - output_loss: 0.3473 - aux_output_loss: 0.5017 - val_loss: 0.3689 - val_output_loss: 0.3521 - val_aux_output_loss: 0.5204
    Epoch 19/100
    11610/11610 [==============================] - 1s 122us/sample - loss: 0.3614 - output_loss: 0.3464 - aux_output_loss: 0.4961 - val_loss: 0.3723 - val_output_loss: 0.3568 - val_aux_output_loss: 0.5114
    Epoch 20/100
    11610/11610 [==============================] - 1s 123us/sample - loss: 0.3536 - output_loss: 0.3385 - aux_output_loss: 0.4887 - val_loss: 0.3662 - val_output_loss: 0.3497 - val_aux_output_loss: 0.5153
    Epoch 21/100
    11610/11610 [==============================] - 1s 123us/sample - loss: 0.3511 - output_loss: 0.3367 - aux_output_loss: 0.4821 - val_loss: 0.3595 - val_output_loss: 0.3440 - val_aux_output_loss: 0.5000
    Epoch 22/100
    11610/11610 [==============================] - 1s 122us/sample - loss: 0.3472 - output_loss: 0.3329 - aux_output_loss: 0.4753 - val_loss: 0.3638 - val_output_loss: 0.3481 - val_aux_output_loss: 0.5057
    Epoch 23/100
    11610/11610 [==============================] - 1s 127us/sample - loss: 0.3473 - output_loss: 0.3336 - aux_output_loss: 0.4704 - val_loss: 0.3553 - val_output_loss: 0.3404 - val_aux_output_loss: 0.4899
    Epoch 24/100
    11610/11610 [==============================] - 1s 122us/sample - loss: 0.3609 - output_loss: 0.3491 - aux_output_loss: 0.4669 - val_loss: 0.3609 - val_output_loss: 0.3460 - val_aux_output_loss: 0.4948
    Epoch 25/100
    11610/11610 [==============================] - 1s 124us/sample - loss: 0.3480 - output_loss: 0.3355 - aux_output_loss: 0.4606 - val_loss: 0.3536 - val_output_loss: 0.3398 - val_aux_output_loss: 0.4788
    Epoch 26/100
    11610/11610 [==============================] - 1s 125us/sample - loss: 0.3425 - output_loss: 0.3301 - aux_output_loss: 0.4560 - val_loss: 0.3537 - val_output_loss: 0.3400 - val_aux_output_loss: 0.4771
    Epoch 27/100
    11610/11610 [==============================] - 1s 121us/sample - loss: 0.3400 - output_loss: 0.3275 - aux_output_loss: 0.4523 - val_loss: 0.3524 - val_output_loss: 0.3393 - val_aux_output_loss: 0.4708
    Epoch 28/100
    11610/11610 [==============================] - 1s 123us/sample - loss: 0.3384 - output_loss: 0.3264 - aux_output_loss: 0.4479 - val_loss: 0.3522 - val_output_loss: 0.3394 - val_aux_output_loss: 0.4674
    Epoch 29/100
    11610/11610 [==============================] - 1s 122us/sample - loss: 0.3405 - output_loss: 0.3290 - aux_output_loss: 0.4436 - val_loss: 0.5467 - val_output_loss: 0.5551 - val_aux_output_loss: 0.4704
    Epoch 30/100
    11610/11610 [==============================] - 1s 128us/sample - loss: 0.3459 - output_loss: 0.3351 - aux_output_loss: 0.4452 - val_loss: 0.3531 - val_output_loss: 0.3407 - val_aux_output_loss: 0.4656
    Epoch 31/100
    11610/11610 [==============================] - 1s 121us/sample - loss: 0.3578 - output_loss: 0.3488 - aux_output_loss: 0.4387 - val_loss: 0.4585 - val_output_loss: 0.4555 - val_aux_output_loss: 0.4848
    Epoch 32/100
    11610/11610 [==============================] - 1s 124us/sample - loss: 0.3421 - output_loss: 0.3309 - aux_output_loss: 0.4416 - val_loss: 0.3526 - val_output_loss: 0.3391 - val_aux_output_loss: 0.4743
    Epoch 33/100
    11610/11610 [==============================] - 1s 122us/sample - loss: 0.3358 - output_loss: 0.3249 - aux_output_loss: 0.4337 - val_loss: 0.3513 - val_output_loss: 0.3377 - val_aux_output_loss: 0.4738
    Epoch 34/100
    11610/11610 [==============================] - 1s 123us/sample - loss: 0.3318 - output_loss: 0.3210 - aux_output_loss: 0.4284 - val_loss: 0.3472 - val_output_loss: 0.3351 - val_aux_output_loss: 0.4565
    Epoch 35/100
    11610/11610 [==============================] - 1s 125us/sample - loss: 0.3346 - output_loss: 0.3244 - aux_output_loss: 0.4265 - val_loss: 0.3694 - val_output_loss: 0.3592 - val_aux_output_loss: 0.4616
    Epoch 36/100
    11610/11610 [==============================] - 1s 123us/sample - loss: 0.3301 - output_loss: 0.3198 - aux_output_loss: 0.4226 - val_loss: 0.3486 - val_output_loss: 0.3372 - val_aux_output_loss: 0.4512
    Epoch 37/100
    11610/11610 [==============================] - 1s 129us/sample - loss: 0.3293 - output_loss: 0.3190 - aux_output_loss: 0.4217 - val_loss: 0.3461 - val_output_loss: 0.3353 - val_aux_output_loss: 0.4433
    Epoch 38/100
    11610/11610 [==============================] - 2s 136us/sample - loss: 0.3298 - output_loss: 0.3200 - aux_output_loss: 0.4189 - val_loss: 0.3552 - val_output_loss: 0.3442 - val_aux_output_loss: 0.4542
    Epoch 39/100
    11610/11610 [==============================] - 1s 117us/sample - loss: 0.3272 - output_loss: 0.3174 - aux_output_loss: 0.4149 - val_loss: 0.3474 - val_output_loss: 0.3364 - val_aux_output_loss: 0.4460
    Epoch 40/100
    11610/11610 [==============================] - 1s 116us/sample - loss: 0.3307 - output_loss: 0.3215 - aux_output_loss: 0.4128 - val_loss: 0.3488 - val_output_loss: 0.3378 - val_aux_output_loss: 0.4481
    Epoch 41/100
    11610/11610 [==============================] - 1s 119us/sample - loss: 0.3254 - output_loss: 0.3159 - aux_output_loss: 0.4108 - val_loss: 0.3389 - val_output_loss: 0.3280 - val_aux_output_loss: 0.4366
    Epoch 42/100
    11610/11610 [==============================] - 1s 118us/sample - loss: 0.3243 - output_loss: 0.3151 - aux_output_loss: 0.4083 - val_loss: 0.3425 - val_output_loss: 0.3320 - val_aux_output_loss: 0.4375
    Epoch 43/100
    11610/11610 [==============================] - 1s 116us/sample - loss: 0.3230 - output_loss: 0.3139 - aux_output_loss: 0.4046 - val_loss: 0.3421 - val_output_loss: 0.3319 - val_aux_output_loss: 0.4341
    Epoch 44/100
    11610/11610 [==============================] - 1s 118us/sample - loss: 0.3237 - output_loss: 0.3147 - aux_output_loss: 0.4045 - val_loss: 0.3503 - val_output_loss: 0.3399 - val_aux_output_loss: 0.4451
    Epoch 45/100
    11610/11610 [==============================] - 1s 124us/sample - loss: 0.3243 - output_loss: 0.3156 - aux_output_loss: 0.4025 - val_loss: 0.3384 - val_output_loss: 0.3265 - val_aux_output_loss: 0.4454
    Epoch 46/100
    11610/11610 [==============================] - 1s 118us/sample - loss: 0.3211 - output_loss: 0.3121 - aux_output_loss: 0.4014 - val_loss: 0.3403 - val_output_loss: 0.3296 - val_aux_output_loss: 0.4372
    Epoch 47/100
    11610/11610 [==============================] - 1s 120us/sample - loss: 0.3213 - output_loss: 0.3126 - aux_output_loss: 0.3990 - val_loss: 0.3383 - val_output_loss: 0.3281 - val_aux_output_loss: 0.4304
    Epoch 48/100
    11610/11610 [==============================] - 1s 118us/sample - loss: 0.3202 - output_loss: 0.3115 - aux_output_loss: 0.3977 - val_loss: 0.3453 - val_output_loss: 0.3349 - val_aux_output_loss: 0.4390
    Epoch 49/100
    11610/11610 [==============================] - 1s 118us/sample - loss: 0.3217 - output_loss: 0.3134 - aux_output_loss: 0.3958 - val_loss: 0.3408 - val_output_loss: 0.3315 - val_aux_output_loss: 0.4248
    Epoch 50/100
    11610/11610 [==============================] - 1s 119us/sample - loss: 0.3194 - output_loss: 0.3113 - aux_output_loss: 0.3942 - val_loss: 0.3392 - val_output_loss: 0.3298 - val_aux_output_loss: 0.4237
    Epoch 51/100
    11610/11610 [==============================] - 1s 120us/sample - loss: 0.3312 - output_loss: 0.3240 - aux_output_loss: 0.3967 - val_loss: 0.3342 - val_output_loss: 0.3232 - val_aux_output_loss: 0.4335
    Epoch 52/100
    11610/11610 [==============================] - 1s 126us/sample - loss: 0.3184 - output_loss: 0.3100 - aux_output_loss: 0.3933 - val_loss: 0.3334 - val_output_loss: 0.3234 - val_aux_output_loss: 0.4238
    Epoch 53/100
    11610/11610 [==============================] - 1s 121us/sample - loss: 0.3189 - output_loss: 0.3108 - aux_output_loss: 0.3924 - val_loss: 0.3359 - val_output_loss: 0.3262 - val_aux_output_loss: 0.4237
    Epoch 54/100
    11610/11610 [==============================] - 1s 120us/sample - loss: 0.3179 - output_loss: 0.3098 - aux_output_loss: 0.3902 - val_loss: 0.3361 - val_output_loss: 0.3263 - val_aux_output_loss: 0.4244
    Epoch 55/100
    11610/11610 [==============================] - 1s 119us/sample - loss: 0.3171 - output_loss: 0.3090 - aux_output_loss: 0.3893 - val_loss: 0.4166 - val_output_loss: 0.4169 - val_aux_output_loss: 0.4140
    Epoch 56/100
    11610/11610 [==============================] - 1s 122us/sample - loss: 0.3214 - output_loss: 0.3138 - aux_output_loss: 0.3901 - val_loss: 0.3342 - val_output_loss: 0.3233 - val_aux_output_loss: 0.4320
    Epoch 57/100
    11610/11610 [==============================] - 1s 122us/sample - loss: 0.3148 - output_loss: 0.3070 - aux_output_loss: 0.3864 - val_loss: 0.3390 - val_output_loss: 0.3300 - val_aux_output_loss: 0.4197



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
root_logdir = Path('tf_logs')

def get_run_logdir():
    import time
    run_id = time.strftime('run_%Y_%m_%d_%H_%M_%S')
    return root_logdir.joinpath(run_id).as_posix()
```


```python
get_run_logdir()
```




    'tf_logs/run_2020_01_02_09_14_53'



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
    1200/1200 [==============================] - 1s 873us/sample - loss: 1.3061 - accuracy: 0.5925 - val_loss: 0.6875 - val_accuracy: 0.7667
    Epoch 2/30
    1200/1200 [==============================] - 0s 177us/sample - loss: 0.5643 - accuracy: 0.8217 - val_loss: 0.4844 - val_accuracy: 0.8567
    Epoch 3/30
    1200/1200 [==============================] - 0s 192us/sample - loss: 0.4466 - accuracy: 0.8567 - val_loss: 0.4179 - val_accuracy: 0.8733
    Epoch 4/30
    1200/1200 [==============================] - 0s 170us/sample - loss: 0.3986 - accuracy: 0.8758 - val_loss: 0.3851 - val_accuracy: 0.8900
    Epoch 5/30
    1200/1200 [==============================] - 0s 203us/sample - loss: 0.3716 - accuracy: 0.8775 - val_loss: 0.3723 - val_accuracy: 0.8800
    Epoch 6/30
    1200/1200 [==============================] - 0s 289us/sample - loss: 0.3552 - accuracy: 0.8808 - val_loss: 0.3558 - val_accuracy: 0.8800
    Epoch 7/30
    1200/1200 [==============================] - 0s 254us/sample - loss: 0.3410 - accuracy: 0.8892 - val_loss: 0.3442 - val_accuracy: 0.8867
    Epoch 8/30
    1200/1200 [==============================] - 0s 166us/sample - loss: 0.3305 - accuracy: 0.8858 - val_loss: 0.3381 - val_accuracy: 0.8867
    Epoch 9/30
    1200/1200 [==============================] - 0s 139us/sample - loss: 0.3246 - accuracy: 0.8908 - val_loss: 0.3340 - val_accuracy: 0.8867
    Epoch 10/30
    1200/1200 [==============================] - 0s 152us/sample - loss: 0.3194 - accuracy: 0.8900 - val_loss: 0.3292 - val_accuracy: 0.8867
    Epoch 11/30
    1200/1200 [==============================] - 0s 140us/sample - loss: 0.3132 - accuracy: 0.8875 - val_loss: 0.3338 - val_accuracy: 0.8900
    Epoch 12/30
    1200/1200 [==============================] - 0s 140us/sample - loss: 0.3099 - accuracy: 0.8867 - val_loss: 0.3277 - val_accuracy: 0.8833
    Epoch 13/30
    1200/1200 [==============================] - 0s 148us/sample - loss: 0.3070 - accuracy: 0.8892 - val_loss: 0.3226 - val_accuracy: 0.8867
    Epoch 14/30
    1200/1200 [==============================] - 0s 150us/sample - loss: 0.2998 - accuracy: 0.8900 - val_loss: 0.3222 - val_accuracy: 0.8800
    Epoch 15/30
    1200/1200 [==============================] - 0s 166us/sample - loss: 0.2964 - accuracy: 0.8933 - val_loss: 0.3171 - val_accuracy: 0.8867
    Epoch 16/30
    1200/1200 [==============================] - 0s 165us/sample - loss: 0.2941 - accuracy: 0.8908 - val_loss: 0.3169 - val_accuracy: 0.8867
    Epoch 17/30
    1200/1200 [==============================] - 0s 221us/sample - loss: 0.2929 - accuracy: 0.8950 - val_loss: 0.3206 - val_accuracy: 0.8833
    Epoch 18/30
    1200/1200 [==============================] - 0s 191us/sample - loss: 0.2897 - accuracy: 0.8942 - val_loss: 0.3168 - val_accuracy: 0.8833
    Epoch 19/30
    1200/1200 [==============================] - 0s 173us/sample - loss: 0.2872 - accuracy: 0.8925 - val_loss: 0.3140 - val_accuracy: 0.8900
    Epoch 20/30
    1200/1200 [==============================] - 0s 161us/sample - loss: 0.2860 - accuracy: 0.8958 - val_loss: 0.3240 - val_accuracy: 0.8867
    Epoch 21/30
    1200/1200 [==============================] - 0s 159us/sample - loss: 0.2840 - accuracy: 0.8958 - val_loss: 0.3144 - val_accuracy: 0.8867
    Epoch 22/30
    1200/1200 [==============================] - 0s 171us/sample - loss: 0.2830 - accuracy: 0.8950 - val_loss: 0.3133 - val_accuracy: 0.8900
    Epoch 23/30
    1200/1200 [==============================] - 0s 178us/sample - loss: 0.2808 - accuracy: 0.8942 - val_loss: 0.3127 - val_accuracy: 0.8867
    Epoch 24/30
    1200/1200 [==============================] - 0s 170us/sample - loss: 0.2789 - accuracy: 0.8925 - val_loss: 0.3158 - val_accuracy: 0.8867
    Epoch 25/30
    1200/1200 [==============================] - 0s 165us/sample - loss: 0.2771 - accuracy: 0.8950 - val_loss: 0.3104 - val_accuracy: 0.8900
    Epoch 26/30
    1200/1200 [==============================] - 0s 166us/sample - loss: 0.2772 - accuracy: 0.8908 - val_loss: 0.3112 - val_accuracy: 0.8933
    Epoch 27/30
    1200/1200 [==============================] - 0s 152us/sample - loss: 0.2745 - accuracy: 0.8983 - val_loss: 0.3091 - val_accuracy: 0.8900
    Epoch 28/30
    1200/1200 [==============================] - 0s 183us/sample - loss: 0.2724 - accuracy: 0.8958 - val_loss: 0.3103 - val_accuracy: 0.8933
    Epoch 29/30
    1200/1200 [==============================] - 0s 173us/sample - loss: 0.2729 - accuracy: 0.8983 - val_loss: 0.3096 - val_accuracy: 0.8900
    Epoch 30/30
    1200/1200 [==============================] - 0s 162us/sample - loss: 0.2702 - accuracy: 0.9025 - val_loss: 0.3094 - val_accuracy: 0.8867



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
    1200/1200 [==============================] - 1s 893us/sample - loss: 2.0109 - accuracy: 0.1733 - val_loss: 1.7095 - val_accuracy: 0.2433
    Epoch 2/30
    1200/1200 [==============================] - 0s 203us/sample - loss: 1.5423 - accuracy: 0.3992 - val_loss: 1.4448 - val_accuracy: 0.4833
    Epoch 3/30
    1200/1200 [==============================] - 0s 180us/sample - loss: 1.2906 - accuracy: 0.5058 - val_loss: 1.2221 - val_accuracy: 0.5100
    Epoch 4/30
    1200/1200 [==============================] - 0s 181us/sample - loss: 1.0909 - accuracy: 0.5683 - val_loss: 1.0581 - val_accuracy: 0.5533
    Epoch 5/30
    1200/1200 [==============================] - 0s 188us/sample - loss: 0.9372 - accuracy: 0.6542 - val_loss: 0.9213 - val_accuracy: 0.6667
    Epoch 6/30
    1200/1200 [==============================] - 0s 207us/sample - loss: 0.8025 - accuracy: 0.7425 - val_loss: 0.7897 - val_accuracy: 0.7067
    Epoch 7/30
    1200/1200 [==============================] - 0s 186us/sample - loss: 0.6859 - accuracy: 0.7717 - val_loss: 0.6809 - val_accuracy: 0.7600
    Epoch 8/30
    1200/1200 [==============================] - 0s 179us/sample - loss: 0.5965 - accuracy: 0.8042 - val_loss: 0.5962 - val_accuracy: 0.7567
    Epoch 9/30
    1200/1200 [==============================] - 0s 189us/sample - loss: 0.5241 - accuracy: 0.8233 - val_loss: 0.5559 - val_accuracy: 0.7900
    Epoch 10/30
    1200/1200 [==============================] - 0s 171us/sample - loss: 0.4722 - accuracy: 0.8317 - val_loss: 0.4790 - val_accuracy: 0.8167
    Epoch 11/30
    1200/1200 [==============================] - 0s 185us/sample - loss: 0.4304 - accuracy: 0.8517 - val_loss: 0.4425 - val_accuracy: 0.8267
    Epoch 12/30
    1200/1200 [==============================] - 0s 181us/sample - loss: 0.4013 - accuracy: 0.8558 - val_loss: 0.4305 - val_accuracy: 0.8533
    Epoch 13/30
    1200/1200 [==============================] - 0s 183us/sample - loss: 0.3786 - accuracy: 0.8725 - val_loss: 0.3937 - val_accuracy: 0.8500
    Epoch 14/30
    1200/1200 [==============================] - 0s 175us/sample - loss: 0.3665 - accuracy: 0.8700 - val_loss: 0.3867 - val_accuracy: 0.8567
    Epoch 15/30
    1200/1200 [==============================] - 0s 181us/sample - loss: 0.3501 - accuracy: 0.8800 - val_loss: 0.3751 - val_accuracy: 0.8667
    Epoch 16/30
    1200/1200 [==============================] - 0s 189us/sample - loss: 0.3423 - accuracy: 0.8817 - val_loss: 0.3604 - val_accuracy: 0.8567
    Epoch 17/30
    1200/1200 [==============================] - 0s 179us/sample - loss: 0.3276 - accuracy: 0.8883 - val_loss: 0.3508 - val_accuracy: 0.8667
    Epoch 18/30
    1200/1200 [==============================] - 0s 183us/sample - loss: 0.3250 - accuracy: 0.8817 - val_loss: 0.3502 - val_accuracy: 0.8700
    Epoch 19/30
    1200/1200 [==============================] - 0s 182us/sample - loss: 0.3166 - accuracy: 0.8875 - val_loss: 0.3332 - val_accuracy: 0.8733
    Epoch 20/30
    1200/1200 [==============================] - 0s 167us/sample - loss: 0.3110 - accuracy: 0.8892 - val_loss: 0.3370 - val_accuracy: 0.8667
    Epoch 21/30
    1200/1200 [==============================] - 0s 249us/sample - loss: 0.3086 - accuracy: 0.8900 - val_loss: 0.3385 - val_accuracy: 0.8467
    Epoch 22/30
    1200/1200 [==============================] - 0s 198us/sample - loss: 0.3035 - accuracy: 0.8900 - val_loss: 0.3366 - val_accuracy: 0.8700
    Epoch 23/30
    1200/1200 [==============================] - 0s 187us/sample - loss: 0.3005 - accuracy: 0.8892 - val_loss: 0.3416 - val_accuracy: 0.8733
    Epoch 24/30
    1200/1200 [==============================] - 0s 171us/sample - loss: 0.2972 - accuracy: 0.8908 - val_loss: 0.3336 - val_accuracy: 0.8733
    Epoch 25/30
    1200/1200 [==============================] - 0s 190us/sample - loss: 0.2936 - accuracy: 0.8908 - val_loss: 0.3351 - val_accuracy: 0.8700
    Epoch 26/30
    1200/1200 [==============================] - 0s 157us/sample - loss: 0.2924 - accuracy: 0.8992 - val_loss: 0.3187 - val_accuracy: 0.8767
    Epoch 27/30
    1200/1200 [==============================] - 0s 182us/sample - loss: 0.2907 - accuracy: 0.8983 - val_loss: 0.3140 - val_accuracy: 0.8867
    Epoch 28/30
    1200/1200 [==============================] - 0s 173us/sample - loss: 0.2911 - accuracy: 0.8925 - val_loss: 0.3478 - val_accuracy: 0.8800
    Epoch 29/30
    1200/1200 [==============================] - 0s 184us/sample - loss: 0.2839 - accuracy: 0.8967 - val_loss: 0.3172 - val_accuracy: 0.8933
    Epoch 30/30
    1200/1200 [==============================] - 0s 188us/sample - loss: 0.2801 - accuracy: 0.9008 - val_loss: 0.3472 - val_accuracy: 0.8800



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
%load_ext tensorboard
%tensorboard --logdir=./tf_logs --port=6006
```

    The tensorboard extension is already loaded. To reload it, use:
      %reload_ext tensorboard



    Reusing TensorBoard on port 6006 (pid 10023), started 0:10:20 ago. (Use '!kill 10023' to kill it.)




<iframe id="tensorboard-frame-4ce1309d9507be43" width="100%" height="800" frameborder="0">
</iframe>
<script>
  (function() {
    const frame = document.getElementById("tensorboard-frame-4ce1309d9507be43");
    const url = new URL("/", window.location);
    url.port = 6006;
    frame.src = url;
  })();
</script>



## Fine-tuning neural network hyperparameters


```python

```
