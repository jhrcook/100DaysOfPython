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
model.add(keras.layers.Dense(300, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))
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
    dense (Dense)                (None, 300)               235500    
    _________________________________________________________________
    dense_1 (Dense)              (None, 100)               30100     
    _________________________________________________________________
    dense_2 (Dense)              (None, 10)                1010      
    =================================================================
    Total params: 266,610
    Trainable params: 266,610
    Non-trainable params: 0
    _________________________________________________________________


The list of layers can be accessed and specifically indexed by name.


```python
model.layers
```




    [<tensorflow.python.keras.layers.core.Flatten at 0x1a42727b90>,
     <tensorflow.python.keras.layers.core.Dense at 0x1a5dd8b8d0>,
     <tensorflow.python.keras.layers.core.Dense at 0x1a1bbd3650>,
     <tensorflow.python.keras.layers.core.Dense at 0x1a1bbd38d0>]




```python
model.get_layer('dense_1')
```




    <tensorflow.python.keras.layers.core.Dense at 0x1a1bbd3650>




```python
weights, biases = model.get_layer('dense_1').get_weights()
```


```python
weights
```




    array([[ 0.11562026, -0.09146738,  0.03914911, ..., -0.05768988,
            -0.08965203, -0.10332903],
           [ 0.03291359,  0.04758274, -0.06447461, ...,  0.11550095,
            -0.11767888, -0.11875456],
           [ 0.11963636, -0.06059965, -0.00824888, ..., -0.04262203,
             0.1223997 ,  0.01008546],
           ...,
           [ 0.12000973, -0.00090205,  0.04413744, ...,  0.0543538 ,
            -0.07552329, -0.07220218],
           [ 0.08202974,  0.10626879,  0.06814391, ..., -0.0877194 ,
            -0.05468229,  0.09630699],
           [ 0.04204778, -0.06469244,  0.00569629, ..., -0.01384209,
            -0.11549228, -0.00111773]], dtype=float32)




```python
biases
```




    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
          dtype=float32)



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
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))
```

    Train on 55000 samples, validate on 5000 samples
    Epoch 1/30
    55000/55000 [==============================] - 8s 151us/sample - loss: 0.6948 - accuracy: 0.7739 - val_loss: 0.5662 - val_accuracy: 0.7940
    Epoch 2/30
    55000/55000 [==============================] - 7s 135us/sample - loss: 0.4829 - accuracy: 0.8317 - val_loss: 0.4518 - val_accuracy: 0.8428
    Epoch 3/30
    55000/55000 [==============================] - 7s 134us/sample - loss: 0.4405 - accuracy: 0.8459 - val_loss: 0.4497 - val_accuracy: 0.8402
    Epoch 4/30
    55000/55000 [==============================] - 7s 135us/sample - loss: 0.4138 - accuracy: 0.8554 - val_loss: 0.4078 - val_accuracy: 0.8578
    Epoch 5/30
    55000/55000 [==============================] - 7s 135us/sample - loss: 0.3934 - accuracy: 0.8608 - val_loss: 0.3909 - val_accuracy: 0.8634
    Epoch 6/30
    55000/55000 [==============================] - 7s 136us/sample - loss: 0.3771 - accuracy: 0.8675 - val_loss: 0.3766 - val_accuracy: 0.8670
    Epoch 7/30
    55000/55000 [==============================] - 7s 136us/sample - loss: 0.3634 - accuracy: 0.8712 - val_loss: 0.3698 - val_accuracy: 0.8708
    Epoch 8/30
    55000/55000 [==============================] - 7s 136us/sample - loss: 0.3512 - accuracy: 0.8761 - val_loss: 0.3682 - val_accuracy: 0.8714
    Epoch 9/30
    55000/55000 [==============================] - 9s 164us/sample - loss: 0.3416 - accuracy: 0.8788 - val_loss: 0.3537 - val_accuracy: 0.8744
    Epoch 10/30
    55000/55000 [==============================] - 8s 141us/sample - loss: 0.3328 - accuracy: 0.8821 - val_loss: 0.3504 - val_accuracy: 0.8802
    Epoch 11/30
    55000/55000 [==============================] - 8s 140us/sample - loss: 0.3234 - accuracy: 0.8850 - val_loss: 0.3500 - val_accuracy: 0.8752
    Epoch 12/30
    55000/55000 [==============================] - 8s 144us/sample - loss: 0.3171 - accuracy: 0.8867 - val_loss: 0.3653 - val_accuracy: 0.8642
    Epoch 13/30
    55000/55000 [==============================] - 8s 149us/sample - loss: 0.3098 - accuracy: 0.8895 - val_loss: 0.3405 - val_accuracy: 0.8792
    Epoch 14/30
    55000/55000 [==============================] - 8s 148us/sample - loss: 0.3031 - accuracy: 0.8909 - val_loss: 0.3696 - val_accuracy: 0.8718
    Epoch 15/30
    55000/55000 [==============================] - 8s 145us/sample - loss: 0.2960 - accuracy: 0.8934 - val_loss: 0.3269 - val_accuracy: 0.8836
    Epoch 16/30
    55000/55000 [==============================] - 8s 143us/sample - loss: 0.2907 - accuracy: 0.8948 - val_loss: 0.3371 - val_accuracy: 0.8774
    Epoch 17/30
    55000/55000 [==============================] - 8s 145us/sample - loss: 0.2852 - accuracy: 0.8968 - val_loss: 0.3239 - val_accuracy: 0.8832
    Epoch 18/30
    55000/55000 [==============================] - 8s 144us/sample - loss: 0.2802 - accuracy: 0.8995 - val_loss: 0.3268 - val_accuracy: 0.8806
    Epoch 19/30
    55000/55000 [==============================] - 9s 158us/sample - loss: 0.2752 - accuracy: 0.9008 - val_loss: 0.3248 - val_accuracy: 0.8848
    Epoch 20/30
    55000/55000 [==============================] - 8s 145us/sample - loss: 0.2705 - accuracy: 0.9019 - val_loss: 0.3249 - val_accuracy: 0.8844
    Epoch 21/30
    55000/55000 [==============================] - 8s 145us/sample - loss: 0.2644 - accuracy: 0.9047 - val_loss: 0.3165 - val_accuracy: 0.8858
    Epoch 22/30
    55000/55000 [==============================] - 8s 149us/sample - loss: 0.2604 - accuracy: 0.9063 - val_loss: 0.3217 - val_accuracy: 0.8820
    Epoch 23/30
    55000/55000 [==============================] - 8s 147us/sample - loss: 0.2560 - accuracy: 0.9076 - val_loss: 0.3127 - val_accuracy: 0.8894
    Epoch 24/30
    55000/55000 [==============================] - 8s 149us/sample - loss: 0.2517 - accuracy: 0.9104 - val_loss: 0.3099 - val_accuracy: 0.8906
    Epoch 25/30
    55000/55000 [==============================] - 8s 148us/sample - loss: 0.2481 - accuracy: 0.9105 - val_loss: 0.3159 - val_accuracy: 0.8884
    Epoch 26/30
    55000/55000 [==============================] - 8s 149us/sample - loss: 0.2432 - accuracy: 0.9120 - val_loss: 0.3104 - val_accuracy: 0.8900
    Epoch 27/30
    55000/55000 [==============================] - 9s 164us/sample - loss: 0.2405 - accuracy: 0.9133 - val_loss: 0.3069 - val_accuracy: 0.8918
    Epoch 28/30
    55000/55000 [==============================] - 8s 147us/sample - loss: 0.2358 - accuracy: 0.9147 - val_loss: 0.3173 - val_accuracy: 0.8880
    Epoch 29/30
    55000/55000 [==============================] - 9s 161us/sample - loss: 0.2321 - accuracy: 0.9162 - val_loss: 0.3068 - val_accuracy: 0.8904
    Epoch 30/30
    55000/55000 [==============================] - 8s 144us/sample - loss: 0.2287 - accuracy: 0.9185 - val_loss: 0.3102 - val_accuracy: 0.8890



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




    [0.32945477788448335, 0.8841]



We can also use the model to make predictions.
Using the `predict()` method, we get a probability per class.


```python
X_new = fashion_preprocesser.transform(X_test[:3])
np.round(model.predict(X_new), 2)
```




    array([[0.  , 0.  , 0.  , 0.  , 0.  , 0.02, 0.  , 0.03, 0.  , 0.95],
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
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(1, activation='relu')
])

model.compile(loss='mean_squared_error', optimizer='sgd')

history = model.fit(X_train, y_train,
                    epochs=20,
                    validation_data=(X_valid, y_valid))
```

    Train on 11610 samples, validate on 3870 samples
    Epoch 1/20
    11610/11610 [==============================] - 2s 131us/sample - loss: 0.6884 - val_loss: 0.4887
    Epoch 2/20
    11610/11610 [==============================] - 1s 103us/sample - loss: 0.4448 - val_loss: 0.4605
    Epoch 3/20
    11610/11610 [==============================] - 1s 106us/sample - loss: 0.4064 - val_loss: 0.4276
    Epoch 4/20
    11610/11610 [==============================] - 1s 98us/sample - loss: 0.3884 - val_loss: 0.4057
    Epoch 5/20
    11610/11610 [==============================] - 1s 81us/sample - loss: 0.3796 - val_loss: 0.4043
    Epoch 6/20
    11610/11610 [==============================] - 1s 120us/sample - loss: 0.3718 - val_loss: 0.4000
    Epoch 7/20
    11610/11610 [==============================] - 1s 93us/sample - loss: 0.3670 - val_loss: 0.3934
    Epoch 8/20
    11610/11610 [==============================] - 1s 83us/sample - loss: 0.3610 - val_loss: 0.3886
    Epoch 9/20
    11610/11610 [==============================] - 1s 79us/sample - loss: 0.3785 - val_loss: 0.4102
    Epoch 10/20
    11610/11610 [==============================] - 1s 80us/sample - loss: 0.3650 - val_loss: 0.4145
    Epoch 11/20
    11610/11610 [==============================] - 2s 132us/sample - loss: 0.3555 - val_loss: 0.3898
    Epoch 12/20
    11610/11610 [==============================] - 1s 104us/sample - loss: 0.3513 - val_loss: 0.3822
    Epoch 13/20
    11610/11610 [==============================] - 1s 85us/sample - loss: 0.3491 - val_loss: 0.3758
    Epoch 14/20
    11610/11610 [==============================] - 1s 84us/sample - loss: 0.3437 - val_loss: 0.4791
    Epoch 15/20
    11610/11610 [==============================] - 1s 85us/sample - loss: 0.3421 - val_loss: 0.3696
    Epoch 16/20
    11610/11610 [==============================] - 1s 94us/sample - loss: 0.3381 - val_loss: 0.3761
    Epoch 17/20
    11610/11610 [==============================] - 1s 86us/sample - loss: 0.3354 - val_loss: 0.3704
    Epoch 18/20
    11610/11610 [==============================] - 1s 89us/sample - loss: 0.3325 - val_loss: 0.3644
    Epoch 19/20
    11610/11610 [==============================] - 1s 86us/sample - loss: 0.3291 - val_loss: 0.3626
    Epoch 20/20
    11610/11610 [==============================] - 1s 88us/sample - loss: 0.3265 - val_loss: 0.3682



```python
model.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_3 (Dense)              (None, 30)                270       
    _________________________________________________________________
    dense_4 (Dense)              (None, 50)                1550      
    _________________________________________________________________
    dense_5 (Dense)              (None, 1)                 51        
    =================================================================
    Total params: 1,871
    Trainable params: 1,871
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




    0.34637213358583374




```python
y_pred = model.predict(X_test)
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color='b', s=10, alpha=0.2)
plt.plot(np.linspace(0, 5, 10), np.linspace(0, 5, 10), 'k--')
plt.title('Evaluation of the regression MLP')
plt.xlabel('Real', fontsize=12)
plt.ylabel('Predicted', fontsize=12)
plt.show()
```


![png](homl_ch10_Introduction-to-artificial-neural-networks-with-keras_files/homl_ch10_Introduction-to-artificial-neural-networks-with-keras_49_0.png)


### Building complex models using the functional API


```python

```
