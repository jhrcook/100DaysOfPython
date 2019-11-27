# Chapter 3. Classification


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(0)

plt.style.use('seaborn-whitegrid')
plt.rc('figure', figsize=(8, 5), facecolor='white')
```

## MNIST

Often considered the "Hello World" of ML, Scikit-Learn offers helper functions for downloading the MNIST 


```python
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', data_home='assets/scikit_learn_data/')
```


```python
mnist.data
```




    array([[0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           ...,
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.]])




```python
mnist.DESCR
```




    "**Author**: Yann LeCun, Corinna Cortes, Christopher J.C. Burges  \n**Source**: [MNIST Website](http://yann.lecun.com/exdb/mnist/) - Date unknown  \n**Please cite**:  \n\nThe MNIST database of handwritten digits with 784 features, raw data available at: http://yann.lecun.com/exdb/mnist/. It can be split in a training set of the first 60,000 examples, and a test set of 10,000 examples  \n\nIt is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image. It is a good database for people who want to try learning techniques and pattern recognition methods on real-world data while spending minimal efforts on preprocessing and formatting. The original black and white (bilevel) images from NIST were size normalized to fit in a 20x20 pixel box while preserving their aspect ratio. The resulting images contain grey levels as a result of the anti-aliasing technique used by the normalization algorithm. the images were centered in a 28x28 image by computing the center of mass of the pixels, and translating the image so as to position this point at the center of the 28x28 field.  \n\nWith some classification methods (particularly template-based methods, such as SVM and K-nearest neighbors), the error rate improves when the digits are centered by bounding box rather than center of mass. If you do this kind of pre-processing, you should report it in your publications. The MNIST database was constructed from NIST's NIST originally designated SD-3 as their training set and SD-1 as their test set. However, SD-3 is much cleaner and easier to recognize than SD-1. The reason for this can be found on the fact that SD-3 was collected among Census Bureau employees, while SD-1 was collected among high-school students. Drawing sensible conclusions from learning experiments requires that the result be independent of the choice of training set and test among the complete set of samples. Therefore it was necessary to build a new database by mixing NIST's datasets.  \n\nThe MNIST training set is composed of 30,000 patterns from SD-3 and 30,000 patterns from SD-1. Our test set was composed of 5,000 patterns from SD-3 and 5,000 patterns from SD-1. The 60,000 pattern training set contained examples from approximately 250 writers. We made sure that the sets of writers of the training set and test set were disjoint. SD-1 contains 58,527 digit images written by 500 different writers. In contrast to SD-3, where blocks of data from each writer appeared in sequence, the data in SD-1 is scrambled. Writer identities for SD-1 is available and we used this information to unscramble the writers. We then split SD-1 in two: characters written by the first 250 writers went into our new training set. The remaining 250 writers were placed in our test set. Thus we had two sets with nearly 30,000 examples each. The new training set was completed with enough examples from SD-3, starting at pattern # 0, to make a full set of 60,000 training patterns. Similarly, the new test set was completed with SD-3 examples starting at pattern # 35,000 to make a full set with 60,000 test patterns. Only a subset of 10,000 test images (5,000 from SD-1 and 5,000 from SD-3) is available on this site. The full 60,000 sample training set is available.\n\nDownloaded from openml.org."




```python
mnist.target
```




    array(['5', '0', '4', ..., '4', '5', '6'], dtype=object)




```python
X, y = mnist.data, mnist.target
```

Below is one of the data points shown as an image of its pixel intensities.


```python
def plot_mnist_image(digit):
    """Plot a digit from MNIST."""
    plt.imshow(digit.reshape(28, 28),  interpolation=r'nearest')

for i in range(5):
    plot_mnist_image(X[i])
    plt.show()
```


![png](homl_ch03_files/homl_ch03_9_0.png)



![png](homl_ch03_files/homl_ch03_9_1.png)



![png](homl_ch03_files/homl_ch03_9_2.png)



![png](homl_ch03_files/homl_ch03_9_3.png)



![png](homl_ch03_files/homl_ch03_9_4.png)


The MNIST data set is already split into training and testing sets at the 60,000 data point.


```python
a = 60000
X_train, X_test, y_train, y_test = X[:a], X[a:], y[:a], y[a:]
```

We also want to shuffle the training data set to ensure there is no order to it.


```python
idx = np.random.permutation(a)
X_train, y_train = X_train[idx], y_train[idx]
```

## Training a binary classifier

For a simpler example, we will try to create a binary classifier that can recognize 5's.


```python
y_train_5 = y_train == '5'
y_test_5 = y_test == '5'
```

A simple type of classifier is the Stochastic Gradient Descent (SGD) classifer, the `SGDClassifier` class in Scikit-Learn.


```python
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=0)
sgd_clf.fit(X_train, y_train_5)
```




    SGDClassifier(alpha=0.0001, average=False, class_weight=None,
                  early_stopping=False, epsilon=0.1, eta0=0.0, fit_intercept=True,
                  l1_ratio=0.15, learning_rate='optimal', loss='hinge',
                  max_iter=1000, n_iter_no_change=5, n_jobs=None, penalty='l2',
                  power_t=0.5, random_state=0, shuffle=True, tol=0.001,
                  validation_fraction=0.1, verbose=0, warm_start=False)




```python
# Try it on the first few training data points.
sgd_clf.predict(X[:5])
```




    array([ True, False, False, False, False])




```python
# Actual values.
y[:5]
```




    array(['5', '0', '4', '1', '9'], dtype=object)



## Performance measures

First, we can see how well the model performs using cross-validation.


```python
from sklearn.model_selection import cross_val_score

cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring='accuracy')
```




    array([0.9668 , 0.96315, 0.962  ])



While the accuracies seem high, all are above 90%, this might just happen because most values are not 5's.


```python
1 - sum(y_train_5) / len(y_train_5)
```




    0.90965



This shows that just guessing that the value is *not* 5 would result in an accuracy of 91%.
This is an example of why accuracy is often not the best descriptor of a classifier, especially on such a skewed data set.

### Confusion matrix

The confusion matrix shows the true and false positives and negatives.
The rows are the actual cases and the columns are predicted values.


```python
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

# Get the predictions for each cross validation.
y_train_predict = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

# Build the confusion matrix.
confusion_matrix(y_train_5, y_train_predict)
```




    array([[53561,  1018],
           [ 1143,  4278]])



### Precision and recall

The **precision** is the ratio of true positives to all predicted positives.

$$
\frac{TP}{TP + FP}
$$

The **recall** is the ratio of true positives to the total number of real positives (the sum of true positives and false negatives).

$$
\frac{TP}{TP + FN}
$$


```python
from sklearn.metrics import  precision_score, recall_score

# Measure the model's precision.
precision_score(y_train_5, y_train_predict)
```




    0.8077794561933535




```python
# Measure the model's accuracy.
recall_score(y_train_5, y_train_predict)
```




    0.789153292750415



The precision and recall can be combined into the $\text{F}_{1}$ score.

$$
\text{F}_1 = \frac{2}{\frac{1}{\text{precision}} + \frac{1}{\text{recall}}}
= 2 \times \frac{\text{precision} \times \text{recall}}{\text{precision} \times \text{recall}}
= \frac{TP}{TP + \frac{FN + FP}{2}}
$$


```python
from sklearn.metrics import f1_score

f1_score(y_train_5, y_train_predict)
```




    0.7983577493701595



### Precision/recall tradeoff



Increasing precision reduces recall and vice versa.

The SGD classifier computes a *decision function* and then uses a threshold to decide to report positive or negative.
We can get these values directly by using the `decision_function()` method on the classifier object.


```python
sgd_clf.decision_function([X_train[1]])
```




    array([-11758.50171933])



We can get the values for all values by running the CV but asking for these decision function values.


```python
y_scores = cross_val_predict(sgd_clf,
                             X_train,
                             y_train_5, 
                             cv=3, 
                             method='decision_function')
len(y_scores)
```




    60000



We can create the precision-recall curve over various thresholds.


```python
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], 'b-', label='Precision')
    plt.plot(thresholds, recalls[:-1], 'g-', label='Recall')
    plt.xlabel('Threshold')
    plt.title('Precision and recall curves for different thresholds')
    plt.legend(loc='best'),
    plt.ylim([0, 1])
    
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()
```


![png](homl_ch03_files/homl_ch03_37_0.png)



```python
def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, 'r-')
    plt.title('Precision vs. Recall')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0, 1])
    plt.xlim([0, 1])

plot_precision_vs_recall(precisions, recalls)
```


![png](homl_ch03_files/homl_ch03_38_0.png)


### The ROC curve

The ROC curve plots the *true positive* rate over the *false positive* rate.
Ideally, the curve stays as far from the $y = x$ line as possible.
A measure of this is the *area under the curve*, AUC, of the ROC.
1 is a perfect score, where 0.5 means the model was no better than a random guess.


```python
from sklearn.metrics import roc_curve


def plot_roc_curve(fpr, tpr, label=None):
    """Plot the ROC curve given the FPR and TPR for a model."""
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')  # black dotted line along y = x
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')

    
# Calculate and plot the FPR and TPC values.
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
plot_roc_curve(fpr, tpr)
plt.show()
```


![png](homl_ch03_files/homl_ch03_41_0.png)



```python
from sklearn.metrics import  roc_auc_score

# Measure the AUC of the ROC.
roc_auc_score(y_train_5, y_scores)
```




    0.9667389622712784



We now will train a random forest classifier to compare against the SGD.
There is no `decision_function()` method available for this model, instead, it provides the probabilities that each data point is part of each possible class.
This is found in the `predict_proba()` method.


```python
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=0, n_estimators=10)
y_probas_forest = cross_val_predict(forest_clf, 
                                    X_train, 
                                    y_train_5, 
                                    cv=3, 
                                    method='predict_proba')
```

The positive class's probability will be the score.


```python
y_scores_forest  = y_probas_forest[:, 1]
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)

plt.plot(fpr, tpr, 'g:', label='SGD')
plot_roc_curve(fpr_forest, tpr_forest, label='Random Forest')
plt.legend(loc='best')
plt.show()
```


![png](homl_ch03_files/homl_ch03_46_0.png)



```python
roc_auc_score(y_train_5, y_scores_forest)
```




    0.993244900251192



## Multiclass classification

There are ways of using multiple binary classifiers for a multiclass classification.
*One-versus-all* (OvA) is where each classifier is trained to recognize just one class. When classifying a new input, the classifier with the highest score wins.
*One-versus-one* (OvO) is where each classifier is trained to discern between a pair of classes.
When classifying a new input, the class that wins the most duels wins overall.

Scikit-Learn automatically detects when a binary classifier is used with multiple classes and uses OvA for all except for SVM which uses OvO (for performance reasons).


```python
sgd_clf.fit(X_train, y_train)
```


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-29-618cbbb8cdf1> in <module>
    ----> 1 sgd_clf.fit(X_train, y_train)
    

    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/sklearn/linear_model/stochastic_gradient.py in fit(self, X, y, coef_init, intercept_init, sample_weight)
        711                          loss=self.loss, learning_rate=self.learning_rate,
        712                          coef_init=coef_init, intercept_init=intercept_init,
    --> 713                          sample_weight=sample_weight)
        714 
        715 


    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/sklearn/linear_model/stochastic_gradient.py in _fit(self, X, y, alpha, C, loss, learning_rate, coef_init, intercept_init, sample_weight)
        552 
        553         self._partial_fit(X, y, alpha, C, loss, learning_rate, self.max_iter,
    --> 554                           classes, sample_weight, coef_init, intercept_init)
        555 
        556         if (self.tol is not None and self.tol > -np.inf


    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/sklearn/linear_model/stochastic_gradient.py in _partial_fit(self, X, y, alpha, C, loss, learning_rate, max_iter, classes, sample_weight, coef_init, intercept_init)
        507                                  learning_rate=learning_rate,
        508                                  sample_weight=sample_weight,
    --> 509                                  max_iter=max_iter)
        510         elif n_classes == 2:
        511             self._fit_binary(X, y, alpha=alpha, C=C,


    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/sklearn/linear_model/stochastic_gradient.py in _fit_multiclass(self, X, y, alpha, C, learning_rate, sample_weight, max_iter)
        613                                 validation_mask=validation_mask,
        614                                 random_state=seed)
    --> 615             for i, seed in enumerate(seeds))
        616 
        617         # take the maximum of n_iter_ over every binary fit


    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/joblib/parallel.py in __call__(self, iterable)
       1004                 self._iterating = self._original_iterator is not None
       1005 
    -> 1006             while self.dispatch_one_batch(iterator):
       1007                 pass
       1008 


    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/joblib/parallel.py in dispatch_one_batch(self, iterator)
        832                 return False
        833             else:
    --> 834                 self._dispatch(tasks)
        835                 return True
        836 


    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/joblib/parallel.py in _dispatch(self, batch)
        751         with self._lock:
        752             job_idx = len(self._jobs)
    --> 753             job = self._backend.apply_async(batch, callback=cb)
        754             # A job can complete so quickly than its callback is
        755             # called before we get here, causing self._jobs to


    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/joblib/_parallel_backends.py in apply_async(self, func, callback)
        199     def apply_async(self, func, callback=None):
        200         """Schedule a func to be run"""
    --> 201         result = ImmediateResult(func)
        202         if callback:
        203             callback(result)


    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/joblib/_parallel_backends.py in __init__(self, batch)
        580         # Don't delay the application, to avoid keeping the input
        581         # arguments in memory
    --> 582         self.results = batch()
        583 
        584     def get(self):


    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/joblib/parallel.py in __call__(self)
        254         with parallel_backend(self._backend, n_jobs=self._n_jobs):
        255             return [func(*args, **kwargs)
    --> 256                     for func, args, kwargs in self.items]
        257 
        258     def __len__(self):


    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/joblib/parallel.py in <listcomp>(.0)
        254         with parallel_backend(self._backend, n_jobs=self._n_jobs):
        255             return [func(*args, **kwargs)
    --> 256                     for func, args, kwargs in self.items]
        257 
        258     def __len__(self):


    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/sklearn/linear_model/stochastic_gradient.py in fit_binary(est, i, X, y, alpha, C, learning_rate, max_iter, pos_weight, neg_weight, sample_weight, validation_mask, random_state)
        411                            pos_weight, neg_weight,
        412                            learning_rate_type, est.eta0,
    --> 413                            est.power_t, est.t_, intercept_decay)
        414 
        415     else:


    KeyboardInterrupt: 



```python
some_digit = X_train[1]
sgd_clf.predict([some_digit])
```


```python
y_train[1]
```

Under the hood, Scikit-Learn trained 10 SGD classifiers, one for each digit, then compared their scores for the predictions.
We can see this directly by retrieving the decision function.


```python
some_digit_scores = sgd_clf.decision_function([some_digit])
some_digit_scores
```


```python
sgd_clf.classes_[np.argmax(some_digit_scores)]
```

Here is an example of using a random forest for multiclass classification.


```python
forest_clf.fit(X_train, y_train)
forest_clf.predict([some_digit])
```


```python
# Manually extract the class with the highest prediction.
forest_clf.classes_[np.argmax(forest_clf.predict_proba([some_digit]))]
```

These classifiers can be evaluated using CV, too.


```python
cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring='accuracy')
```


```python
cross_val_score(forest_clf, X_train, y_train, cv=3, scoring='accuracy')
```

### Error analysis

Error analysis should be done after selecting and tuning a shortlist of models and increasing their performance through feature selection and preparation.

We can begin by using a confusion matrix to identify where the errors were made.


```python
y_train_pred = cross_val_predict(forest_clf, X_train, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
conf_mx
```


```python
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.xlabel('Predicted')
plt.ylabel('Real')
plt.show()
```

Since the diagonal is the lightest region, the confusion matrix looks good.
Focusing on the error rates, instead though, can show which digits the classifier has the most trouble with.


```python
# Divide the rows by the number of samples of that digit.
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums

# Fill the diagonal with 0 to focus on the errors.
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.xlabel('Predicted')
plt.ylabel('Real')
plt.show()
```

From this visualization, it is clear that the random forest has assigned a lot of 5's as 3's and 9's as 4's.

### Multilabel classification

A classifier that outputs multiple binary labels is a *multilabel classifier*.
Here is a simple example to declare if each digit is a  7, 8, or 9 and  whether the digit is odd.


```python
from sklearn.neighbors import KNeighborsClassifier

# New labels for greater than 6 and if odd.
y_train_large = y_train.astype('int') >= 7
y_train_odd = y_train.astype('int') % 2 == 1

y_multilabel = np.c_[y_train_large, y_train_odd]
y_multilabel
```


```python
# A KNN multilabel classifier.
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)

# Predict on a value.
some_digit = X_train[1]
knn_clf.predict([some_digit])
```


```python
y_train[1]
```

Evaluation of a multilabel classifier depends on the requirements and goals of the project.
One method is to compute the $\text{F}_1$ score across all labels.


```python
y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
f1_score(y_multilabel, y_train_knn_pred, average='macro')
```

### Multioutput classification

*Multioutput classification* is a generalization of multilabel classification where each label is multiclass.

As an example, we will build a "denoiser", classifier that  removes noise from the digit images.
This is multilabel because each pixel will be evaluated, and each label can have multiple input values within the range 0 through 255.


```python
train_noise = np.random.randint(0, 100, (len(X_train), 784))
test_noise = np.random.randint(0, 100, (len(X_test), 784))

X_train_mod = X_train + train_noise
X_test_mod = X_test + test_noise

y_train_mod = X_train
y_test_mod = X_test
```


```python
# Show an example of the noisey training image and its label.
fig = plt.figure()

ax1 = fig.add_subplot(1, 2, 1)
plot_mnist_image(X_train_mod[0])
plt.show()

ax2 = fig.add_subplot(1, 2, 2)
plot_mnist_image(y_train_mod[0])
plt.show()
```


```python
# Train a KNN to denoise.
knn_clf.fit(X_train_mod, y_train_mod)

# Show the result of denoising the first image.
example_clean_digit = forest_clf.predict([X_train_mod[0]])
plot_mnist_image(example_clean_digit)
```


```python

```
