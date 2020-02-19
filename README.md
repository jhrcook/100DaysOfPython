# 100 Days of Python

![100DaysOfCodePython](https://img.shields.io/badge/100DaysOfCode-Python-3776AB.svg?style=flat&logo=python)
[![jhc github](https://img.shields.io/badge/GitHub-jhrcook-lightgrey.svg?style=flat&logo=github)](https://github.com/jhrcook)
[![jhc twitter](https://img.shields.io/badge/Twitter-@JoshDoesA-00aced.svg?style=flat&logo=twitter)](https://twitter.com/JoshDoesa)
[![jhc website](https://img.shields.io/badge/Website-Joshua_Cook-5087B2.svg?style=flat&logo=telegram)](https://joshuacook.netlify.com)

**Start Date: October 21, 2019  
End Date: January 30, 2020**

## Sources

To begin, I worked through [*Python for Data Analysis, 2nd Edition*](http://shop.oreilly.com/product/0636920050896.do).
The notebooks can be found [here](./PythonForDataAnalysis/).

I am currently working through [*Hands-On Machine Learning with Scikit-Learn, Keras, and Tensorflow*](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/), using the new seconnd edition of the book that has been updated for TF 2.0.
The notebooks can be found [here](./HandsOnMachineLearningWithScikitLearnAndTensorFlow/).

I took a brief, but important, interlude to learn how to effectively use Matplotlib, Seaborn, and Plotly, so that I can use Python for my research.
Each library has its own notebook [here](./Plotting-tutorials/).

Here is a running list of other tutorials and sources I want to use:

* [Python 101: iterators, generators, coroutines](https://www.integralist.co.uk/posts/python-generators/)
* [SymPy](https://docs.sympy.org/1.5.1/tutorial/index.html)
* [SciPy](https://docs.scipy.org/doc/scipy/reference/tutorial/index.html)


## Interesting Images

During my progress through my learning materials, I have created a lot of fun and interesting data visualizations.
I have documented some of my favorites [here](cool-plots-and-viz.md).


## Creating the environment

I am using the Anaconda distribution of Python 3.7, downloaded from [here](https://www.anaconda.com/distribution/).
I created a virtual environment for this repository called "daysOfCode-env".

```bash
conda create --name "daysOfCode-env" python=3.7
```

After that, I needed to initialize conda for `zsh`.

```bash
conda init zsh
```

After that, I could activate the virtual environment.

```bash
conda activate daysOfCode-env
```

I also added [Jupyter notebook extensions](https://github.com/ipython-contrib/jupyter_contrib_nbextensions) and the [Jupyter Nbextensions Configurator](https://github.com/Jupyter-contrib/jupyter_nbextensions_configurator).

```bash
# Jupyter notebook extensions
conda install -c conda-forge jupyter_contrib_nbextensions
jupyter contrib nbextension install --user

# Jupyter Nbextensions Configurator
conda install -c conda-forge jupyter_nbextensions_configurator
```


## Log

**Day 1 - October 21, 2019:**
I got Anaconda installed, my virtual environment set up, and began working through Python for Data Analysis (PDA).
I am also just getting used to VS Code.

**Day 2 - October 22, 2019:**
I am getting more confortable with VS Code and Jupyter notebooks.
I continued working through PDA and learned about some of the basic Python types.

**Day 3 - October 23, 2019:**
I learned about control flow in Python using `if`, `elif`, and `else`.
I also covered tuples and lists.

**Day 4 - October 24, 2019:**
I learned about dictionaries, sets, and comprehensions.

**Day 5 - October 25, 2019:**
I learned about functions, iterators, generators, and error handling as I continue my journey through the basics of Python.

**Day 6 - October 26, 2019:**
I finished up Ch. 3 learning about how to open and read files.
I then began Ch. 4 learning about NumPy and `ndarray`.

**Day 7 - October 27, 2019:**
I continued learning about using ndarrys in NumPy.
We covered indexing and *ufuncs*.

**Day 8 - October 28, 2019:**
I finished the chapter on NumPy learning more about how vectorization is implemented in the library.
We finished the chapter with an example of using NumPy to easily simulate many thousands of random walks simultaneously.

**Day 9 - October 29, 2019:**
I began learning about pandas DataFrames, mainly just the basics.

**Day 10 - October 30, 2019:**
I learned more about indexing DataFrame objects using positional index, column and row names, and booleans data structures.

**Day 11 - October 31, 2019:**
The book is not introducing arithmetic using Series and DataFrames.
They will definitely take some practice to get used to.

**Day 12 - November 1, 2019:**
I am still learning the basics of pandas.
Today, we went over ranking, sorting, and applying functions over column and rows.

**Day 13 - November 2, 2019:**
I finally finished the chapter introducing pandas.
We finished up by learning about computing descriptive and summary statistics on Series and DataFrames, including an example analysis using financial data.

**Day 14 - November 3, 2019:**
I began chapter 6 on reading and writing data.
So far, I am still learning about the various ways to read in files from disk.

**Day 15 - November 04, 2019:**
I learned about reading in HTML, XML, HDF5, and Excel data.

**Day 16 - November 05, 2019:**
We breifly covered interacting with web APIs to finish up the data reading and writing chapter.
I then began Chapter 7 on data cleaning and prepraration where we have gone over dropping and filling in missing values.

**Day 17 - November 06, 2019:**
I worked through section 7.
2 on data transformations, mainly centered around mapping on arrays.
We did also cover binnning and making dummy matrices, too.

**Day 18 - November 07, 2019:**
I completed the section of regular expressions - annoyingly, they are faily different than those used in R.
Also, the API for the re library in Python and the stringr package in R are flipped.
I also began the next chapter on data wrangling.
This one is focussed on manipulating the shapes of DataFrames.

**Day 19 - November 08, 2019:**
We covered merging and concatenating Series and DataFrames.
Thankfully, the logic is simmillar to that used in R by dplyr.

**Day 20 - November 09, 2019:**
I finished chapter on reshaping and pivoting data.
There is a lot of overlap with the functionality available in the dplyr and tidyr packages in R - they also share a lot of the same API.
I then began the chapter on matplotlib and visualization.
I am very excited for this section.

**Day 21 - November 10, 2019:**
I made good progress through the chapter on plotting, finishing the primer to matplotlib.
I decided to continue working through this chapter for a second hour today.
I got introduced to a few features of the seaborn library and the built-in plotting of pandas Series and DataFrame objects.

**Day 22 - November 11, 2019:**
Today, I learned how to group a pandas DataFrame or Series by the data in a column or subset of columns or group the columns by their data type.
It works generally the same as in 'dplyr' in R, but the API is a bit different.

**Day 23 - November 12, 2019:**
I finished the chapter on the split-apply-combine data aggregation pattern in pandas.
It fits well with the analogous operations in R and the tidyverse.

**Day 24 - November 13, 2019:**
I skipped over the chapter on Time Series and went to Advanced pandas as it is more relevant to my current work.
The chapter begins by teaching about the Categorical type in pandas.
It is analogous to the factor type in R and comes with performance enhancement for many pandas operations and functions.

**Day 25 - November 14, 2019:**
I finished up the (suprizingly short) chapter on advanced pandas by learning about tools for method chaining including the assign() method, the pipe() method, and callables.
I then began the chapter on statistical modeling in Python.

**Day 26 - November 15, 2019:**
I worked through the introduction the ML chapter where the author gave a brief introduction to patsy, statsmodels, and sklearn.
I think that is the end of my journey through Python for Data Analysis and I will move on to another educational source.

**Day 27 - November 16, 2019:**
I began working on Stanford's CS 20 Tensorflow course.
I get the environment set up and completed the introductory lecture.

**Day 28 - November 17, 2019:**
I learned about various TF data structures and how to create Variables.

**Day 29 - November 18, 2019:**
Because of all of the hiccups with TF 2.
0 in CS 20, I decided to first work through *Hands-On ML with Scikit-Learn and TF* to get the fundamentals of ML in Python.
I can then return to the CS 20 course to perhaps get a better theoretical understanding of TF and ML/DL.

**Day 30 - November 19, 2019:**
I lost a lot of time struggling with VSCode.
Eventually I switched over to just using jupyter in the browser and everything was much better.
I continued preparing the data for running the model.
We looked at correlations and visualized the data to help get an understanding of what features may be worth maintaining.

**Day 31 - November 20, 2019:**
I finsihed the data preparation for the housing data where I learned about using and creating Scikit-Learn transformers and pipelines.

**Day 32 - November 21, 2019:**
I finally trained a model using Scikit-Learn.
We tried using linear regression, a decision tree regressor, and a random forest regressor.
We also calculated the RMSE of the model on the training data via CV to access how well the model performed.

**Day 33 - November 22, 2019:**
I finished chapter 2 which was the first foray into the process of creating an ML model.
We used grid and randomized search to tune hyperparamters.

**Day 34 - November 24, 2019:**
I began working on the Classification chapter, specifically focusing on learning the common assessment metrics used.

**Day 35 - November 25, 2019:**
I finished the binary classification section and have started learning about multiclass classification and the evaluation methods and metrics commonly used.

**Day 36 - November 26, 2019:**
I finished the chapter on Classification by learning about multilabel classification.
The notebook has not been fully evaluated because it took too long and timed-out; I will try again tomorrow.

**Day 37 - November 27, 2019:**
We began the chapter that takes a deeper dive into training.
We are working on linear models at the moments, solving them using a closed formula for finding the minimal RMSE and using gradient descent.

**Day 38 - November 29, 2019:**
 I continued learning about different training methods, including stochastic gradient descent.

**Day 39 - November 29, 2019:**
I finished up the section on the different gradient descent algorithms by learning about mini-batch GD.
I took some time to make plots of the paths taken by the various gradient descent algorithms, too.

**Day 40 - November 30, 2019:**
I learned about fitting linear models and polynoial models to data.
I also took some time to focus on my Python programming using sklearn, numpy, and pandas to produce some cool plots.
We finished of today's hour by learning about the learning curve of a model.

**Day 41 - December 01, 2019:**
I finished the section on linear model regulatization and early stopping.
I also made a lot of interesting plots and practised using sklearn pipelines.

**Day 42 - December 02, 2019:**
I began the section on logistic regression.
I made a really cool plot where I got some good  practise using matplotlib and numpy.

**Day 43 - December 03, 2019:**
I finished the Linear Regression chapter by learning about logistic regression and multinomial logistic regression.
In the process, I got more practice with matplotlib.

**Day 44 - December 04, 2019:**
I began Chapter 5 on SVM.
This chapter may take me awhile to get through because there are some complex plots that I want to be able to make.
I'm looking forward to it!

**Day 45 - December 05, 2019:**
We continued with the chapter on SVM by learning about kernel models instead of increasing the feature space with polynomials.

**Day 46 - December 06, 2019:**
We continued on with SVM models and learned how to use the Guassian Radial Basis Function to compute measures of similarity as new features.

**Day 47 - December 07, 2019:**
I finished the section on SVMs by playing around with linear, polynomial, and kernelized methods.
I continued on to the next chapter covering decision trees in the evening.
There, I learned how to train and visualize a simple decision tree classifier.

**Day 48 - December 08, 2019:**
I learned and played around with the regularization parameters for a decision tree classifier.

**Day 49 - December 09, 2019:**
I learned about and experimented with decision tree regressors.
I did another few hours of playing with the regression trees in the evening.
Here, I demonstrated the sensitivty of decision trees classifiers to rotation (combatting it with PCA) and how different models can be obtainined from the same training data.

**Day 50 - December 10, 2019:**
I finished the section of decision trees by building a rudimentary random forest classifier as one of the exercises at the end of the chpater.
I then began working on Chapter 7 learning about Ensemble methods.

**Day 51 - December 11, 2019:**
I learned about bagging and pasting, using random subsamples of the training data to build diverse and robust models.
We specifically made an ensemble of decision trees that massively outperformed an individual tree.

**Day 52 - December 12, 2019:**
I learned about random patches and subspaces, methods that subsample the feature space for training the models of an ensemble, random forests, and extra-trees classifiers, random forests with trees trained using an algorithm that randomly selects feature thresholds instead of finding the optimal split.

**Day 53 - December 13, 2019:**
We covered Adaboost and Gradient Boosting for decision tree ensembles.
These are both methods of boosting, whereby each model in the ensemble is trained in order to improve upon the weaknesses of the predecessor.

**Day 54 - December 14, 2019:**
We continued learning about random forest ensembles by learning about turing boosting decision tree ensembles using the learning rate, depth of the trees, using early stopping to prevent overfitting.

**Day 55 - December 15, 2019:**
We learned about stacking, the process of using another model for the aggregation steps in an ensemble.
I am still expexperimenting and working on the exercises, but am almost done with the chapter.

**Day 56 - December 16, 2019:**
I am still working on implementing my own stacking classifier, but training the SVM takes a long time.
While it was training, I read the introduction to Chapter 8 on Dimensionality Reduction and learned about how to conduct PCA in Scikit-Learn.

**Day 57 - December 17, 2019:**
I finished creating my simple stacking classifier for the MNIST data to finish off chapter 7 on ensembles.
Without any real tuning, the Decision Tree model for the blender model out-performed both soft and hard voting on the test set.

**Day 58 - December 18, 2019:**
I finally got to the coding portion fro PCA.
I spent a bit more time on the plotting of the PCs because that is a skill I want to improve upon.
I returned in the evening and learned how I can determine the number of PCs to keep by keeping the top n PCs that perserve 90 % of the variance.
I also learned how PCA can be a form of compression by using the inverse transformation equation to project the reduced data into the original dimensions.

**Day 59 - December 19, 2019:**
I continued learning about using PCA and got some experience with Kernel PCA.
I experimented with the various hyperparameters and how to use Grid Search to optimize them as part of a regression (or classifier) pipeline.

**Day 60 - December 20, 2019:**
I learned about optimizing KPCA by finding the hyperparamters that reduce the error when the original dimensions are recreated.
I also learned about other dimensionality reduction techniques including LLE, random projections, MDS, Isomap, t-SNE, and LDA.

**Day 61 - December 21, 2019:**
I finished chapter 8 on dimensionality reduction by playing around with some of the other techniques including MDS, isomap, and t-SNE.
I also completed one of the exercises by demonstrating the improvement in a simple random forest at classifying the MNIST data after preprocessing the data with PCA.
I began the chapter on clustering with K-means.
Later in the day, I did a bit more work on K-means clustering and learned about various under the hood details and about optimizing $k$ using an elbow plot and the silhouette scores.

**Day 62 - December 22, 2019:**
I continued working with K-means clustering, creating silhouette diagrams and using it to segment the colors of an image.

**Day 63 - December 23, 2019:**
I learned about using K-Means clustering as a preprocessing step in a classification pipeline and in label-propagation for semi-supervised learning.

**Day 64 - December 24, 2019:**
I learned about DBSCAN and using it in conjunction with a classifier to make predictions.
I made examples of using it with KNN and AdaBoost RF models.

**Day 65 - December 25, 2019:**
We were briefly introduced to several other clustering algorithms including agglomerative clustering, BIRCH, mean-shift, affinity-propagation, and spectral clustering.
I did a second hour where I began working through the section on Gaussian Mixture models.

**Day 66 - December 26, 2019:**
I continued with the section on GMMs by learning about different ways of constraining the covariance matrix parameters, anomaly detection, and model selection using AIC and BIC.

**Day 67 - December 27, 2019:**
I finished up Ch 9 on clustering by learning about Bayesian Gaussian Mixture models.
I then began Part II of the book on NN and Deep Learning by learning about the origins of neural networks and perceptrons.

**Day 68 - December 28, 2019:**
I learned about multilayer perceptrons and the backpropagation algorithm.
I played around with the common activation functions used in artifical neurons.
Finally, I learned the basics of regression and classification with MLPs.
Soon, I will begin playing with Keras!

**Day 69 - December 29, 2019:**
I learned about the fundamentals of the Keras API and TF2's implementation.
I began making my first ANN and loading and preparing the MNIST Fashion data.
We built a MLP with 2 hidden layers.

**Day 70 - December 30, 2019:**
I learned how to use the Keras sequential API to build a classification and regression ANN.
In the evening, I began learning about Keras's functional API by creating a Wide and Deep model.

**Day 71 - December 31, 2019:**
I learned about creating and fitting ANNs with multiple inputs or outputs using the funational API of Keras.

**Day 72 - January 01, 2020:**
I finished learning about the main Keras APIs with a brief introduction to the Subclassing API.
I then learned about saving a loading models and using Callbacks.

**Day 73 - January 02, 2020:**
I learned about using Tensorboard today.
To do so, I also got some practive building ANNs using the Sequential API and saving the event history to a directory for TensorBoard.

**Day 74 - January 03, 2020:**
I practiced using Grid Search and Randomized Search with CV to tune a Keras model's hyperparameters.

**Day 75 - January 04, 2020:**
I finished the introduction chapter to ANNs with TF and Keras.
I have begun Chapter 11 on training DNNs.

**Day 76 - January 05, 2020:**
I learned about various activation functions including leaky ReLU and some variants.
We also covered Batch Normalization which I need to look further into to full understand.

**Day 77 - January 06, 2020:**
I learned about transfer learning today and how to accomplish it with Keras.

**Day 78 - January 07, 2020:**
We went over various optimizers to extend my tool-kit beyond SGD.
We covered momentum optimization, Nesterov accelerated gradient, RMSProp, Adam, and a few others.

**Day 79 - January 08, 2020:**
I learned about the theory and how to implement learning rate schedulers.
I also began the section on regularization.

**Day 80 - January 09, 2020:**
I learned about Dropout regularization and hoow to perform and use Monte Carlo Dropout to obtain between prediction probabilities.

**Day 81 - January 10, 2020:**
I began Ch 12 which provides a deeper dive into TF.
I am very excited for this chapter because it is a peak behind the curtain of TF and should provide a better understanding of what is happening when I `compile()` or `fit()` a model with the Keras API.
We began by learning about Tensors and Variables.

**Day 82 - January 11, 2020:**
I learned about creating custom Loss functions and how to save and load the model that employs them.
I ran into a bug that has been reported, so I will need to address that next.

**Day 83 - January 12, 2020:**
We learned about making other custom loss components and metrics, including streaming metrics.

**Day 84 - January 13, 2020:**
I learned how to create custom layers and models.

**Day 85 - January 14, 2020:**
We went over how to implement custom losses and metrics that use internal knowledge about the model instead of being limited to the predicted and known labels like before.
There seem to be a few bugs in TF at the moment with implementing the custom metric, but the loss appears to we working (I need to make a change that will be checked-in in another commit).

**Day 86 - January 15, 2020:**
I did some work on the Matplotlib tutorials and learned about computing gradients with autodiff in TF.

**Day 87 - January 16, 2020:**
I learned how to implement a custom training loop.
In my opinion, there are too many things that can go wrong to worry about this until I am very experienced in ML and TF2.
Still, it was a good look under the hood.

**Day 88 - January 17, 2020:**
I finished up the chapter of HOML about custom TF components and now have a better understadning of what is happening under the hood.
I then switched over to some Matplotlib tutorials and made some fun plots.

**Day 89 - January 18, 2020:**
I followed a brief tutorial on loading and displaying images in Matplotlib.
I followd a tutorial from Matplotlib's website on creating a full-featured plot from start to finish.

**Day 90 - January 19, 2020:**
I learned about style sheets for Matplotlib and how to customize a figure legend.

**Day 91 - January 20, 2020:**
I finished up the matplotlib tutorials by learning about cyclers.
I started the Plotly tutorials and learned about the fundamentals of creating an interactive plot.

**Day 92 - January 21, 2020:**
I learned about many of the commonly used functions of Plotly Express and how to make a reactive plot with widgets.

**Day 93 - January 22, 2020:**
I finished up the small interlude into plotting in Python by learning about reacting to click events in Plotly, and then touring around Seaborn.

**Day 94 - January 23, 2020:**
I began working through some of the tutorials offered for SciPy.

**Day 95 - January 24, 2020:**
I finished up with SciPy and then began learning about the statsmodels package.
I think this will be how I finish off my 100 Days of Python.

**Day 96 - January 25, 2020:**
I worked through the statsmodels tutorials and have a working-knowledge of using the library.

**Day 97 - January 26, 2020:**
I began Chapter 13 of HOML and began learning about the TF Data API.
It seems incredibly useful and efficient, so I'm looking forward to learning how to use it effectively.

**Day 98 - January 27, 2020:**
I learned about preprocessing data in a TF Dataset object and how to use it in training and validating a Keras ANN.

**Day 99 - January 28, 2020:**
I learned about TF Records and how to include preprocessing steps as layers in the ANN.

**Day 100 - January 29, 2020:**
I learned about embedding categorical variables in new dimensions and how to use an Embedding layer in a Keras model.

**Day 101 - January 30, 2020:**
I finished up Chapter 13 on handling data in TF.
We learned about the Preprocessing layers that will be coming to TF Keras shortly and the TensorFlow Datasets project.
I used TFDS to loading MNIST to train a small NN.

**Day 102 - January 31, 2020:**
I began chapter 14 on convolutional neural networks were we covered the basics, learning about convolutional layers and how they create and interpret feature maps.
We began loading example images, inspecting their shape, and applying filters (horizontal and vertical lines).

**Day 103 - February 01, 2020:**
I continued learning about the layers used in CNNs.
Specifcially, I learned about pooling layers, depthwise pooling layers, and global pooling layers.
Next, we start going through modern CNN architectures.

**Day 104 - February 03, 2020:**
I began learning about some of the most popular and important CNN architectures.
We learned about LeNet-5 and AlexNet.

**Day 105 - February 04, 2020:**
I finished reading and taking notes on the explanations of various powerful and important CNN architectures.

**Day 106 - February 06, 2020:**
We constructed a ResNet-34 model using a custom Residual Unit layer and the Keras Sequential API.
We also learned about using pretrained models from TensorFlow.
Finally, we breifly demonstrated a simple example of implementing transfer learning use a pretrained Xception model.

**Day 107 - February 10, 2020:**
I finished Chapter 14 on computer vision with deep models by learning about object localization, object detection, and semantic segmentation.

**Day 108 - February 11, 2020:**
I experimented with using a pretraining neural network for image segmentation, but that failed - I may give it another shot in the future.
I began the chapter on Recurrent Neural Networks (chapter 15).

**Day 109 - February 12, 2020:**
We covered the basics of RNNs and the different types based on their input and output shapes.
We began a simple example on mock training data.

**Day 110 - February 13, 2020:**
We created our first deep RNN.
Also, we learned about how to make predictions about the next N steps, not just the immediate next step.
The model was trained for just a few epochs, but performed very well.

**Day 111 - February 14, 2020:**
I implemented the sequence-to-seuqence model which was able to predict quite well the entire curve made from combining two sine waves.
We then learned about some of the problems with long sequences and implemented Layer Normalization in a custom memory cell to combat the exploding gradients.

**Day 112 - February 15, 2020:**
I learned about alternaties to the simple RNN such as the LSTM cell and GRU cell.
I also learned about using a convlutional layer with the RNN.

**Day 113 - February 17, 2020:**
I finished Chapter 15 on RNNs by learning about WaveNet, a CNN architecture that can learn long and short patterns very efficiently.
I began Chapter 17 (skipp 16 for now) on Autoencoders and Generative Adversarial Networks (GAN).

**Day 114 - February 18, 2020:**
I began coding Autoencoders today.
We began with a simple one that effectively mimicked PCA.
Then we built a Stacked AE that learned the Fashion MNIST data.

**Day 115 - February 19, 2020:**
We learned about using AE for dimensionality reduction and visualization and using it to pretrain a classification NN.
We then began learning about a few training methods for AE, starting with tying weights of symmetric AE.
