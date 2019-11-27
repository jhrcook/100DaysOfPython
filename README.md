# 100 Days of Python

![100DaysOfCodePython](https://img.shields.io/badge/100DaysOfCode-Python-3776AB.svg?style=flat&logo=python)
[![jhc github](https://img.shields.io/badge/GitHub-jhrcook-lightgrey.svg?style=flat&logo=github)](https://github.com/jhrcook)
[![jhc twitter](https://img.shields.io/badge/Twitter-@JoshDoesA-00aced.svg?style=flat&logo=twitter)](https://twitter.com/JoshDoesa)
[![jhc website](https://img.shields.io/badge/Website-Joshua_Cook-5087B2.svg?style=flat&logo=telegram)](https://joshuacook.netlify.com)

**Start Date: October 21, 2019  
End Date: January 30, 2020**

## Sources

To begin, I am working through [
Python for Data Analysis, 2nd Edition](http://shop.oreilly.com/product/0636920050896.do).

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
