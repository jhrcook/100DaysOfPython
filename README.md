# 100 Days of Python

![100DaysOfCodePython](https://img.shields.io/badge/100DaysOfCode-Python-3776AB.svg?style=flat&logo=python)
[![jhc github](https://img.shields.io/badge/GitHub-jhrcook-lightgrey.svg?style=flat&logo=github)](https://github.com/jhrcook)
[![jhc twitter](https://img.shields.io/badge/Twitter-@JoshDoesA-00aced.svg?style=flat&logo=twitter)](https://twitter.com/JoshDoesa)
[![jhc website](https://img.shields.io/badge/Website-Joshua_Cook-5087B2.svg?style=flat&logo=telegram)](https://joshuacook.netlify.com)

**Start Date: October 21, 2019  
End Date: January 29, 2020**

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

**Day 15 - November 04, 2019**
I learned about reading in HTML, XML, HDF5, and Excel data.

**Day 16 - November 05, 2019**
We breifly covered interacting with web APIs to finish up the data reading and writing chapter.
I then began Chapter 7 on data cleaning and prepraration where we have gone over dropping and filling in missing values.

**Day 17 - November 06, 2019**
I worked through section 7.
2 on data transformations, mainly centered around mapping on arrays.
We did also cover binnning and making dummy matrices, too.

**Day 18 - November 07, 2019**
I completed the section of regular expressions - annoyingly, they are faily different than those used in R.
Also, the API for the re library in Python and the stringr package in R are flipped.
I also began the next chapter on data wrangling.
This one is focussed on manipulating the shapes of DataFrames.
