#!/bin/zsh


# Make the Markdown file for each iPython jupyter notebook in a directory.
ipynb_to_markdown () {
	for notebook in $1/*ipynb
	do
	    outputFileName=$(basename $notebook .ipynb).md
	    jupyter nbconvert $notebook --to markdown
	done
}


if [[ $1 == "pfda" ]];
then
	ipynb_to_markdown "PythonForDataAnalysis"
fi


if [[ $1 == "cs20" ]];
then
	ipynb_to_markdown "CS20TensorflowforDeepLearningResearch/my_notes"
fi


if [[ $1 == "homl" ]];
then
	ipynb_to_markdown "HandsOnMachineLearningWithScikitLearnAndTensorFlow"
fi
