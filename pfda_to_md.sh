#!/bin/zsh

for notebook in PythonForDataAnalysis/*ipynb
do
    outputFileName=$(basename $notebook .ipynb).md
    jupyter nbconvert $notebook --to markdown
done

exit