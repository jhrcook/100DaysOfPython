#!/bin/zsh

for notebook in CS20TensorflowforDeepLearningResearch/my_notes/*ipynb
do
    outputFileName=$(basename $notebook .ipynb).md
    jupyter nbconvert $notebook --to markdown
done

exit
