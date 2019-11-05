#!/bin/zsh


## Get day number
echo "What number day is it?"
read dayNumber

re='^[0-9]+$'
if ! [[ $dayNumber =~ $re ]];
then
   echo "error: You did not enter an integer."
   exit 1
fi



## Build Python for Data Analysis notebooks
source activate daysOfCode-env
zsh pfda_to_md.zsh



## Write virtual environment to requirements.txt
conda list -e > requirements.txt
conda deactivate



## Add to README
currentDate=$(date +"%B %d, %Y")

echo "What did you learn today?"
read todaysMessage

readmeMessage="
**Day $dayNumber - $currentDate**
$(echo $todaysMessage | sed 's/\. */.\
/g')"
echo $readmeMessage >> README.md



## Git Commit
commitMessage="$dayNumber of 100 Days of Python
$(echo $todaysMessage | fold -w 80 -s)"

echo "\n---\nHere is todays commit message:"
echo $commitMessage
echo "\n"
echo "Shall I commit to git? (y/n)"
read shouldCommit
if [[ $shouldCommit == 'y' ]]
then
	echo "Comitting to git repo..."
	git add --all
	git commit -m $commitMessage
else
	echo "Nothing commited."
fi



exit